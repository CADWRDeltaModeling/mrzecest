"""Estimate the Martinez EC model using the
conditional optimization approach in the 2001 Annual Report Chapter 11.
An outer, generic optimization searches over beta (time scale)
parameter of gmodel and npow, the power "npow" sometimes called
a "shape factor". The npow has been generalized to include an npow
and and npow_tide for the part that multiplies the tide.
Conditional on those variables, the rest of the fit can
be performed using Generalized Linearized Models,
which are appropriate given the log link and non-normal errors.

As a side note, the fit will by definition always be worse than
least squares if the basis of comparison is also squared squared
error. The point in the 2001 article is that this is a poor basis
of comparison for this variable.

"""

import pandas as pd
import numpy as np
from vtools import *
import matplotlib.pyplot as plt
from mrzecest.fitting_util import parse_config, validate_fit_config, validate_model
from mrzecest.ec_boundary import (
    gcalc,
    ndo_mod,
    ec_kernel,
    _sigmoid,
    _g_threshold_from_ec,
    _front_weight,
    build_common_features,
)
import statsmodels.api as sm
from statsmodels.genmod.families.links import Log
from statsmodels.genmod.families import Gamma
import scipy
import scipy.optimize
import logging
logger = logging.getLogger(__name__)


calls = 0


def _fit_glm_given_g(g, data, npow, sb, so, *, fix_b0: bool, b0_value: float, g_thr_tide: float, width_tide: float, weight_power: float = 0.0):
    """Fit inner GLM conditional on g (and data z-columns).

    Returns
    -------
    result : statsmodels GLMResults
    X_clean : pandas.DataFrame
    y_clean : pandas.Series
    preds : pandas.DataFrame  (without intercept)
    params : pandas.Series (always includes 'const')
    """
    family = sm.families.Gamma(Log())
    gnpow = g.pow(npow)
    # Bounded logistic tidal gate W(g) in [0,1]; replaces the unbounded g**npow_tide.
    gpow_tide = pd.Series(
        _sigmoid((g.to_numpy(dtype=float) - g_thr_tide) / width_tide), index=g.index
    )

    y_ec = data.ec_obs
    y_ec = y_ec.clip(lower=sb + 1.0)
    eps = 1e-6
    y = ((y_ec - sb) / (so - sb)).clip(lower=eps)

    zcols = [col for col in data.columns if col.lower().startswith("z")]
    main = 0.001 * gnpow
    lag = 0.001 * data.loc[:, zcols].mul(gpow_tide, axis=0)

    preds = pd.concat((main, lag), axis=1)
    preds.columns = ["gnpow"] + zcols

    if not g.index.equals(data.index):
        extra_in_data = data.index.difference(g.index)
        extra_in_g = g.index.difference(data.index)
        raise ValueError(
            f"g.index != data.index: "
            f"data_minus_g={len(extra_in_data)} g_minus_data={len(extra_in_g)} "
            f"first_data_minus_g={extra_in_data[0] if len(extra_in_data) else None}"
        )
    if fix_b0:
        X = preds  # no intercept column when b0 is fixed
        offset = pd.Series(float(b0_value), index=X.index)
    else:
        X = sm.add_constant(preds)
        offset = None
    Xv = X.to_numpy(dtype=float)

    if not np.isfinite(Xv).all():
        bad_row = int(np.where(~np.isfinite(Xv).all(axis=1))[0][0])
        t = X.index[bad_row]
        bad_cols = [X.columns[j] for j in np.where(~np.isfinite(Xv[bad_row]))[0]]

        # g should be a Series aligned to X.index
        gv = float(g.loc[t])

        # (optional) compute power for context; may be nan if gv invalid
        try:
            gtp = float(gv**npow)
        except Exception:
            gtp = np.nan

        raise ValueError(
            f"exog (X) contains nonfinite at {t}: cols={bad_cols}\n"
            f"  g={gv}\n"
            f"  g**npow={gtp}\n"
            f"  npow={npow}\n"
        )

    # 2) Allow y missing: drop only rows where y is NaN (or nonfinite)
    yv = y.to_numpy(dtype=float)
    y_ok = np.isfinite(
        yv
    )  # this treats NaN as missing; also drops inf if it ever occurred

    if not y_ok.any():
        raise ValueError("All y are missing/nonfinite; cannot fit.")

    X_fit = X.loc[y_ok]
    y_fit = y.loc[y_ok]

    if offset is not None:
        offset_fit = offset.loc[y_ok]
    else:
        offset_fit = None

    # High-EC weighting: var_weights = y**weight_power up-weights high-salinity points
    # so the (relative-error) Gamma deviance stops discounting the high-EC amplitude.
    # weight_power = 0 -> all weights 1 (unweighted, original behavior).
    var_weights_fit = np.power(y_fit.to_numpy(dtype=float), float(weight_power))

    mod = sm.GLM(y_fit, X_fit, family=family, offset=offset_fit, var_weights=var_weights_fit)
    result = mod.fit()
    # Guard against pathological fits that technically return but are unusable.
    if not np.all(np.isfinite(result.params.to_numpy(dtype=float))):
        raise ValueError("Non-finite GLM params (overflow/underflow in IRLS)")


    # Always synthesize 'const' so downstream code doensn't have to work
    params = result.params.copy()
    if fix_b0:
        # result.params has no 'const' in this mode; add it explicitly.
        params = pd.concat([pd.Series({"const": float(b0_value)}), params])

    return result, X_fit, y_fit, preds, params


def outer_fit(x, data, return_coefs=False, plot_fits=False):
    """
    Fit outer model parameters to optimize energy flux calculations using GLM.
    This function serves as the objective function for an optimization routine that
    calibrates Martinez estimates given outflow and tidal data.
    Parameters
    ----------
    x : array-like
        Optimizer vector containing scaled outer parameter values. Maps to keys
        defined in fit_spec.outer_params with indices corresponding to parameter
        order.
    data : xarray.Dataset
        Input data containing:
        - ndo : observed normalized dissipation
        - d_elev_filt : filtered elevation changes
        - energy : energy values
        - ec_obs : observed energy flux
        - attrs["fit_spec"] : dict with fit configuration including:
            - outer_params : list of dicts with keys "key" and "scale"
            - sb : spring-neap modulation factor
            - so : stratification coefficient (optional in outer_params)
    return_coefs : bool, optional
        If True, return fitted coefficients and predictions alongside objective value.
        Default is False.
    plot_fits : bool, optional
        If True, generate diagnostic plots when call count exceeds thresholds.
        Default is False.
    Returns
    -------
    float or tuple
        If return_coefs is False:
            deviance : float
                GLM deviance (objective to minimize)
        If return_coefs is True:
            tuple of (x, params, ypred)
            - x : array-like
                Input optimizer vector
            - params : statsmodels parameter series
                Fitted GLM coefficients
            - ypred : DataFrame
                Observed vs fitted energy flux values
    Raises
    ------
    ValueError
        If fit_spec is missing from data.attrs, outer_params is empty,
        required parameters missing from outer_params, or if g_base contains
        non-finite or negative values.

    """
    FAIL = 1.0e99
    use_ols = False
    global calls
    calls = calls + 1
    logger.debug("Entering outer fit with x = %s", x)
    fit_spec = data.attrs.get("fit_spec", None)
    if fit_spec is None:
        raise ValueError(
            "fit_spec missing from data.attrs; cannot interpret outer parameters"
        )

    outer_params = fit_spec.get("outer_params")
    if not outer_params:
        raise ValueError("fit_spec.outer_params is empty")

    # Map optimizer vector -> canonical model parameter values.
    pvals = {}
    for i, p in enumerate(outer_params):
        key = p.get("key")
        if not key:
            raise ValueError(f"outer_params[{i}] is missing a 'key' field")

        scale = float(p.get("scale", 1.0))
        pvals[key] = float(x[i]) * scale

    # Required
    try:
        log10beta = pvals["beta_log10"]
        npow = pvals["npow"]
        area_coef = pvals["area_coef"]
        energy_coef = pvals["energy_coef"]
    except KeyError as e:
        raise ValueError(
            "fit_run.outer_params must include beta_log10, npow, "
            "area_coef, energy_coef; missing {e}"
        )

    # Optional new knobs
    fit_spec = data.attrs["fit_spec"]
    sb = float(fit_spec["sb"])
    so_config = float(fit_spec["so"])

    # If 'so' is present in outer_params, use it; otherwise fall back
    so = float(pvals.get("so", so_config))

    # Inner intercept handling (optional fixed b0)
    fix_b0 = bool(fit_spec.get("fix_b0", False))
    b0_value = float(fit_spec.get("b0_value", 0.0))
    weight_power = float(fit_spec.get("weight_power", 0.0))

    # Bounded tidal gate: transport threshold (cfs) and fractional width.
    g_thr_tide = float(pvals["g_thr_tide"])
    width_frac_tide = float(pvals.get("width_frac_tide", fit_spec.get("width_frac_tide", 0.6)))
    width_tide = max(1e-6, width_frac_tide * g_thr_tide)

    ndo = data.ndo
    ec_obs = data.ec_obs
    g0 = 1000.0
    # Base pass: ndomod uses only area correction (energy_coef repurposed elsewhere)
    ndomod_base = ndo_mod(
        data.ndo, data.d_elev_filt, area_coef, data.energy, energy_coef
    ).squeeze()
    g_base = gcalc(ndomod_base, log10beta=log10beta, g0=g0)

    # Check for NaNs or negative values in g_base
    gb = g_base.squeeze().astype(float)
    bad = ~np.isfinite(gb.to_numpy())
    has_negative = (gb < 0).any()

    if bad.any() or has_negative:
        nbad = int(bad.sum())
        first_t = gb.index[np.where(bad)[0][0]] if bad.any() else None
        # cheap context
        logger.debug(
            "ndomod_base min/max: %s %s",
            float(np.nanmin(ndomod_base.values)),
            float(np.nanmax(ndomod_base.values)),
        )
        logger.debug(
            "g_base finite? %s neg? %s first_bad: %s",
            not bad.any(),
            has_negative,
            first_t,
        )
        raise ValueError(
            f"g_base nonfinite or negative: nonfinite={nbad}, negative={has_negative}"
        )

    # First-pass GLM (conditional on g_base) to get beta0/beta1 for EC->g threshold
    try:
        result0, X0, y0, preds0, params0 = _fit_glm_given_g(
            g_base, data, npow=npow, sb=sb, so=so,
            fix_b0=fix_b0, b0_value=b0_value, g_thr_tide=g_thr_tide, width_tide=width_tide,
            weight_power=weight_power
        )
    except Exception as e:
        # Infeasible point in outer optimizer: return a huge penalty.
        if return_coefs:
            raise
        logger.exception("GLM failed (base g) at x=%s: %s: %s", x, type(e).__name__, e)
        return FAIL
    beta0_0 = float(params0["const"])
    beta1_0 = 0.001 * float(result0.params["gnpow"])

    # Soft g threshold corresponding to EC ~= 1000 (mean; z_sum=0). No upper threshold.
    gthr = _g_threshold_from_ec(
        20000.0, beta0=beta0_0, beta1=beta1_0, npow=npow, so=so, sb=sb
    )
    w_front = _front_weight(g_base, gthr, width_frac=0.10)

    # Low-energy weight (large at low energy, small at high energy)
    energy_ref = float(np.nanmedian(data.energy.squeeze().astype(float).values))
    if not np.isfinite(energy_ref) or energy_ref <= 0.0:
        raise ValueError(
            "Invalid energy_ref derived from data.energy (must be finite and > 0)."
        )
    low_energy_weight = 1.0 / (1.0 + (data.energy.squeeze().astype(float) / energy_ref))

    # Repurposed energy_coef: amplitude (in ndomod units) of stratification term
    strat_term = energy_coef * w_front * low_energy_weight

    ndomod = (ndomod_base - strat_term).rename("ndo")
    g = gcalc(ndomod, log10beta=log10beta, g0=g0)

    # Final inner fit using the updated g
    try:
        result, X_clean, y_clean, preds, params = _fit_glm_given_g(
            g, data, npow=npow, sb=sb, so=so,
            fix_b0=fix_b0, b0_value=b0_value, g_thr_tide=g_thr_tide, width_tide=width_tide,
            weight_power=weight_power
        )
    except Exception as e:
        if return_coefs:
            raise
        logger.exception("GLM failed at x=%s: %s: %s", x, type(e).__name__, e)
        return FAIL

    logger.debug("NaN check found null: %s", preds.isnull().any(axis=None))
    logger.debug("Condition Number of design matrix: %s", np.linalg.cond(X_clean))
    if (calls > 600000) and plot_fits:
        fig, ax = plt.subplots(1)
        ax.plot(preds.index, preds.values)
        ax.legend(preds.columns)
        plt.show()

    logger.debug("GLM result summary:\n%s", result.summary())  # this has the coefficients that I require

    predictions = result.fittedvalues
    rss = np.sum((y_clean - predictions) ** 2)
    rmse = np.sqrt(rss / len(y_clean))

    ypred = y_clean.copy().to_frame()
    ypred.columns = ["data"]
    ypred["fit"] = np.nan
    ypred.loc[:, "fit"] = predictions

    if ((calls > 1000) | return_coefs) and plot_fits:
        fig, ax = plt.subplots(1)
        ax.plot(ypred.index, ypred.values)
        ax.legend(["data", "fit"])
        plt.show()

    if use_ols:
        out = float(rss)
        logger.debug("RSS %s", out)
    else:
        # GLM objective: deviance (lower is better)
        out = float(result.deviance)
        logger.debug("RMSE %s RSS %s deviance %s", rmse, rss, out)

    # ---- Freeze regime/gating constants for portability (no window stats at eval) ----
    ec_target = 20000.0
    width_frac = 0.10
    # gthr should be computed from final params (no z-term) and ec_target
    # NOTE: the GLM uses a design column scaled by 0.001 (see _fit_glm_given_g).
    # The effective beta1 used in the EC kernel is therefore 0.001 * params['gnpow'].
    gthr = _g_threshold_from_ec(
        ec_target, float(params["const"]), 0.001 * float(params["gnpow"]), npow, so, sb
    )
    # energy_ref must be computed ONCE from the fit window (not recomputed at eval)
    energy_ref = float(np.nanmedian(data["energy"].to_numpy(dtype=float)))
    front_spec = {
        "ec_target": float(ec_target),
        "gthr": float(gthr),
        "energy_ref": float(energy_ref),
        "width_frac": float(width_frac),
    }

    if return_coefs:
        return x, params, ypred, front_spec
    else:
        return out


def fit_mrz_ecest(config, elev=None, ndo=None, ec_obs=None, plot_fits=False):
    """Main fitting routine for Martinez EC boundary estimation.

    This function orchestrates the outer optimization loop, which searches over
    beta (log10 scale), npow (shape factor), area_coef, and energy_coef parameters.
    For each candidate outer parameter set, it calls outer_fit() to perform the
    inner GLM fit and compute the objective (deviance).

    Parameters
    ----------
    config : str or dict
        Configuration mapping with keys: filter_setup, so, sb, fit_run, outer_params.
    elev : pandas.Series, optional
        Elevation time series.
    ndo : pandas.Series, optional
        Normalized dissolved oxygen time series.
    ec_obs : pandas.Series, optional
        Observed EC time series.
    plot_fits : bool, default False
        If True, plot intermediate and final fits (when call count exceeds threshold).

    Returns
    -------
    x_res : numpy.ndarray
        Optimal outer parameter vector.
    coefs : pandas.Series
        GLM coefficients (const, gnpow, z0, z1, ...).
    ypred : pandas.DataFrame
        DataFrame with columns ["data", "fit"] showing observed vs predicted EC.

    Notes
    -----
    The outer optimization via scipy.optimize.minimize will repeatedly call
    outer_fit() with different x values until convergence. outer_fit() then
    fits the inner GLM conditional on the computed g values.
    """
    logger.debug("Entering fit routine")
    if isinstance(config, str):
        config = parse_config(config)

    # No legacy/backward constraints: validate and fail fast.
    validate_fit_config(config)

    fit_run = config.get("fit_run")
    if not isinstance(fit_run, dict):
        raise ValueError("fit_run is required and must be a mapping")

    fit_start_key = fit_run.get("start")
    fit_end_key = fit_run.get("end")
    if fit_start_key is None or fit_end_key is None:
        raise ValueError("fit_run must include 'start' and 'end'")
    start = pd.to_datetime(fit_start_key)
    end = pd.to_datetime(fit_end_key)
    solver = str(fit_run.get("solver", "powell")).lower()

   # Inner fixed-intercept (optional, strict)
    inner = fit_run.get("inner", None)
    fix_b0 = False
    b0_value = 0.0
    weight_power = 0.0
    if inner is not None:
        if not isinstance(inner, dict):
            raise ValueError("fit_run.inner must be a mapping if provided")
        b0spec = inner.get("b0", None)
        if b0spec is not None:
            if not isinstance(b0spec, dict):
                raise ValueError("fit_run.inner.b0 must be a mapping if provided")
            if "fix" not in b0spec or "value" not in b0spec:
                raise ValueError("fit_run.inner.b0 must include keys: fix, value")
            fix_b0 = bool(b0spec["fix"])
            b0_value = float(b0spec["value"])
        weight_power = float(inner.get("weight_power", 0.0))

    # Outer parameters: beta_log10, npow, area_coef, energy_coef, so (optional)
    outer_params = fit_run.get("outer_params")
    if not isinstance(outer_params, (list, tuple)) or len(outer_params) == 0:
        raise ValueError(
            "fit_run.outer_params is required and must be a non-empty list"
        )


    # Build x0/bounds from outer_params. Keys must match canonical model param names.
    x0 = []
    bounds = []
    any_finite_bounds = False
    for i, p in enumerate(outer_params):
        if not isinstance(p, dict):
            raise TypeError(f"outer_params[{i}] must be a mapping, got {type(p)}")
        key = p.get("key")
        if not key:
            raise ValueError(f"outer_params[{i}] missing 'key'")
        if "x0" not in p:
            raise ValueError(f"outer_params[{i}] ({key}) missing 'x0'")
        if "bounds" not in p:
            raise ValueError(f"outer_params[{i}] ({key}) missing 'bounds'")
        x0_i = float(p["x0"])
        b = p["bounds"]
        if b is None or (
            isinstance(b, (list, tuple))
            and len(b) == 2
            and b[0] is None
            and b[1] is None
        ):
            bnd = (None, None)
        else:
            if not (isinstance(b, (list, tuple)) and len(b) == 2):
                raise ValueError(
                    f"outer_params[{i}] ({key}) bounds must be [lo, hi] or [null, null]"
                )
            lo = None if b[0] is None else float(b[0])
            hi = None if b[1] is None else float(b[1])
            bnd = (lo, hi)
            if lo is not None or hi is not None:
                any_finite_bounds = True
        # Fail fast if x0 outside finite bounds
        lo, hi = bnd
        if lo is not None and x0_i < lo:
            raise ValueError(f"outer_params[{i}] ({key}) x0 {x0_i} < lower bound {lo}")
        if hi is not None and x0_i > hi:
            raise ValueError(f"outer_params[{i}] ({key}) x0 {x0_i} > upper bound {hi}")
        x0.append(x0_i)
        bounds.append(bnd)

    # ------------------------------------------------------------
    # Common feature construction (single source of truth)
    # ------------------------------------------------------------
    ctx = build_common_features(
        ndo=ndo,
        elev=elev,
        filter_setup=config["filter_setup"],
        pad=pd.Timedelta("9d"),
        start=start,
        end=end,
    )

    eval_index = ctx["eval_index"]

    solu_df = pd.concat(
        (
            ec_obs.loc[eval_index],
            ctx["ndo"],  # ndo sliced to eval_index by build_common_features()
            ctx["elev_filt"],
            ctx["elev_tidal"],
            ctx["d_elev_filt"],
            ctx["energy"],
            ctx["Z"],  # DataFrame with z0..z{filter_length-1}
        ),
        axis=1,
    )

    solu_df.columns = [
        "ec_obs",
        "ndo",
        "elev_filt",
        "elev_tidal",
        "d_elev_filt",
        "energy",
    ] + list(ctx["Z"].columns)

    expected_z = int(config["filter_setup"]["filter_length"])
    zcols = [c for c in solu_df.columns if c.lower().startswith("z")]
    if len(zcols) != expected_z:
        raise ValueError(
            f"Expected {expected_z} z columns, got {len(zcols)}: {zcols[:5]}..."
        )

    # Attach fit spec (constants + parameter interpretation) to the data frame.
    solu_df.attrs["fit_spec"] = {
        "so": float(config["so"]),
        "sb": float(config["sb"]),
        "outer_params": outer_params,
        "fix_b0": bool(fix_b0),
        "b0_value": float(b0_value),
        "width_frac_tide": float(config.get("width_frac_tide", 0.6)),
        "weight_power": float(weight_power),
    }
    # solu_df.to_csv("test.csv", index=True, header=True, float_format="%.3f")

    tol = float(fit_run.get("tol", 5e-3))
    res = scipy.optimize.minimize(
        outer_fit,
        x0,
        args=(solu_df,),
        tol=tol,
        method=(
            "Nelder-Mead"
            if solver in ("nelder-mead", "nelder_mead", "neldermead")
            else solver
        ),
        bounds=bounds,
    )
    logger.info("Optimizer finished: success=%s message=%s", res.success, res.message)

    x_res, coefs, ypred, front_spec = outer_fit(res.x, solu_df, return_coefs=True, plot_fits=plot_fits)

    # Reconstruct canonical parameter values using outer_parasms + scale
    pvals = {}
    for i, p in enumerate(outer_params):
        key = p.get("key")
        if not key:
            raise ValueError(f"outer_params[{i}] missing 'key'")
        scale = float(p.get("scale", 1.0))
        pvals[key] = float(x_res[i]) * scale

    summary_items = [
        ("beta_log10", pvals.get("beta_log10")),
        ("npow", pvals.get("npow")),
        ("g_thr_tide", pvals.get("g_thr_tide")),
        ("area_coef", pvals.get("area_coef")),
        ("energy_coef", pvals.get("energy_coef")),
        ("so", pvals.get("so")),
    ]
    logger.info("%s", " ".join(f"{k}={round(v, 3)}" for k, v in summary_items if v is not None))

    logger.info("beta0 = %s", coefs["const"])
    logger.info("beta1 = 0.001*%s", coefs["gnpow"])
    logger.info(
        "z coefs = 0.001*%s",
        [round(zval, 3) for zval in coefs[coefs.index.str.startswith("z")].values],
    )

    return x_res, coefs, ypred, front_spec


# Canonical order of the flat parameter vector used by the LS touch-up.
_TOUCHUP_OUTER_KEYS = (
    "beta_log10",
    "npow",
    "g_thr_tide",
    "area_coef",
    "energy_coef",
    "so",
)


def _touchup_bounds(config, model, fix_b0=False):
    """Build (lo, hi) physical-space bounds for the flat touch-up vector.

    Outer parameters inherit bounds from ``fit_run.outer_params`` (expressed in
    physical units by multiplying the scaled bounds by ``scale``). Inner linear
    coefficients (b0, b1, afilt) are unbounded. When ``fix_b0`` is True, b0 is
    held out of the optimized vector entirely.
    """
    cfg = config if isinstance(config, dict) else parse_config(config)
    outer_params = (cfg.get("fit_run") or {}).get("outer_params") or []

    phys_bounds = {}
    for p in outer_params:
        key = p.get("key")
        if not key:
            continue
        scale = float(p.get("scale", 1.0))
        b = p.get("bounds")
        if not (isinstance(b, (list, tuple)) and len(b) == 2):
            phys_bounds[key] = (-np.inf, np.inf)
            continue
        lo = -np.inf if b[0] is None else float(b[0]) * scale
        hi = np.inf if b[1] is None else float(b[1]) * scale
        phys_bounds[key] = (lo, hi)

    lo_vec = []
    hi_vec = []
    for key in _TOUCHUP_OUTER_KEYS:
        lo, hi = phys_bounds.get(key, (-np.inf, np.inf))
        lo_vec.append(lo)
        hi_vec.append(hi)

    # Inner coefficients: (b0,) b1, afilt[k] -> unbounded. b0 omitted if fixed.
    n_inner = (1 if fix_b0 else 2) + len(model["afilt"])
    lo_vec.extend([-np.inf] * n_inner)
    hi_vec.extend([np.inf] * n_inner)

    return np.asarray(lo_vec, dtype=float), np.asarray(hi_vec, dtype=float)


def _pack_touchup(model, fix_b0=False):
    """Pack a canonical model dict into the flat LS vector (physical units).

    When ``fix_b0`` is True, b0 is omitted (held fixed outside the optimizer).
    """
    outer = [float(model[k]) for k in _TOUCHUP_OUTER_KEYS]
    inner = [] if fix_b0 else [float(model["b0"])]
    inner.append(float(model["b1"]))
    inner.extend(float(a) for a in model["afilt"])
    return np.asarray(outer + inner, dtype=float)


def _unpack_touchup(vec, model, fix_b0=False, b0_value=0.0):
    """Unpack the flat LS vector back into an updated model dict.

    Only the optimized fields are replaced; frozen structure (filter_setup,
    front, g0, sb) is carried over from ``model``. When ``fix_b0`` is True, b0
    is not read from ``vec`` and is set to ``b0_value``.
    """
    vec = np.asarray(vec, dtype=float)
    n_outer = len(_TOUCHUP_OUTER_KEYS)
    n_afilt = len(model["afilt"])
    out = dict(model)
    for i, key in enumerate(_TOUCHUP_OUTER_KEYS):
        out[key] = float(vec[i])
    if fix_b0:
        out["b0"] = float(b0_value)
        b1_idx = n_outer
    else:
        out["b0"] = float(vec[n_outer])
        b1_idx = n_outer + 1
    out["b1"] = float(vec[b1_idx])
    out["afilt"] = [float(a) for a in vec[b1_idx + 1 : b1_idx + 1 + n_afilt]]
    # front/filter_setup/g0/sb are inherited unchanged (frozen).
    out["front"] = dict(model["front"])
    out["filter_setup"] = dict(model["filter_setup"])
    return out


def touchup_least_squares(config, model, elev=None, ndo=None, ec_obs=None):
    """Least-squares 'touch-up' of a fitted model, seeded at the GLM solution.

    This performs a *local* Levenberg-Marquardt/TRF minimization of the plain
    sum-of-squared residuals ``(EC_pred - EC_obs)`` in EC units, over all model
    parameters simultaneously (the six outer/nonlinear parameters plus the inner
    linear kernel coefficients b0, b1, and the lagged tidal coefficients afilt).

    The stratification gating geometry from the baseline fit
    (``model['front']`` : gthr, energy_ref, width_frac) is **frozen**, so the
    touch-up is a well-defined local refinement around the original outcome
    (up to bounds-hugging). Features are built once and reused every iteration.

    Parameters
    ----------
    config : str or dict
        Fitting configuration (used only for the outer-parameter bounds).
    model : dict
        Canonical model dict from the baseline fit (e.g. build_model_from_fit()).
        Serves as the optimizer start point x0.
    elev, ndo, ec_obs : pandas.Series
        Elevation, Net Delta Outflow, and observed EC. The fit window is taken
        from ``config['fit_run']`` start/end.

    Returns
    -------
    model_ls : dict
        Refined canonical model dict (validated).
    result : scipy.optimize.OptimizeResult
        The least_squares result object.
    """
    cfg = config if isinstance(config, dict) else parse_config(config)
    validate_fit_config(cfg)

    fit_run = cfg.get("fit_run") or {}
    start = pd.to_datetime(fit_run["start"])
    end = pd.to_datetime(fit_run["end"])

    # Honor the baseline inner intercept convention: if b0 is fixed in the
    # fitting config, hold it out of the LS vector too (apples-to-apples with
    # the GLM fit rather than silently adding a degree of freedom).
    inner_spec = fit_run.get("inner") or {}
    b0spec = inner_spec.get("b0") or {}
    fix_b0 = bool(b0spec.get("fix", False))
    b0_value = float(b0spec.get("value", 0.0))

    sb = float(model["sb"])
    g0 = float(model.get("g0", 5000.0))
    width_frac_tide = float(model.get("width_frac_tide", 0.6))
    gthr = float(model["front"]["gthr"])
    energy_ref = float(model["front"]["energy_ref"])
    width_frac = float(model["front"]["width_frac"])

    # Build all shared features exactly once on the fit window.
    ctx = build_common_features(
        ndo=ndo,
        elev=elev,
        filter_setup=cfg["filter_setup"],
        pad=pd.Timedelta("9d"),
        start=start,
        end=end,
    )
    eval_index = ctx["eval_index"]
    ndo_eval = ctx["ndo"]
    d_elev_filt = ctx["d_elev_filt"]
    energy = ctx["energy"].squeeze().astype(float)
    Z = ctx["Z"].to_numpy(dtype=float)

    obs = ec_obs.loc[eval_index].squeeze().astype(float).to_numpy()

    # Static residual mask: finite observations, excluding row 0 (ec_kernel
    # sets EC[0]=NaN by construction). Fixed across iterations so the residual
    # vector length is constant.
    mask = np.isfinite(obs)
    mask[0] = False
    if not mask.any():
        raise ValueError("No finite observations on the fit window for LS touch-up.")
    obs_masked = obs[mask]

    low_energy_weight = 1.0 / (1.0 + (energy / energy_ref))
    lew = low_energy_weight.to_numpy(dtype=float)

    n_afilt = len(model["afilt"])
    n_outer = len(_TOUCHUP_OUTER_KEYS)
    PENALTY = 1.0e6

    def residual(vec):
        (beta_log10, npow, g_thr_tide, area_coef, energy_coef, so) = vec[:n_outer]
        width_tide = max(1e-6, width_frac_tide * g_thr_tide)
        if fix_b0:
            b0 = b0_value
            b1 = vec[n_outer]
            afilt = vec[n_outer + 1 : n_outer + 1 + n_afilt]
        else:
            b0 = vec[n_outer]
            b1 = vec[n_outer + 1]
            afilt = vec[n_outer + 2 : n_outer + 2 + n_afilt]

        ndomod_base = ndo_mod(ndo_eval, d_elev_filt, area_coef, energy, energy_coef).squeeze()
        g_base = gcalc(ndomod_base, log10beta=beta_log10, g0=g0)
        w_front = _front_weight(g_base, gthr, width_frac=width_frac)
        strat_term = energy_coef * w_front.to_numpy(dtype=float) * lew
        ndomod = (ndomod_base.to_numpy(dtype=float) - strat_term)
        ndomod = pd.Series(ndomod, index=eval_index, name="ndo")
        g = gcalc(ndomod, log10beta=beta_log10, g0=g0)

        z_sum = Z @ np.asarray(afilt, dtype=float)
        ec = ec_kernel(g.to_numpy(dtype=float), z_sum, b0, b1, npow, so, sb, g_thr_tide, width_tide)

        res = ec[mask] - obs_masked
        # Guard: replace any nonfinite residual (e.g. transient negative g under
        # fractional power) with a large finite penalty so TRF steers away.
        bad = ~np.isfinite(res)
        if bad.any():
            res = res.copy()
            res[bad] = PENALTY
        return res

    x0 = _pack_touchup(model, fix_b0=fix_b0)
    lo, hi = _touchup_bounds(cfg, model, fix_b0=fix_b0)
    # Keep the start point strictly inside the bounds for TRF.
    x0 = np.clip(x0, lo, hi)

    result = scipy.optimize.least_squares(
        residual,
        x0,
        bounds=(lo, hi),
        method="trf",
        x_scale="jac",
    )
    logger.info(
        "LS touch-up finished: success=%s status=%s cost=%.6g",
        result.success,
        result.status,
        float(result.cost),
    )

    model_ls = _unpack_touchup(result.x, model, fix_b0=fix_b0, b0_value=b0_value)
    validate_model(model_ls)
    return model_ls, result
