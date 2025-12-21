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
from mrzecest.fitting_util import parse_config, validate_fit_config
from mrzecest.ec_boundary import (
    gcalc,
    ndo_mod,
    _g_threshold_from_ec,
    _front_weight,
    build_common_features,
)
import statsmodels.api as sm
from statsmodels.genmod.families.links import Log
from statsmodels.genmod.families import Gamma
import scipy
import logging
logger = logging.getLogger(__name__)


calls = 0


def _fit_glm_given_g(g, data, npow, npow_tide, sb, so, *, fix_b0: bool, b0_value: float):
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
    gpow_tide = g.pow(npow_tide)

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

    mod = sm.GLM(y_fit, X_fit, family=family, offset=offset_fit)
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


    # npow_tide can be different from npow; if not provided, default == npow
    npow_tide = float(pvals.get("npow_tide", npow))

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
            g_base, data, npow=npow, npow_tide=npow_tide, sb=sb, so=so,
            fix_b0=fix_b0, b0_value=b0_value
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
            g, data, npow=npow, npow_tide=npow_tide, sb=sb, so=so,
            fix_b0=fix_b0, b0_value=b0_value
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
        ("npow_tide", pvals.get("npow_tide")),
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
