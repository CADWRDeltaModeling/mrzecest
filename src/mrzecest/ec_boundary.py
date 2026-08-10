"""Calculate g-model based estimate of Martinez boundary."""

import yaml
import pandas as pd
import numpy as np
import datetime as dt
from vtools import *
import logging
logger = logging.getLogger(__name__)
import os

import numba

from mrzecest.fitting_util import read_model_yaml, validate_model


def gcalc(ndo, log10beta=10.1, g0=5000.0):
    """Calculates antecedent outflow from a stream of ndo integrating using the trapezoidal method

    Parameters
    ----------
        ndo: pd.DataFrame
            a regular time series. Must be 15MIN, 1HOUR. Thus, NDO has been interpolated first.

        log10beta: float
            log10 of g-model parameter, which pre-log is in units ((cfs/s)*s).
            Values of 1.5e9 - R. Denton [cf] to 1.5e10 Ateljevich used previously,
            So a log range of 8.5-11 probably covers the range that needs to be explored
            in parameter fitting.

        g0: float
            initial condition. If g0 is not given it is equal to ndo at the first time step.

    Returns
    -------
        g: pd.DataFrame

          a regular time series, same sampling rate as input with the same start time as ndo
    """

    ti = ndo.index.freq
    nstep = len(ndo)
    if ti == pd.Timedelta("15min"):
        dt = 900.0  # [s]
    elif ti == pd.Timedelta("1h"):
        dt = 3600.0  # [s]
    else:
        raise ValueError("NDO time step must be 15MIN or 1HOUR. Please interpolate")

    beta = 10.0**log10beta
    ndo = ndo.dropna()

    g = ndo.copy()
    g.columns = ["g"]
    g.iloc[:] = np.nan
    g = g.squeeze()

    # Set initial condition
    if g0 is None:
        g0 = ndo["ndo"].iloc[0]

    # solve implicitly with trapezoidal method
    # using g_kernel to accelerate, which requires
    # pandas conversion to/from numpy
    g.iloc[:] = g_kernel(ndo.squeeze().to_numpy(), beta, g0, dt)

    return g


@numba.jit
def g_kernel(ndo, beta, g0, dt):
    """numpy based integration kernel for g(t)
    using trapezoidal method and numba."""
    div2dt = 2.0 * beta / dt  # units?
    g = np.empty(len(ndo), dtype=float)
    g[0] = g0
    ntime = len(g)
    qpast = ndo[0]
    gpast = qpast
    for i in np.arange(1, ntime):
        q = ndo[i]
        qterm = q - div2dt
        gnew = 0.5 * (
            qterm + np.sqrt(qterm**2 - 4 * (gpast**2 - gpast * (qpast + div2dt)))
        )
        g[i] = gnew
        gpast = gnew
        qpast = q
    return g


def ndo_mod(ndo, d_elev_filt, area_coef, energy, energy_coef):
    """Compute modified NDO (ndomod) used to drive the g-model.

    IMPORTANT: 'energy_coef' is repurposed to represent a *stratification* term
    (strongest at low tidal energy and diminishing with energy) that is applied
    in ec_est() (and during fitting) using a soft g-threshold. It is **not**
    applied here as a linear multiplier of the energy proxy.

    This function therefore returns the *base* ndomod using only the area term.
    The signature is retained to minimize interface churn.
    """
    ndo_mod = ndo.squeeze() + area_coef * d_elev_filt.squeeze()
    ndo_mod.name = "ndo"
    return ndo_mod.to_frame()


def _sigmoid(x):
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _g_threshold_from_ec(ec_target, beta0, beta1, npow, so, sb):
    """Compute the g at which the *mean* (z_sum=0) EC equals ec_target.

    Uses the kernel (with z_sum=0):
        EC = exp(beta0 + beta1 * g**npow) * (so - sb) + sb

    Solving gives:
        beta0 + beta1 * g**npow = log((EC - sb)/(so - sb))

    Returns g_threshold > 0. Raises on invalid inputs.
    """
    ec_target = float(ec_target)
    so = float(so)
    sb = float(sb)
    if not (so > sb):
        raise ValueError("Require so > sb.")
    if not (sb < ec_target < so):
        raise ValueError(
            f"ec_target must satisfy sb < ec_target < so; got {ec_target} with sb={sb}, so={so}."
        )
    beta0 = float(beta0)
    beta1 = float(beta1)
    npow = float(npow)
    if beta1 == 0.0:
        raise ValueError("beta1 must be nonzero to derive g threshold from EC.")
    frac = (ec_target - sb) / (so - sb)
    rhs = (np.log(frac) - beta0) / beta1
    if rhs <= 0.0 or not np.isfinite(rhs):
        raise ValueError(
            f"Derived g**npow is non-positive or non-finite (rhs={rhs}) for ec_target={ec_target}. "
            "Check beta0/beta1/so/sb consistency."
        )
    gthr = rhs ** (1.0 / npow)
    if not np.isfinite(gthr) or gthr <= 0.0:
        raise ValueError(f"Derived g threshold is invalid: {gthr}")
    return float(gthr)


def _front_weight(
    g_base: pd.Series, gthr: float, width_frac: float = 0.10
) -> pd.Series:
    """
    Compute a smooth frontal weighting based on transport magnitude.

    This function computes a sigmoid-based weight in the range [0, 1] that
    increases smoothly as the transport proxy ``g_base`` exceeds a threshold
    ``gthr``. The resulting weight is used to blend between background and
    front-enhanced behavior in the EC boundary formulation.

    The transition width is controlled as a fraction of the threshold value,
    ensuring scale-aware smoothing across different flow regimes.

    Parameters
    ----------
    g_base : pandas.Series
        Baseline transport proxy (e.g., g without stratification effects).
        Must be a one-dimensional, finite-valued series indexed in time.
        The index is preserved in the output.
    gthr : float
        Threshold value of ``g_base`` at which the weight is approximately 0.5.
        Must be positive.
    width_frac : float, optional
        Fraction of ``gthr`` over which the transition from 0 to 1 occurs.
        The effective transition width is ``max(width_frac * gthr, eps)``,
        where ``eps`` is a small positive value to avoid numerical singularity.
        Default is 0.10.

    Returns
    -------
    w_front : pandas.Series
        Frontal weighting factor in the range [0, 1], with the same index as
        ``g_base``. Values near 0 correspond to sub-threshold (background)
        conditions, and values near 1 correspond to super-threshold (front-
        dominated) conditions.

    Raises
    ------
    ValueError
        If ``gthr`` is non-positive or if ``g_base`` contains non-finite values.

    Notes
    -----
    The weighting is computed using a logistic (sigmoid) function of the form::

        w = 1 / (1 + exp(-(g_base - gthr) / w0))

    where ``w0 = width_frac * gthr``. This ensures a smooth, differentiable
    transition suitable for use in optimization and gradient-based fitting.

    This function performs no alignment or resampling; all inputs are assumed
    to be prevalidated and aligned upstream.
    """
    gthr = float(gthr)
    if gthr <= 0.0:
        raise ValueError("gthr must be > 0.")
    w = max(1e-6, float(width_frac) * gthr)

    gvals = g_base.to_numpy(dtype=float, copy=False)
    w_front = _sigmoid((gvals - gthr) / w)
    return pd.Series(w_front, index=g_base.index, name="w_front")


def _infer_fixed_dt(idx: pd.DatetimeIndex) -> pd.Timedelta:
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex.")
    if idx.has_duplicates:
        raise ValueError("Index must not contain duplicates.")
    if not idx.is_monotonic_increasing:
        raise ValueError("Index must be monotonic increasing.")
    if len(idx) < 2:
        raise ValueError("Need at least 2 timestamps to infer dt.")
    d = idx.to_series().diff().dropna()
    # strict: every step identical
    if (d != d.iloc[0]).any():
        raise ValueError("Index is not on a fixed sampling interval.")
    return pd.Timedelta(d.iloc[0])


def validate_inputs(
    ndo: pd.Series,
    elev: pd.Series,
    *,
    start=None,
    end=None,
    pad: pd.Timedelta,
    allowed_dt=(pd.Timedelta("15min"), pd.Timedelta("1h")),
) -> tuple[
    pd.Series, pd.Series, pd.DatetimeIndex, pd.Timestamp, pd.Timestamp, pd.Timedelta
]:
    """
    Validate inputs for EC estimation.

    Contract:
      - NDO defines the evaluation window and output index, though this can be expliclty overridden (shorter window).
      - Elevation must cover [start-pad, end+pad] and must include all NDO timestamps
        on [start, end] (superset condition).
      - No missing values in ndo or elev.
      - No resampling or interpolation is performed.
      - Hard fail on PeriodIndex.

    Returns:
      ndo_eval: ndo sliced to [start, end]
      elev_pad: elev sliced to [start-pad, end+pad]
      eval_index: ndo_eval.index
      start, end: resolved evaluation window
      dt: inferred sampling interval (must match ndo/elev and be allowed)
      eval_index: index for evaluation window
    """
    if isinstance(ndo.index, pd.PeriodIndex):
        raise TypeError(
            "ndo index must be a DatetimeIndex (PeriodIndex is not allowed)."
        )
    if isinstance(elev.index, pd.PeriodIndex):
        raise TypeError(
            "elev index must be a DatetimeIndex (PeriodIndex is not allowed)."
        )
    if not isinstance(ndo.index, pd.DatetimeIndex):
        raise TypeError("ndo index must be a DatetimeIndex.")
    if not isinstance(elev.index, pd.DatetimeIndex):
        raise TypeError("elev index must be a DatetimeIndex.")
    ndo = ndo.squeeze()
    elev = elev.squeeze()
    if ndo.isna().any():
        raise ValueError("ndo contains NaNs; missing data are not allowed for ndo.")
    if elev.isna().any():
        raise ValueError("elev contains NaNs; missing data are not allowed for elev.")

    ndo_dt = _infer_fixed_dt(ndo.index)
    elev_dt = _infer_fixed_dt(elev.index)
    if ndo_dt != elev_dt:
        raise ValueError(f"Sampling dt mismatch: ndo={ndo_dt}, elev={elev_dt}")
    if ndo_dt not in allowed_dt:
        raise ValueError(f"Unsupported dt={ndo_dt}. Allowed: {allowed_dt}")

    # Resolve eval window from ndo by default
    start = ndo.index[0] if start is None else pd.to_datetime(start)
    end = ndo.index[-1] if end is None else pd.to_datetime(end)

    if start >= end:
        raise ValueError("start must be < end.")

    # start/end must be within ndo (since ndo defines evaluation window)
    if start < ndo.index[0] or end > ndo.index[-1]:
        raise ValueError(
            "start/end must lie within ndo index (ndo defines evaluation window)."
        )

    ndo_eval = ndo.loc[start:end]
    eval_index = ndo_eval.index

    # Elevation must include every ndo timestamp over eval window (superset condition)
    missing = eval_index.difference(elev.index)
    if len(missing) > 0:
        ex = missing[:3].to_pydatetime().tolist()
        raise ValueError(
            "elev must contain all ndo timestamps over [start,end] "
            f"(missing {len(missing)} stamps; example missing={ex})"
        )

    # Padding coverage for elevation to accommodate filtration
    need_lo = start - pad
    need_hi = end + pad
    if elev.index[0] > need_lo or elev.index[-1] < need_hi:
        raise ValueError(f"elev must cover [{need_lo}, {need_hi}] for padding.")

    elev_pad = elev.loc[need_lo:need_hi]
    # (No NaNs already enforced above; slicing preserves that)

    return ndo_eval, elev_pad, eval_index, start, end, ndo_dt


def build_common_features(
    ndo: pd.Series,
    elev: pd.Series,
    *,
    filter_setup: dict,
    pad: pd.Timedelta,
    start=None,
    end=None,
):
    """
    Build all shared EC-estimator features exactly once.

    This function is the *only* place where filtering, derivatives,
    energy, and lagged tidal features are constructed.

    Returns a dict with strictly aligned Series/DataFrames.
    """

    # --- strict validation & window resolution ---
    ndo_eval, elev_pad, eval_index, start, end, dt = validate_inputs(
        ndo,
        elev,
        start=start,
        end=end,
        pad=pad,
    )

    fs = filter_setup
    filter_dt = pd.Timedelta(fs["dt"])
    k0 = int(fs["k0"])
    filter_len = int(fs["filter_length"])

    # --- filtering ---
    elev_filt = cosine_lanczos(elev_pad, "40h")
    elev_tidal = elev_pad - elev_filt
    energy = cosine_lanczos(elev_tidal * elev_tidal, "40h")

    # --- time derivative (central) ---
    two_dtsec = 2.0 * dt.total_seconds()
    d_elev_filt = (elev_filt.shift(-1) - elev_filt.shift(1)) / two_dtsec

    # --- lagged Z matrix (NOT summed) ---
    dstep = int(filter_dt / dt)
    z_df = pd.DataFrame(index=elev_tidal.index)

    for k in range(filter_len):
        z_df[f"z{k}"] = elev_tidal.shift(-(k0 - k) * dstep)

    # Drop rows where any z{k} is undefined
    z_df = z_df.dropna()

    # --- enforce eval-window coverage ---
    missing = eval_index.difference(z_df.index)
    if len(missing) > 0:
        raise ValueError(
            f"z features undefined on eval window "
            f"(missing {len(missing)} timestamps; first={missing[0]})"
        )

    # --- final slicing (single authority) ---
    out = {
        "eval_index": eval_index,
        "ndo": ndo_eval,
        "elev_filt": elev_filt.loc[eval_index],
        "elev_tidal": elev_tidal.loc[eval_index],
        "energy": energy.loc[eval_index],
        "d_elev_filt": d_elev_filt.loc[eval_index],
        "Z": z_df.loc[eval_index],
        "dt": dt,
        "start": start,
        "end": end,
    }

    # Hard fail on nonfinite values inside eval window
    for k, v in out.items():
        if isinstance(v, (pd.Series, pd.DataFrame)):
            if not np.isfinite(v.values).all():
                raise ValueError(f"Nonfinite values in feature '{k}'")

    return out


def ec_est_yaml(ndo, elev, yaml_fn, start=None, end=None, return_components=False):
    """Estimate EC using a model specification YAML."""

    model = read_model_yaml(yaml_fn)
    validate_model(model)

    fs = model["filter_setup"]
    npow = float(model["npow"])
    front = model.get("front")

    return ec_est(
        ndo,
        elev,
        model["area_coef"],
        model["energy_coef"],
        model["beta_log10"],
        model["b0"],
        model["b1"],
        npow,
        fs["k0"],
        model["afilt"],
        pd.Timedelta(fs["dt"]),
        model["so"],
        model["sb"],
        front_ec_target=float(front["ec_target"]),
        front_gthr=float(front["gthr"]),
        front_energy_ref=float(front["energy_ref"]),
        front_width_frac=float(front["width_frac"]),
        g_thr_tide=float(model["g_thr_tide"]),
        width_frac_tide=float(model.get("width_frac_tide", 0.6)),
        start=start,
        end=end,
        return_components=return_components,
    )


def ec_est(
    ndo,
    elev,
    area_coef,
    energy_coef,
    log10beta,
    beta0,
    beta1,
    npow,
    filter_k0,
    filt_coefs,
    filter_dt,
    so,
    sb,
    front_ec_target,
    front_gthr,
    front_energy_ref,
    front_width_frac,
    g_thr_tide,
    width_frac_tide=0.6,
    start=None,
    end=None,
    return_components=False,
):
    """
    Estimate electrical conductivity (EC) at the Martinez boundary using the g-model.

    Parameters
    ----------
    ndo : pandas.DataFrame or pandas.Series
        Net Delta Outflow (NDO) as a regular time series in cfs (must be 15MIN or 1HOUR frequency).
    elev : pandas.DataFrame or pandas.Series
        Water surface elevation as a regular time series in ft (must be 15MIN or 1HOUR frequency).
    area_coef : float
        Coefficient for the area term in the NDO modification.
    energy_coef : float
        Coefficient for the energy term in the NDO modification.
    log10beta : float
        Log10 of the g-model parameter beta.
    beta0 : float
        Intercept parameter for the EC kernel.
    beta1 : float
        Slope parameter for the EC kernel.
    npow : float
        Exponent parameter for the EC kernel.
    filter_k0 : float
        Filter parameter k0 for the lagged elevation sum.
    filt_coefs : array-like
        Filter coefficients for the lagged elevation sum.
    filter_dt : pandas.Timedelta
        Time step for the filter.
    so : float
        Ocean salinity (upper bound for EC).
    sb : float
        Base salinity (lower bound for EC).
    start : pandas.Timestamp or None, optional
        Start time for the output EC series. If None, uses the earliest time in the input.
    end : pandas.Timestamp or None, optional
        End time for the output EC series. If None, uses the latest time in the input.

    Returns
    -------
    ec : pandas.Series
        Estimated electrical conductivity time series at the Martinez boundary, indexed by time.
    """

    ctx = build_common_features(
        ndo=ndo,
        elev=elev,
        filter_setup={
            "dt": filter_dt,
            "k0": filter_k0,
            "filter_length": len(filt_coefs),
        },
        pad=pd.Timedelta("9d"),
        start=start,
        end=end,
    )
    # Output window: if start/end provided, they must be within the input index.
    if start is not None:
        start = pd.to_datetime(start)
        if start not in ndo.index:
            raise ValueError("start must be an exact timestamp in the input index.")
    if end is not None:
        end = pd.to_datetime(end)
        if end not in ndo.index:
            raise ValueError("end must be an exact timestamp in the input index.")
    if start is not None and end is not None and start >= end:
        raise ValueError("start must be < end.")

    Z = ctx["Z"].values
    z_sum = Z @ np.asarray(filt_coefs)
    elev_filt = ctx["elev_filt"]
    elev_tidal = ctx["elev_tidal"]
    d_elev_filt = ctx["d_elev_filt"]
    energy = ctx["energy"]
    eval_index = ctx["eval_index"]
    start = ctx["start"]
    end = ctx["end"]
    ndo = ctx["ndo"]

    if len(z_sum) != len(eval_index):
        raise ValueError("z_sum length mismatch vs eval_index (unexpected).")

    logger.info(
        "NDO mean is %.0f cfs, max is %.0f cfs, min is %.0f cfs",
        float(ndo.mean()),
        float(ndo.max()),
        float(ndo.min()),
    )
    if not d_elev_filt.index.equals(ndo.index):
        raise ValueError("d_elev_filt and ndo must be aligned on the eval index.")
    ndomod_base = ndo_mod(ndo, d_elev_filt, area_coef, energy, energy_coef).squeeze()

    # Two-pass stratification correction (repurposed energy_coef):
    #  - compute g_base from base ndomod (area only)
    #  - derive a soft g-threshold corresponding to EC ~= front_ec_target (mean, z_sum=0)
    #  - apply low-energy weight so the effect is strongest at low tidal energy
    g_base = gcalc(ndomod_base, log10beta=log10beta)
    gthr = float(front_gthr)
    if (not np.isfinite(gthr)) or (gthr <= 0.0):
        raise ValueError(f"front_gthr must be finite and > 0, got {gthr}")

    logger.info(
        "gthr=%s g_base mean=%s max=%s min=%s",
        gthr,
        float(g_base.mean()),
        float(g_base.max()),
        float(g_base.min()),
    )
    width_frac = float(front_width_frac)
    if not (0.0 < width_frac < 1.0):
        raise ValueError(f"front_width_frac must be in (0,1), got {width_frac}")
    w_front = _front_weight(g_base, gthr, width_frac=width_frac)

    energy_eval = energy.squeeze().loc[eval_index]

    # Disallow NaNs inside eval window
    if (~np.isfinite(energy_eval.values)).any():
        bad = energy_eval.index[~np.isfinite(energy_eval.values)]
        ex = bad[:3].to_pydatetime().tolist()
        raise ValueError(
            f"energy has nonfinite values inside eval window "
            f"(count={len(bad)}; first={bad[0]}; examples={ex}). "
            "Increase elevation padding or adjust filter/window."
        )

    energy_ref = float(front_energy_ref)
    if (not np.isfinite(energy_ref)) or (energy_ref <= 0.0):
        raise ValueError(f"front_energy_ref must be finite and > 0, got {energy_ref}")

    low_energy_weight = 1.0 / (1.0 + (energy_eval / energy_ref))

    if (
        (w_front.min() < -1e-6)
        or ((w_front.max() > 1.0 + 1e-6))
        or (~np.isfinite(w_front.values)).any()
    ):
        raise ValueError(
            f"w_front out of [0,1] or nonfinite: min={w_front.min()} max={w_front.max()}"
        )

    if (
        (low_energy_weight.min() < -1e-6)
        or (low_energy_weight.max() > 1.0 + 1e-6)
        or (~np.isfinite(low_energy_weight.values)).any()
    ):
        raise ValueError(
            f"low_energy_weight out of [0,1] or nonfinite: min={low_energy_weight.min()} max={low_energy_weight.max()}"
        )

    strat_term = energy_coef * w_front * low_energy_weight

    ndomod = (ndomod_base - strat_term).rename("ndo").to_frame()

    def _tail_stats(name, s):
        s = s.dropna().astype(float).squeeze()
        qs = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 0.5]  # min, 0.01%, 0.1%, 1%, 5%, median
        qv = s.quantile(qs)

    corr_area = area_coef * d_elev_filt
    corr_energy = strat_term

    ce = pd.Series(corr_energy.squeeze()).astype(float)
    bad = ~np.isfinite(ce.values)
    n_bad = int(bad.sum())
    if n_bad > 0:
        # Warn once for user-visible alert and provide context at DEBUG level
        logger.warning("corr_energy nonfinite: %d", n_bad)
        i = np.where(bad)[0][0]
        t = ce.index[i]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "first nonfinite at %s corr_energy=%s energy_coef=%s w_front=%s "
                "low_energy_weight=%s energy=%s g_base=%s gthr=%s",
                t,
                float(ce.loc[t]),
                energy_coef,
                float(w_front.loc[t]),
                float(low_energy_weight.loc[t]),
                float(energy.loc[t]),
                float(g_base.loc[t]),
                gthr,
            )

    abs_ce = ce.abs()
    tmax = abs_ce.idxmax()

    # Keep strictness about timestamps: ndomod must exist on eval_index
    missing = eval_index.difference(ndomod.index)
    if len(missing) > 0:
        ex = missing[:3].to_pydatetime().tolist()
        raise ValueError(
            f"ndomod missing {len(missing)} eval timestamps (example {ex}). "
            "This indicates derivative/filter edge effects or insufficient elevation padding."
        )

    ndomod = ndomod.loc[eval_index]

    logger.info(
        "Modified NDO mean is %.0f cfs, max is %.0f cfs, min is %.0f cfs",
        float(ndomod.mean().values[0]),
        float(ndomod.max().values[0]),
        float(ndomod.min().values[0]),
    )
    # calculate g-model results
    g = gcalc(ndomod, log10beta=log10beta)

    ec = pd.Series(index=eval_index, dtype=float)
    logger.debug("solving for ec")
    width_tide = max(1e-6, float(width_frac_tide) * float(g_thr_tide))
    ec.iloc[:] = ec_kernel(
        g.to_numpy(), z_sum, beta0, beta1, npow, so, sb, g_thr_tide, width_tide
    )
    logger.debug("done")

    ec = ec.loc[eval_index]

    logger.info(
        "Estimated EC mean is %.0f, max is %.0f, min is %.0f",
        float(ec.mean()),
        float(ec.max()),
        float(ec.min()),
    )

    if return_components:
        gv = g.loc[eval_index].to_numpy(dtype=float)
        # Decompose the kernel exponent: ln((EC-sb)/(so-sb)) = mean_term + tide_term
        mean_term = beta0 + beta1 * np.power(gv, npow)
        tide_term = _sigmoid((gv - g_thr_tide) / width_tide) * np.asarray(z_sum, dtype=float)
        components = pd.DataFrame(
            {
                "g": gv,
                "z_sum": np.asarray(z_sum, dtype=float),
                "q": ndomod["ndo"].to_numpy(dtype=float),
                "mean_term": mean_term,
                "tide_term": tide_term,
                "ec": ec.to_numpy(dtype=float),
            },
            index=eval_index,
        )
        return ec, components

    return ec


@numba.jit
def ec_kernel(g, z_sum, beta0, beta1, npow, so, sb, g_thr_tide, width_tide):
    """numpy based kernel for ec(t) using numba.

    The tidal term is gated by a bounded logistic W(g) in [0,1] that rises with g
    (falls with salinity), so the *observable* EC tidal amplitude ~ (EC-sb)*W(g)
    vanishes toward fresh (via the (EC-sb) prefactor) and holds a broad plateau at
    high salinity, instead of the unbounded g**npow_tide factor.
    """

    ec = np.empty(len(g), dtype=float)
    ntime = len(g)

    for i in range(ntime):
        if i == 0:
            ec[i] = np.nan
            continue

        g_main = g[i] ** npow
        x = (g[i] - g_thr_tide) / width_tide
        if x >= 0.0:
            gate = 1.0 / (1.0 + np.exp(-x))
        else:
            ex = np.exp(x)
            gate = ex / (1.0 + ex)

        ecfrac = beta0 + beta1 * g_main + gate * z_sum[i]
        ec[i] = np.exp(ecfrac) * (so - sb) + sb

    return ec
