"""Calculate g-model based estimate of Martinez boundary."""

import yaml
import pandas as pd
import numpy as np
import datetime as dt
from vtools import *
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
    if ti is None:
        freq_str = ndo.index.inferred_freq
        # If the string is just a unit ('h', 'D', 'W', etc.), prepend '1'
        # This converts 'h' to '1h', which pd.Timedelta can parse.
        if freq_str and freq_str.isalpha():
            ti = pd.Timedelta("1 " + freq_str)
        else:
            ti = pd.Timedelta(freq_str)
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
    ndo_mod = (
        ndo.squeeze()
        + area_coef * d_elev_filt.squeeze()
        + energy_coef * energy.squeeze()
    )
    ndo_mod.name = "ndo"

    return ndo_mod.to_frame()


def z_sum_term(z, filter_k0, filt_coefs, filter_dt):

    filter_len = len(filt_coefs)
    df_freq = pd.Timedelta(z.index.freq)
    d_step = int(filter_dt / df_freq)  # number of rows for each Dt

    # Ensure z is a DataFrame with column 'elev_tidal'
    if isinstance(z, pd.Series):
        z = z.to_frame(name="elev_tidal")
    else:
        z.columns = ["elev_tidal"]

    for k in range(0, filter_len):
        z[f"z{k}"] = z["elev_tidal"].shift(-int((filter_k0 - k) * d_step))

    z = z.dropna()
    z_sum = pd.Series(
        index=z.index,
        data=np.nansum(
            z.iloc[:, 1 : (len(filt_coefs) + 1)].values * filt_coefs, axis=1
        ),
    )

    return z_sum


import pandas as pd


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
) -> tuple[pd.Series, pd.Series, pd.DatetimeIndex, pd.Timestamp, pd.Timestamp, pd.Timedelta]:
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
        raise TypeError("ndo index must be a DatetimeIndex (PeriodIndex is not allowed).")
    if isinstance(elev.index, pd.PeriodIndex):
        raise TypeError("elev index must be a DatetimeIndex (PeriodIndex is not allowed).")
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
        raise ValueError("start/end must lie within ndo index (ndo defines evaluation window).")

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



def ec_est_yaml(ndo, elev, yaml_fn, start=None, end=None):
    """Estimate EC using a model specification YAML.

    """
    
    model = read_model_yaml(yaml_fn)
    validate_model(model)

    fs = model["filter_setup"]
    return ec_est(
        ndo,
        elev,
        model["area_coef"],
        model["energy_coef"],
        model["beta_log10"],
        model["b0"],
        model["b1"],
        model["npow"],
        fs["k0"],
        model["afilt"],
        pd.Timedelta(fs["dt"]),
        model["so"],
        model["sb"],
        start=start,
        end=end,
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
    start=None,
    end=None,
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

    # strict validation; ndo defines evaluation grid; elev is padded superset
    ndo, elev, eval_index, start, end, dt = validate_inputs(
        ndo,
        elev,
        start=start,
        end=end,
        pad=pd.Timedelta("9d"),   # later: read from model.yaml requirements
    )

    print(f"Tidal mean is {elev.mean():.2f} ft, range is {elev.max()-elev.min():.2f} ft")

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

    # Apply a cosine Lanczos filter (low-pass) to the elevation dataframe (low-pass) to the elevation dataframe
    elev_filt = cosine_lanczos(elev, "40h")
    elev_tidal = elev.copy() - elev_filt  # isolate tidal part for z_sum term
    energy = cosine_lanczos(elev_tidal * elev_tidal, "40h")

    # calculate subtidal effects on ndo
    offset = elev_filt.index.freq

    two_dtsec = (
        2.0 * pd.Timedelta(offset, unit=offset.freqstr.lower()).total_seconds()
    )  # dt term to be used for estimating derivative of tide
    d_elev_filt = (elev_filt.shift(-1) - elev_filt.shift(1)) / two_dtsec
    d_elev_filt = d_elev_filt.dropna()

    print(
        f"NDO mean is {ndo.mean():.0f} cfs, max is {ndo.max():.0f} cfs, min is {ndo.min():.0f} cfs"
    )
    ndomod = ndo_mod(ndo, d_elev_filt, area_coef, energy, energy_coef)

    # Keep strictness about timestamps: ndomod must exist on eval_index
    missing = eval_index.difference(ndomod.index)
    if len(missing) > 0:
        ex = missing[:3].to_pydatetime().tolist()
        raise ValueError(
            f"ndomod missing {len(missing)} eval timestamps (example {ex}). "
            "This indicates derivative/filter edge effects or insufficient elevation padding."
        )

    ndomod = ndomod.loc[eval_index] 

    print(
        f"Modified NDO mean is {ndomod.mean().values[0]:.0f} cfs, max is {ndomod.max().values[0]:.0f} cfs, min is {ndomod.min().values[0]:.0f} cfs"
    )
    # calculate g-model results
    g = gcalc(ndomod, log10beta=log10beta)

    # lagged z_sum term (will drop edge times where shifts are undefined)
    # Compute z_sum on the padded tidal series so shifts have support
    z_sum = z_sum_term(elev_tidal, filter_k0, filt_coefs, filter_dt)

    # Now enforce coverage on eval window (strict) and then select
    missing = eval_index.difference(z_sum.index)
    if len(missing) > 0:
        ex0 = missing[0]
        exN = missing[-1]
        raise ValueError(
            f"z_sum is not defined on the full evaluation window "
            f"(missing {len(missing)} timestamps; first missing {ex0}, last missing {exN}). "
            "Choose a narrower start/end (move inward), or provide more elevation padding, "
            "or adjust filter_setup (k0/filter_length/dt/centering)."
        )

    z_sum = z_sum.loc[eval_index]

    # Only now slice the other series for downstream computations/printing
    elev_filt = elev_filt.loc[eval_index]
    elev_tidal = elev_tidal.loc[eval_index]
    energy = energy.loc[eval_index]
    g = g.loc[eval_index]

    ec = pd.Series(index=eval_index, dtype=float)
    print("solving for ec")
    ec.iloc[:] = ec_kernel(
        g.to_numpy(), z_sum.to_numpy(), beta0, beta1, npow, so, sb
    )
    print("done")


    ec = ec.loc[eval_index]
    
    print(
        f"Estimated EC mean is {ec.mean():.0f}, "
        f"max is {ec.max():.0f}, min is {ec.min():.0f}"
    )

    return ec


@numba.jit
def ec_kernel(g, z_sum, beta0, beta1, npow, so, sb):
    """numpy based kernel for ec(t) using numba."""

    ec = np.empty(len(g), dtype=float)
    ntime = len(g)

    for i in np.arange(1, ntime):
        ecfrac = (
            beta0 + beta1 * g[i] ** npow + g[i] ** npow * z_sum[i]
        )  # npow1 and b1 are our parameters to tweak
        ec[i] = np.exp(ecfrac) * (so - sb) + sb  # solving for s term

    return ec
