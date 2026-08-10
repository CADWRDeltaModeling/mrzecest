"""Tests for the UKF/URTS EC smoother layer (mrzecest.ec_filter).

Self-contained: synthetic ndo/elev drivers + the fitted example/model.yaml.
Covers (1) equivalence with the deterministic operator, (2) gaps do not blow up
under a perfect model, and (3) the smart-persistence win over the deterministic
estimate when a slow bias is present.
"""

import os

import numpy as np
import pandas as pd
import pytest

from mrzecest.ec_filter import (
    FilterConfig,
    compute_drivers,
    run_smoother,
    make_gap_mask,
    gap_metrics,
    _obs_log_ratio,
    _ec_from_y,
)

MODEL_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "example",
    "model.yaml",
)


def _synthetic_drivers(freq="1h", months=3):
    """Salt-season-like synthetic ndo (low outflow) + tidal elevation."""
    start = pd.Timestamp("2010-08-01")
    end = start + pd.DateOffset(months=months)
    pad = pd.Timedelta("9d")

    idx_ndo = pd.date_range(start, end, freq=freq)
    idx_elev = pd.date_range(start - pad, end + pad, freq=freq)

    def _hours(idx):
        return (idx - idx[0]).total_seconds().to_numpy() / 3600.0

    th = _hours(idx_elev)
    tidal = (
        2.0 * np.sin(2 * np.pi * th / 12.42)
        + 1.0 * np.sin(2 * np.pi * th / 24.0)
        + 0.4 * np.sin(2 * np.pi * th / (14 * 24.0))  # spring/neap
    )
    subtidal = 0.5 * np.sin(2 * np.pi * th / (30 * 24.0))
    elev = pd.Series(subtidal + tidal, index=idx_elev, name="elev")

    tn = _hours(idx_ndo)
    ndo = pd.Series(
        9000.0 + 5000.0 * np.sin(2 * np.pi * tn / (45 * 24.0)),
        index=idx_ndo,
        name="ndo",
    )
    return ndo, elev, start, end


@pytest.fixture(scope="module")
def drivers_params():
    ndo, elev, start, end = _synthetic_drivers()
    drivers, params = compute_drivers(ndo, elev, MODEL_YAML)
    return ndo, elev, drivers, params


def test_no_obs_reproduces_deterministic(drivers_params):
    """With no observations, tiny P0 and zero g-process noise, the smoother must
    collapse onto the deterministic operator (validates drivers + dynamics + the
    observation reconstruction all match ec_est)."""
    ndo, elev, drivers, params = drivers_params
    idx = drivers.index
    ec_obs = pd.Series(np.nan, index=idx)  # nothing to assimilate

    cfg = FilterConfig(
        q_logg_per_day=0.0,
        sigma_b=1e-8,
        init_logg_std=1e-4,
        init_b_std=1e-8,
        r_std=0.05,
    )
    res = run_smoother(ndo, elev, MODEL_YAML, ec_obs, cfg=cfg)

    # ec_det[0] is NaN by construction (kernel leaves first sample undefined).
    m = np.isfinite(res["ec_det"].to_numpy())
    np.testing.assert_allclose(
        res["ec_smooth"].to_numpy()[m], res["ec_det"].to_numpy()[m], rtol=2e-3
    )
    # g dynamics reproduce the deterministic g-model trajectory (index 0 is the
    # g0 display label in the operator; compare from the first integrated step).
    np.testing.assert_allclose(
        res["g_smooth"].to_numpy()[1:], drivers["g_det"].to_numpy()[1:], rtol=2e-3
    )
    # Bias state never moves without observations.
    assert np.nanmax(np.abs(res["b_smooth"].to_numpy())) < 1e-6


def test_perfect_obs_gap_does_not_blow_up(drivers_params):
    """Assimilating the deterministic EC itself (perfect model). Even across a
    multi-day gap the smoother stays on the deterministic solution."""
    ndo, elev, drivers, params = drivers_params
    idx = drivers.index
    truth = drivers["ec_det"]

    held = make_gap_mask(idx, gaps=[("3D", 1)], seed=1)
    ec_obs = truth.copy()
    ec_obs[held.to_numpy()] = np.nan

    res = run_smoother(ndo, elev, MODEL_YAML, ec_obs, cfg=FilterConfig(r_std=0.02))

    stats = gap_metrics(res, truth, held)
    assert stats["n"] > 0
    # Deterministic fallback in the gap keeps relative error small.
    assert stats["rmse_log"] < 0.05


def test_smart_persistence_beats_deterministic(drivers_params):
    """Introduce a slow log-space bias absent from the deterministic model. The
    smoother should learn it (via b_y) and impute held-out gaps better than the
    deterministic estimate."""
    ndo, elev, drivers, params = drivers_params
    idx = drivers.index
    g_det = drivers["g_det"].to_numpy()

    # Synthetic truth = deterministic kernel + slow bias b(t) (unknown to model).
    z = drivers["z_sum"].to_numpy()
    y0 = _obs_log_ratio(g_det, z, params)
    tdays = (idx - idx[0]).total_seconds().to_numpy() / 86400.0
    bias = 0.30 * np.sin(2 * np.pi * tdays / 40.0)
    truth = pd.Series(_ec_from_y(y0 + bias, params), index=idx)

    held = make_gap_mask(idx, gaps=[("12h", 4), ("2D", 3), ("10D", 1)], seed=7)
    ec_obs = truth.copy()
    ec_obs[held.to_numpy()] = np.nan

    cfg = FilterConfig(q_logg_per_day=0.0, tau_b_days=5.0, sigma_b=0.4, r_std=0.02)
    res = run_smoother(ndo, elev, MODEL_YAML, ec_obs, cfg=cfg)

    stats = gap_metrics(res, truth, held)
    assert stats["n"] > 0
    assert stats["rmse"] < stats["rmse_det"]
