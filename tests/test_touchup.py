import numpy as np
import pandas as pd
import pytest

from mrzecest.ec_boundary_fit import (
    touchup_least_squares,
    _pack_touchup,
    _unpack_touchup,
    _touchup_bounds,
    _TOUCHUP_OUTER_KEYS,
)
from mrzecest.ec_boundary import ec_est_yaml
from mrzecest.fitting_util import write_model_yaml, compare_metrics


FILTER_LENGTH = 13


def _synthetic_inputs():
    """Build synthetic ndo/elev/ec_obs covering a short fit window."""
    fit_start = pd.Timestamp("2004-02-01")
    fit_end = pd.Timestamp("2004-05-01")
    pad = pd.Timedelta("9d")

    idx_ndo = pd.date_range(fit_start, fit_end, freq="1h")
    idx_elev = pd.date_range(fit_start - pad, fit_end + pad, freq="1h")

    def _hours(idx):
        return (idx - idx[0]).total_seconds().values / 3600.0

    th = _hours(idx_elev)
    tidal = 2.0 * np.sin(2 * np.pi * th / 12.42) + 1.0 * np.sin(2 * np.pi * th / 24.0)
    subtidal = 0.5 * np.sin(2 * np.pi * th / (14 * 24.0))
    elev = pd.Series(subtidal + tidal, index=idx_elev, name="elev")

    tn = _hours(idx_ndo)
    ndo = pd.Series(
        8000.0 + 5000.0 * np.sin(2 * np.pi * tn / (60 * 24.0)) + 200.0,
        index=idx_ndo,
        name="ndo",
    )

    ec_obs = pd.Series(
        300.0 + 30000.0 * np.exp(-ndo.values / 8000.0),
        index=idx_ndo,
        name="ec_obs",
    )
    return ndo, elev, ec_obs, fit_start, fit_end


def _config():
    return {
        "so": 55000.0,
        "sb": 200.0,
        "filter_setup": {
            "dt": "3h",
            "k0": 6,
            "filter_length": FILTER_LENGTH,
            "centering": "causal",
        },
        "fit_run": {
            "start": "2004-02-01",
            "end": "2004-05-01",
            "outer_params": [
                {"key": "beta_log10", "x0": 10.0, "bounds": [9.5, 11.5]},
                {"key": "npow", "x0": 0.5, "bounds": [0.2, 0.8]},
                {"key": "g_thr_tide", "x0": 20000.0, "bounds": [5000.0, 60000.0]},
                {"key": "area_coef", "scale": 1e9, "x0": -1.0, "bounds": [-3.0, 0.0]},
                {"key": "energy_coef", "scale": 1e3, "x0": 1.0, "bounds": [0.0, 6.0]},
                {"key": "so", "scale": 1e5, "x0": 0.5, "bounds": [0.4, 0.55]},
            ],
        },
    }


def _seed_model():
    return {
        "so": 50000.0,
        "sb": 200.0,
        "area_coef": -1.0e9,
        "energy_coef": 1000.0,
        "beta_log10": 10.0,
        "npow": 0.5,
        "g_thr_tide": 20000.0,
        "width_frac_tide": 0.6,
        "g0": 5000.0,
        "b0": 0.0,
        "b1": -0.005,
        "filter_setup": {
            "dt": "3h",
            "k0": 6,
            "filter_length": FILTER_LENGTH,
            "centering": "causal",
        },
        "afilt": [0.001 * (1 if k % 2 == 0 else -1) for k in range(FILTER_LENGTH)],
        "front": {
            "ec_target": 20000.0,
            "gthr": 5000.0,
            "energy_ref": 2.0,
            "width_frac": 0.1,
        },
    }


def _sse(obs, pred):
    df = pd.concat(
        [pd.Series(obs).rename("o"), pd.Series(pred).rename("p")],
        axis=1,
        join="inner",
    ).dropna()
    r = df["p"].to_numpy() - df["o"].to_numpy()
    return float(np.sum(r**2))


def test_pack_unpack_roundtrip():
    model = _seed_model()
    vec = _pack_touchup(model)
    assert vec.shape[0] == len(_TOUCHUP_OUTER_KEYS) + 2 + FILTER_LENGTH
    back = _unpack_touchup(vec, model)
    for k in _TOUCHUP_OUTER_KEYS + ("b0", "b1"):
        assert back[k] == pytest.approx(model[k])
    assert back["afilt"] == pytest.approx(model["afilt"])


def test_touchup_reduces_or_holds_sse(tmp_path):
    ndo, elev, ec_obs, fit_start, fit_end = _synthetic_inputs()
    cfg = _config()
    model_glm = _seed_model()

    model_ls, result = touchup_least_squares(
        cfg, model_glm, elev=elev, ndo=ndo, ec_obs=ec_obs
    )

    assert np.isfinite(result.cost)

    # Evaluate both models on the fit window and compare SSE.
    glm_yaml = tmp_path / "model_glm.yaml"
    ls_yaml = tmp_path / "model_ls.yaml"
    write_model_yaml(model_glm, str(glm_yaml))
    write_model_yaml(model_ls, str(ls_yaml))

    pad = pd.Timedelta("9d")
    ndo_in = ndo.loc[fit_start:fit_end]
    elev_in = elev.loc[fit_start - pad : fit_end + pad]
    est_glm = ec_est_yaml(ndo_in, elev_in, str(glm_yaml))
    est_ls = ec_est_yaml(ndo_in, elev_in, str(ls_yaml))

    obs = ec_obs.loc[fit_start:fit_end]
    sse_glm = _sse(obs, est_glm)
    sse_ls = _sse(obs, est_ls)
    # Local refinement must not increase SSE relative to the seed.
    assert sse_ls <= sse_glm * (1.0 + 1e-6)


def test_touchup_respects_bounds():
    ndo, elev, ec_obs, _, _ = _synthetic_inputs()
    cfg = _config()
    model_glm = _seed_model()
    model_ls, _ = touchup_least_squares(
        cfg, model_glm, elev=elev, ndo=ndo, ec_obs=ec_obs
    )
    lo, hi = _touchup_bounds(cfg, model_glm)
    vec = _pack_touchup(model_ls)
    assert np.all(vec >= lo - 1e-6)
    assert np.all(vec <= hi + 1e-6)


def test_touchup_honors_fixed_b0():
    ndo, elev, ec_obs, _, _ = _synthetic_inputs()
    cfg = _config()
    cfg["fit_run"]["inner"] = {"b0": {"fix": True, "value": 0.0}}
    model_glm = _seed_model()
    model_glm["b0"] = 0.0

    model_ls, result = touchup_least_squares(
        cfg, model_glm, elev=elev, ndo=ndo, ec_obs=ec_obs
    )
    # b0 must remain pinned at the configured value.
    assert model_ls["b0"] == 0.0
    # The optimized vector must exclude b0 (one fewer element than free case).
    lo, hi = _touchup_bounds(cfg, model_glm, fix_b0=True)
    assert result.x.shape[0] == len(_TOUCHUP_OUTER_KEYS) + 1 + FILTER_LENGTH
    assert lo.shape[0] == result.x.shape[0]


def test_compare_metrics_basic():
    idx = pd.date_range("2004-01-01", periods=10, freq="h")
    obs = pd.Series(np.linspace(1000.0, 5000.0, 10), index=idx)
    pred = obs + 100.0
    m = compare_metrics(obs, pred, so=55000.0, sb=200.0)
    assert m["mae"] == pytest.approx(100.0)
    assert m["rmse"] == pytest.approx(100.0)
    assert np.isfinite(m["gamma_deviance"])
    assert m["n"] == 10
