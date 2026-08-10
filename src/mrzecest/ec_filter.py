"""Unscented Kalman filter / smoother layer over the deterministic mrzecest estimator.

Phase 1 (MRZ-only):
    - Reuses the deterministic forward operator (``ec_est_yaml`` with
      ``return_components=True``) as the source of the exogenous drivers
      ``q`` (modified NDO) and ``z_sum`` (lagged tidal design), and the
      deterministic transport ``g`` / EC as an initial condition and fallback.
    - Estimates a low-dimensional latent state online in **log-observation space**:

          x = [log_g, b_y]

      where ``g`` is the Denton g-model transport and ``b_y`` is a slowly
      mean-reverting (OU / AR(1)) additive bias in the log-EC-ratio space that
      absorbs structural error ("smart persistence").
    - Observation model (log space):

          y = ln((EC - sb) / (so - sb)) = b0 + b1 * g**npow + W(g) * z_sum + b_y

      which is exactly the deterministic kernel exponent (see
      ``mrzecest.ec_boundary.ec_kernel``) plus the bias state.

The filter is a plain-numpy additive-noise UKF (Van der Merwe scaled sigma
points) with an unscented Rauch-Tung-Striebel (URTS) smoother for imputation.

No numba / filterpy dependency. Kernel and g-model *parameters* are held fixed
at their fitted values (read from the model YAML); only the states are estimated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mrzecest.ec_boundary import ec_est_yaml
from mrzecest.fitting_util import read_model_yaml

__all__ = [
    "ModelParams",
    "FilterConfig",
    "compute_drivers",
    "run_smoother",
    "make_gap_mask",
    "gap_metrics",
]


# --------------------------------------------------------------------------- #
# Model parameters (fixed, from the fitted model YAML)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelParams:
    """Fixed kernel + g-model parameters needed by the filter."""

    so: float
    sb: float
    beta0: float
    beta1: float
    npow: float
    g_thr_tide: float
    width_tide: float
    div2dt: float  # 2 * beta / dt_sec  (g-model Crank-Nicolson constant)
    dt_sec: float

    @classmethod
    def from_yaml(cls, model_yaml: str, dt_sec: float) -> "ModelParams":
        m = read_model_yaml(model_yaml)
        g_thr_tide = float(m["g_thr_tide"])
        width_tide = max(1e-6, float(m["width_frac_tide"]) * g_thr_tide)
        beta = 10.0 ** float(m["beta_log10"])
        return cls(
            so=float(m["so"]),
            sb=float(m["sb"]),
            beta0=float(m["b0"]),
            beta1=float(m["b1"]),
            npow=float(m["npow"]),
            g_thr_tide=g_thr_tide,
            width_tide=width_tide,
            div2dt=2.0 * beta / float(dt_sec),
            dt_sec=float(dt_sec),
        )


# --------------------------------------------------------------------------- #
# Filter configuration (tunable knobs)
# --------------------------------------------------------------------------- #
@dataclass
class FilterConfig:
    """Tunable filter/smoother settings (phase 1a: constant R)."""

    # g-model process noise: per-day std of log(g) random walk (model error).
    # Set 0.0 to hold g exactly on the deterministic trajectory.
    q_logg_per_day: float = 0.05
    # Observation log-bias b_y as an Ornstein-Uhlenbeck / AR(1) process.
    tau_b_days: float = 3.0      # mean-reversion timescale
    sigma_b: float = 0.15        # stationary std (log-EC-ratio units ~ fractional)
    # Constant observation noise std in log-EC-ratio (y) space.
    r_std: float = 0.05
    # Initial state uncertainty.
    init_logg_std: float = 0.10
    init_b_std: float = 0.15
    # Unscented transform parameters (n = 2; kappa chosen so n + kappa = 3).
    alpha: float = 1.0
    beta_ut: float = 2.0
    kappa: float = 1.0
    # Numerical jitter added to covariances before factorization / inversion.
    jitter: float = 1e-9


# --------------------------------------------------------------------------- #
# Elementwise kernel pieces (single source shared by sigma points + rebuild)
# --------------------------------------------------------------------------- #
def _tide_gate(g: np.ndarray, g_thr_tide: float, width_tide: float) -> np.ndarray:
    """Bounded logistic gate W(g) in [0, 1] (numerically stable)."""
    x = (np.asarray(g, dtype=float) - g_thr_tide) / width_tide
    out = np.empty_like(x)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _obs_log_ratio(g: np.ndarray, z_sum: float, p: ModelParams) -> np.ndarray:
    """Deterministic kernel exponent y0(g) = b0 + b1 g**npow + W(g) z_sum.

    Works on scalars or arrays. This mirrors ``ec_boundary.ec_kernel`` exactly
    (minus the exp/affine EC mapping and the bias state).
    """
    g = np.asarray(g, dtype=float)
    gate = _tide_gate(g, p.g_thr_tide, p.width_tide)
    return p.beta0 + p.beta1 * np.power(g, p.npow) + gate * z_sum


def _ec_from_y(y: np.ndarray, p: ModelParams) -> np.ndarray:
    """Map log-ratio y back to EC: EC = exp(y) (so - sb) + sb."""
    return np.exp(np.asarray(y, dtype=float)) * (p.so - p.sb) + p.sb


def _ec_to_y(ec: np.ndarray, p: ModelParams) -> np.ndarray:
    """Map EC observation to log-ratio y, clipping the ratio into (0, 1)."""
    frac = (np.asarray(ec, dtype=float) - p.sb) / (p.so - p.sb)
    frac = np.clip(frac, 1e-6, 1.0 - 1e-6)
    return np.log(frac)


def _g_step(g_prev: float, q_prev: float, q_cur: float, div2dt: float) -> float:
    """Single Crank-Nicolson g-model step (mirrors ``ec_boundary.g_kernel``).

    Guarded against a negative discriminant for perturbed sigma points; the
    deterministic trajectory never triggers the guard.
    """
    qterm = q_cur - div2dt
    disc = qterm * qterm - 4.0 * (g_prev * g_prev - g_prev * (q_prev + div2dt))
    if disc < 0.0:
        disc = 0.0
    return 0.5 * (qterm + np.sqrt(disc))


# --------------------------------------------------------------------------- #
# Drivers: reuse the deterministic operator (no logic duplication)
# --------------------------------------------------------------------------- #
def compute_drivers(
    ndo: pd.Series,
    elev: pd.Series,
    model_yaml: str,
    start=None,
    end=None,
) -> tuple[pd.DataFrame, ModelParams]:
    """Compute exogenous drivers and fixed model params for the filter.

    Returns
    -------
    drivers : pandas.DataFrame
        Indexed by the evaluation timestamps, with columns:
        ``q`` (modified NDO forcing), ``z_sum`` (lagged tidal design),
        ``g_det`` (deterministic transport), ``ec_det`` (deterministic EC).
    params : ModelParams
        Fixed kernel/g-model parameters (with dt inferred from the index).
    """
    ec_det, comp = ec_est_yaml(
        ndo, elev, model_yaml, start=start, end=end, return_components=True
    )
    idx = comp.index
    dt_sec = float(pd.Timedelta(idx.freq or idx.to_series().diff().iloc[1]).total_seconds())
    drivers = pd.DataFrame(
        {
            "q": comp["q"].to_numpy(dtype=float),
            "z_sum": comp["z_sum"].to_numpy(dtype=float),
            "g_det": comp["g"].to_numpy(dtype=float),
            "ec_det": ec_det.to_numpy(dtype=float),
        },
        index=idx,
    )
    params = ModelParams.from_yaml(model_yaml, dt_sec=dt_sec)
    return drivers, params


# --------------------------------------------------------------------------- #
# Unscented transform helpers (n = 2)
# --------------------------------------------------------------------------- #
def _ut_weights(n: int, cfg: FilterConfig) -> tuple[np.ndarray, np.ndarray, float]:
    lam = cfg.alpha ** 2 * (n + cfg.kappa) - n
    c = n + lam
    wm = np.full(2 * n + 1, 1.0 / (2.0 * c))
    wc = np.full(2 * n + 1, 1.0 / (2.0 * c))
    wm[0] = lam / c
    wc[0] = lam / c + (1.0 - cfg.alpha ** 2 + cfg.beta_ut)
    return wm, wc, c


def _sigma_points(m: np.ndarray, P: np.ndarray, c: float, jitter: float) -> np.ndarray:
    n = m.shape[0]
    L = np.linalg.cholesky(c * P + jitter * np.eye(n))
    sig = np.empty((2 * n + 1, n))
    sig[0] = m
    for i in range(n):
        sig[1 + i] = m + L[:, i]
        sig[1 + n + i] = m - L[:, i]
    return sig


# --------------------------------------------------------------------------- #
# Forward UKF + URTS smoother
# --------------------------------------------------------------------------- #
def run_smoother(
    ndo: pd.Series,
    elev: pd.Series,
    model_yaml: str,
    ec_obs: pd.Series,
    cfg: FilterConfig | None = None,
    start=None,
    end=None,
) -> pd.DataFrame:
    """Run the UKF forward filter + URTS smoother to impute/denoise MRZ EC.

    Parameters
    ----------
    ndo, elev : pandas.Series
        Deterministic drivers (see ``mrzecest.ec_boundary``). ``elev`` must be
        padded (>= 9 days each side) to support filtering.
    model_yaml : str
        Path to the fitted model YAML (parameters held fixed).
    ec_obs : pandas.Series
        Observed Martinez EC (uS/cm). Missing samples (gaps to impute) are
        indicated by NaN or simply by absence from the index; both are honored.
    cfg : FilterConfig, optional
        Filter tuning. Defaults to :class:`FilterConfig`.
    start, end : optional
        Evaluation window (passed through to the deterministic operator).

    Returns
    -------
    pandas.DataFrame
        Indexed by evaluation timestamps with columns:
        ``ec_obs`` (assimilated, NaN where absent), ``ec_det`` (deterministic),
        ``ec_filt`` / ``ec_smooth`` (filtered / smoothed EC),
        ``ec_lo`` / ``ec_hi`` (95% smoothed band),
        ``g_filt`` / ``g_smooth``, ``b_filt`` / ``b_smooth`` (log-bias state).
    """
    cfg = cfg or FilterConfig()
    drivers, p = compute_drivers(ndo, elev, model_yaml, start=start, end=end)
    idx = drivers.index
    N = len(idx)
    q = drivers["q"].to_numpy()
    z = drivers["z_sum"].to_numpy()
    g_det = drivers["g_det"].to_numpy()

    # Observations aligned to the eval index -> y space with a validity mask.
    obs = ec_obs.reindex(idx)
    y_obs = _ec_to_y(obs.to_numpy(dtype=float), p)
    has_obs = np.isfinite(obs.to_numpy(dtype=float))

    # --- discrete-time noise / dynamics constants ---
    dt_days = p.dt_sec / 86400.0
    var_logg = (cfg.q_logg_per_day ** 2) * dt_days
    phi = float(np.exp(-p.dt_sec / (cfg.tau_b_days * 86400.0)))
    var_b = cfg.sigma_b ** 2 * (1.0 - phi ** 2)
    Q = np.diag([var_logg, var_b])
    R = cfg.r_std ** 2

    n = 2
    wm, wc, c = _ut_weights(n, cfg)

    # --- storage for the backward smoother ---
    m_filt = np.zeros((N, n))
    P_filt = np.zeros((N, n, n))
    m_pred = np.zeros((N, n))
    P_pred = np.zeros((N, n, n))
    D_cross = np.zeros((N, n, n))  # Cov(x_{k-1|k-1}, x_{k|k-1})

    def h_sigma(sig: np.ndarray, k: int) -> np.ndarray:
        g = np.exp(sig[:, 0])
        return _obs_log_ratio(g, z[k], p) + sig[:, 1]

    # --- initial state ---
    # NOTE: the deterministic g_kernel labels g[0] = g0 but its integration uses
    # ndo[0] (= q[0]) as the effective "previous g" entering the first step. We
    # seed the filter's transport state with q[0] so the propagated trajectory
    # matches the deterministic g-model exactly (g0 is only a display label).
    m = np.array([np.log(max(q[0], 1.0)), 0.0])
    P = np.diag([cfg.init_logg_std ** 2, cfg.init_b_std ** 2])
    if has_obs[0]:
        m, P = _update(m, P, y_obs[0], 0, h_sigma, wm, wc, c, R, cfg.jitter)
    m_filt[0], P_filt[0] = m, P
    m_pred[0], P_pred[0] = m, P

    # --- forward pass ---
    for k in range(1, N):
        # predict k-1 -> k
        sig = _sigma_points(m_filt[k - 1], P_filt[k - 1], c, cfg.jitter)
        fsig = np.empty_like(sig)
        for j in range(sig.shape[0]):
            g_prev = np.exp(sig[j, 0])
            g_new = _g_step(g_prev, q[k - 1], q[k], p.div2dt)
            fsig[j, 0] = np.log(max(g_new, 1e-6))
            fsig[j, 1] = phi * sig[j, 1]
        mp = wm @ fsig
        dp = fsig - mp
        Pp = (wc[:, None, None] * dp[:, :, None] * dp[:, None, :]).sum(axis=0) + Q
        d0 = sig - m_filt[k - 1]
        Dk = (wc[:, None, None] * d0[:, :, None] * dp[:, None, :]).sum(axis=0)
        m_pred[k], P_pred[k], D_cross[k] = mp, Pp, Dk

        # update with observation (if present)
        if has_obs[k]:
            m, P = _update(mp, Pp, y_obs[k], k, h_sigma, wm, wc, c, R, cfg.jitter)
        else:
            m, P = mp, Pp
        m_filt[k], P_filt[k] = m, P

    # --- backward URTS smoother ---
    m_smooth = m_filt.copy()
    P_smooth = P_filt.copy()
    for k in range(N - 2, -1, -1):
        Pp = P_pred[k + 1] + cfg.jitter * np.eye(n)
        G = D_cross[k + 1] @ np.linalg.inv(Pp)
        m_smooth[k] = m_filt[k] + G @ (m_smooth[k + 1] - m_pred[k + 1])
        P_smooth[k] = P_filt[k] + G @ (P_smooth[k + 1] - P_pred[k + 1]) @ G.T

    # --- reconstruct EC (with 95% band via unscented transform of the state) ---
    ec_filt, _ = _reconstruct_ec(m_filt, P_filt, z, p, wm, wc, c, cfg.jitter)
    ec_smooth, ec_band = _reconstruct_ec(
        m_smooth, P_smooth, z, p, wm, wc, c, cfg.jitter
    )

    out = pd.DataFrame(
        {
            "ec_obs": obs.to_numpy(dtype=float),
            "ec_det": drivers["ec_det"].to_numpy(),
            "ec_filt": ec_filt,
            "ec_smooth": ec_smooth,
            "ec_lo": ec_band[:, 0],
            "ec_hi": ec_band[:, 1],
            "g_filt": np.exp(m_filt[:, 0]),
            "g_smooth": np.exp(m_smooth[:, 0]),
            "b_filt": m_filt[:, 1],
            "b_smooth": m_smooth[:, 1],
        },
        index=idx,
    )
    return out


def _update(m, P, y, k, h_sigma, wm, wc, c, R, jitter):
    """Scalar-measurement unscented update."""
    n = m.shape[0]
    sig = _sigma_points(m, P, c, jitter)
    ysig = h_sigma(sig, k)
    y_mean = wm @ ysig
    dy = ysig - y_mean
    Pyy = float((wc * dy * dy).sum() + R)
    dx = sig - m
    Pxy = (wc[:, None] * dx * dy[:, None]).sum(axis=0)  # (n,)
    K = Pxy / Pyy
    m_new = m + K * (y - y_mean)
    P_new = P - np.outer(K, K) * Pyy
    return m_new, P_new


def _reconstruct_ec(m_arr, P_arr, z, p, wm, wc, c, jitter):
    """Unscented transform of the state -> EC mean and 95% band."""
    N = m_arr.shape[0]
    ec_mean = np.empty(N)
    band = np.empty((N, 2))
    for k in range(N):
        sig = _sigma_points(m_arr[k], P_arr[k], c, jitter)
        g = np.exp(sig[:, 0])
        ysig = _obs_log_ratio(g, z[k], p) + sig[:, 1]
        y_mean = float(wm @ ysig)
        dy = ysig - y_mean
        y_var = float((wc * dy * dy).sum())
        y_sd = np.sqrt(max(y_var, 0.0))
        ec_mean[k] = _ec_from_y(y_mean, p)
        band[k, 0] = _ec_from_y(y_mean - 1.96 * y_sd, p)
        band[k, 1] = _ec_from_y(y_mean + 1.96 * y_sd, p)
    return ec_mean, band


# --------------------------------------------------------------------------- #
# Synthetic-gap tooling for validation
# --------------------------------------------------------------------------- #
def make_gap_mask(
    index: pd.DatetimeIndex,
    gaps: list[tuple[str, int]],
    seed: int = 0,
    edge_pad: str = "3D",
) -> pd.Series:
    """Build a boolean held-out mask by punching random gaps of given lengths.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        The evaluation index.
    gaps : list of (length, count)
        Each entry punches ``count`` non-overlapping gaps of duration
        ``length`` (a pandas-parseable offset string, e.g. ``"12h"``, ``"3d"``).
    seed : int
        RNG seed for reproducibility.
    edge_pad : str
        Keep gaps at least this far from the series ends.

    Returns
    -------
    pandas.Series
        Boolean series aligned to ``index``; True where the sample is *held out*
        (to be treated as missing / imputed).
    """
    rng = np.random.default_rng(seed)
    held = pd.Series(False, index=index)
    pad = pd.Timedelta(edge_pad)
    lo, hi = index[0] + pad, index[-1] - pad
    for length, count in gaps:
        dur = pd.Timedelta(length)
        for _ in range(count):
            for _try in range(100):
                t0 = lo + pd.Timedelta(
                    seconds=rng.uniform(0, max((hi - lo - dur).total_seconds(), 0))
                )
                sl = (index >= t0) & (index < t0 + dur)
                if sl.any() and not held[sl].any():
                    held[sl] = True
                    break
    return held


def gap_metrics(result: pd.DataFrame, truth: pd.Series, held: pd.Series) -> dict:
    """Compute imputation error over held-out samples (EC and log space)."""
    idx = result.index
    truth = truth.reindex(idx)
    held = held.reindex(idx).fillna(False).to_numpy()
    yhat = result["ec_smooth"].to_numpy()
    ytru = truth.to_numpy(dtype=float)
    det = result["ec_det"].to_numpy()
    m = held & np.isfinite(ytru)
    if not m.any():
        return {"n": 0}
    err = yhat[m] - ytru[m]
    err_det = det[m] - ytru[m]
    lerr = np.log(np.clip(yhat[m], 1e-6, None)) - np.log(np.clip(ytru[m], 1e-6, None))
    return {
        "n": int(m.sum()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "rmse_det": float(np.sqrt(np.mean(err_det ** 2))),
        "bias": float(np.mean(err)),
        "rmse_log": float(np.sqrt(np.mean(lerr ** 2))),
    }
