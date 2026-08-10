"""Compare the baseline Gamma-GLM/deviance fit against a least-squares
'touch-up' of the same model.

The touch-up is a *local* minimization (scipy least_squares, TRF) of the plain
sum-of-squared EC residuals, seeded at the GLM outcome, over all model
parameters simultaneously. The stratification gating geometry (front spec) is
frozen so the refinement is well defined up to bounds-hugging.

We compare the two fits with three metrics that span different error
philosophies:
    - RMSE  : the touch-up (LS) criterion
    - MAE   : a neutral L1 norm matching neither fit
    - Gamma deviance : the baseline GLM criterion

Metrics are reported both on the fit window (in-sample) and on an out-of-sample
window that lies entirely after the fit period.
"""

import pandas as pd
import matplotlib.pyplot as plt

from vtools import hours
from ndo_chooser import get_ndo

from mrzecest.ec_boundary_fit import fit_mrz_ecest, touchup_least_squares
from mrzecest.ec_boundary import ec_est_yaml
from mrzecest.fitting_util import (
    build_model_from_fit,
    write_model_yaml,
    parse_config,
    compare_metrics,
)
import logging


def _load_inputs():
    ndo_source = "dayflow"  # "dsm2" or "dayflow"
    ndo15 = get_ndo(ndo_source, "15min")

    elev = pd.read_csv(
        "./data/dms_mrz_elev_filled.csv",
        header=0,
        index_col=0,
        parse_dates=["datetime"],
    ).asfreq("15min")

    obs_ec = pd.read_csv(
        "./data/dms_mrz@upper_ec.csv",
        header=0,
        index_col=0,
        parse_dates=["datetime"],
        comment="#",
    )
    obs_ec15 = obs_ec.interpolate(limit=4)
    obs_ec15 = obs_ec15.clip(lower=201).resample("15min").asfreq("15min")

    valid_start = pd.Timestamp(2004, 1, 1)
    valid_end = pd.Timestamp(2025, 4, 1)
    ndo15 = ndo15.loc[valid_start:valid_end]
    elev = elev.loc[(valid_start - hours(96)) : valid_end + hours(99)]
    obs_ec = obs_ec.loc[valid_start:valid_end]
    obs_ec15 = obs_ec15.loc[valid_start:valid_end]
    return ndo15, elev, obs_ec, obs_ec15


def _metrics_table(name, obs, pred, so, sb):
    m = compare_metrics(obs, pred, so, sb)
    print(
        f"  {name:<10s}  RMSE={m['rmse']:>10.1f}  MAE={m['mae']:>10.1f}  "
        f"GammaDev={m['gamma_deviance']:>12.3f}  n={m['n']}"
    )
    return m


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = "fitting_config.yaml"
    cfg = parse_config(config)
    so_cfg = float(cfg["so"])
    sb_cfg = float(cfg["sb"])

    ndo15, elev, obs_ec, obs_ec15 = _load_inputs()
    print("Done with load")

    # --- baseline Gamma-GLM/deviance fit -------------------------------------
    x_res, coefs, pred_df, front_spec = fit_mrz_ecest(
        config, elev=elev, ndo=ndo15, ec_obs=obs_ec15
    )
    model_glm = build_model_from_fit(config, x_res, coefs, front_spec=front_spec)
    write_model_yaml(model_glm, "model.yaml")
    print("Done with baseline fit -> model.yaml")

    # --- least-squares touch-up (local, seeded at GLM outcome) ---------------
    model_ls, ls_result = touchup_least_squares(
        config, model_glm, elev=elev, ndo=ndo15, ec_obs=obs_ec15
    )
    write_model_yaml(model_ls, "model_ls.yaml")
    print(
        f"Done with LS touch-up -> model_ls.yaml (success={ls_result.success}, "
        f"cost={ls_result.cost:.6g})"
    )

    # --- evaluation windows --------------------------------------------------
    # In-sample = fit window; out-of-sample = strictly after the fit period.
    # Dayflow NDO covers 2000 .. 2024-10, so the holdout runs to 2024-09.
    fit_run = cfg["fit_run"]
    fit_start = pd.Timestamp(fit_run["start"])
    fit_end = pd.Timestamp(fit_run["end"])
    oos_start = pd.Timestamp("2018-06-01")
    oos_end = pd.Timestamp("2024-09-15")

    oos_start = pd.Timestamp("2004-06-05")
    oos_end = pd.Timestamp("2016-12-29")

    pad = pd.Timedelta("9d")

    def _eval(model_yaml, start, end):
        ndo_in = ndo15.loc[start:end]
        elev_in = elev.loc[start - pad : end + pad]
        return ec_est_yaml(ndo_in, elev_in, model_yaml)

    so = float(model_glm["so"])  # for deviance normalization (GLM so)

    windows = [
        ("IN-SAMPLE (fit)", fit_start, fit_end),
        ("OUT-OF-SAMPLE", oos_start, oos_end),
    ]

    results = {}
    for label, start, end in windows:
        print(f"\n{label}: {start.date()} .. {end.date()}")
        est_glm = _eval("model.yaml", start, end)
        est_ls = _eval("model_ls.yaml", start, end)
        obs_win = obs_ec.loc[start:end].squeeze()
        _metrics_table("GLM", obs_win, est_glm, so, sb_cfg)
        _metrics_table("LS", obs_win, est_ls, so, sb_cfg)
        results[label] = (obs_win, est_glm, est_ls)

    # --- plots ---------------------------------------------------------------
    fig, axes = plt.subplots(len(windows), 1, figsize=(12, 8), sharex=False)
    for ax, (label, start, end) in zip(axes, windows):
        obs_win, est_glm, est_ls = results[label]
        ax.plot(obs_win.index, obs_win.values, label="obs", color="0.3", lw=1.0)
        ax.plot(est_glm.index, est_glm.values, label="GLM", lw=1.0)
        ax.plot(est_ls.index, est_ls.values, label="LS touch-up", lw=1.0)
        ax.set_title(label)
        ax.set_ylabel("EC (uS/cm)")
        ax.legend(loc="upper right")
    fig.tight_layout()

    # Residual-vs-level scatter (out-of-sample) to expose the character shift.
    obs_win, est_glm, est_ls = results["OUT-OF-SAMPLE"]
    common = obs_win.dropna().index.intersection(est_glm.dropna().index)
    common = common.intersection(est_ls.dropna().index)
    o = obs_win.loc[common].to_numpy()
    fig2, ax2 = plt.subplots(1, figsize=(7, 6))
    ax2.scatter(o, (est_glm.loc[common].to_numpy() - o), s=3, alpha=0.3, label="GLM")
    ax2.scatter(o, (est_ls.loc[common].to_numpy() - o), s=3, alpha=0.3, label="LS")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xlabel("observed EC (uS/cm)")
    ax2.set_ylabel("residual pred - obs (uS/cm)")
    ax2.set_title("Residual vs level (out-of-sample)")
    ax2.legend()
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
