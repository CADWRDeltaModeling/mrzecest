from mrzecest.ec_boundary_fit import fit_mrz_ecest
from mrzecest.ec_boundary import ec_est_yaml
from mrzecest.fitting_util import build_model_from_fit, write_model_yaml

from vtools import rhistinterp, hours
import pandas as pd
import matplotlib.pyplot as plt
from dms_datastore.read_ts import read_ts
from ndo_chooser import get_ndo
import numpy as np
import logging


def main():

    logging.basicConfig(
      level=logging.INFO,   # or DEBUG
      format="%(levelname)s %(name)s: %(message)s",
    )

    config = "fitting_config.yaml"
    ndo_source = "dsm2"  # "dsm2" or "dayflow"
    ndo15 = get_ndo(ndo_source, "15min")

    elev = pd.read_csv(
        "./data/dms_mrz_elev_filled.csv",
        header=0,
        index_col=0,
        parse_dates=["datetime"],
    )
    elev = elev.asfreq("15min")
    obs_ec = pd.read_csv(
        "./data/dms_mrz@upper_ec.csv",
        header=0,
        index_col=0,
        parse_dates=["datetime"],
        comment="#",
    )
    obs_ec15 = obs_ec.resample("15min").interpolate(limit=4)
    obs_ec15 = obs_ec.interpolate(limit=4)
    obs_ec15 = obs_ec15.clip(lower=201).resample("15min").asfreq("15min")

    ndo15_valid = ndo15.dropna(how="all")
    if ndo15_valid.empty:
        raise ValueError("ndo15 has no valid data to determine bounds.")

    valid_start = pd.Timestamp(2004, 1, 1)
    valid_end = pd.Timestamp(2025, 4, 1)
    ndo15 = ndo15.loc[valid_start:valid_end]
    elev = elev.loc[
        (valid_start - hours(96)) : valid_end + hours(99)
    ]  # extra buffer for filtering
    obs_ec = obs_ec.loc[valid_start:valid_end]
    obs_ec15 = obs_ec15.loc[valid_start:valid_end]

    x_res, coefs, pred_df, front_spec = fit_mrz_ecest(
        config, elev=elev, ndo=ndo15, ec_obs=obs_ec15
    )

    # Build the canonical model dict from the fit result and fitting config.
    # This keeps constants (so/sb) and filter setup single-sourced.
    model = build_model_from_fit(config, x_res, coefs, front_spec=front_spec)
    write_model_yaml(model, "model.yaml")

    eval_start = pd.Timestamp("2006-01-01")
    eval_end = pd.Timestamp("2025-01-01")

    pad = pd.Timedelta("9d")  # documented example choice

    ndo_in = ndo15.loc[eval_start:eval_end]
    elev_in = elev.loc[eval_start - pad : eval_end + pad]

    mrzecest = ec_est_yaml(ndo_in, elev_in, "model.yaml")

    ####
    obs_ec_plot = obs_ec.loc[eval_start:eval_end]

    fig, ax = plt.subplots(1)
    ax.plot(obs_ec_plot.index, obs_ec_plot.values)
    ax.plot(mrzecest.index, mrzecest.values)
    ax.legend(["obs", "est"])
    plt.show()


if __name__ == "__main__":
    main()
