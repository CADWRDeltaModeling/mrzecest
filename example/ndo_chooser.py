import pandas as pd
from vtools import rhistinterp
from dms_datastore.read_ts import read_ts
import matplotlib.pyplot as plt

import numpy as np


def get_ndo(ndo_source: str, freq) -> pd.Series:
    """Obtain example ndo source from various locations, including Dayflow or DSM2."""
    if ndo_source == "dsm2":
        ndo_dsm = pd.read_csv(
            "./data/dsm2_ndo_hist.csv", header=0, index_col=0, parse_dates=["datetime"]
        ).squeeze()
        ndo = ndo_dsm
    elif ndo_source == "dayflow":
        ndo2 = read_ts(
            "//cnrastore-bdo/Modeling_Data/dayflow/dayflow_ndoi_flow_2000_9999.csv"
        ).loc["2000":"2025",:]
        ndo2 = ndo2.squeeze()
        ndo2.loc[ndo2 < -2000.0] = np.nan
        ndo2 = ndo2.interpolate(limit=4)
        ndo = ndo2

    ndo = ndo.asfreq("d")
    ndo = ndo.to_period("d")
    ndo15 = rhistinterp(ndo, freq, lowbound=-2000.0)
    ndo15 = ndo15.asfreq(freq)
    return ndo15


def main():
    ndo_dsm = get_ndo("dayflow", "15min")
    ndo_dayflow = get_ndo("dsm2", "15min")
    if ndo_dsm is None or ndo_dayflow is None:
        raise ValueError("Expected NDO series.")

    fig, ax = plt.subplots()
    idx_dsm = (
        ndo_dsm.index.to_timestamp()
        if isinstance(ndo_dsm.index, pd.PeriodIndex)
        else ndo_dsm.index
    )
    idx_dayflow = (
        ndo_dayflow.index.to_timestamp()
        if isinstance(ndo_dayflow.index, pd.PeriodIndex)
        else ndo_dayflow.index
    )
    ax.plot(idx_dsm, ndo_dsm.values, label="Dayflow")
    ax.plot(idx_dayflow, ndo_dayflow.values, label="DSM2")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Net Delta Outflow")
    ax.legend()
    fig.autofmt_xdate()
    plt.show()


if __name__ == "__main__":
    main()
