import pyhecdss
from mrzecest.ec_boundary import ec_est
from dms_datastore.read_ts import read_noaa
from dms_datastore.read_multi import read_ts_repo
from mrzecest.ec_boundary import *
from mrzecest.fitting_util import *
from vtools.functions.unit_conversions import m_to_ft

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
import os

from bokeh.io import show
from bokeh.layouts import column
from bokeh.models import RangeTool
from bokeh.plotting import figure, show, output_file, save
from bokeh.palettes import Magma, Inferno, Plasma, Viridis, Cividis, Colorblind, Bokeh

os.chdir(os.path.dirname(__file__))

# Hypothetical NDO
ndo = pd.read_csv("./data/hypothetical_ndo.csv", parse_dates=[0], index_col=[0])

# Planning Tide
elev = pd.read_csv("./data/planning_tide.csv", parse_dates=[0], index_col=[0])

# Observed EC
mrz_ec = pd.read_csv("./data/mrz_ec.csv", parse_dates=[0], index_col=[0])

# Configuration Parameters
config_fn = "ec_est_params.yaml"

if False:
    # Plot NDO (cfs) and Predicted Stage (ft) over time
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    # Convert PeriodIndex to DatetimeIndex for plotting
    ndo.index = ndo.index.to_timestamp()

    # Plot NDO
    ax1.plot(ndo.index, ndo.values, label="NDO", color="blue")
    ax1.set_ylabel("Flow (cfs)")
    ax1.legend(loc="upper left")
    ax1.grid()

    # Plot MRZ
    ax2.plot(elev.index, elev.values, label="Predicted Stage", color="green")
    ax2.set_ylabel("Stage (ft)")
    ax2.legend(loc="upper left")
    ax2.grid()

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.set_xlabel("")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

# EC Estimation Prep
params = parse_config(config_fn)

# Prep Data
ndo = ndo.asfreq("d")
ndo = ndo.resample("h").interpolate(method="linear")
elev = elev.resample("h").interpolate(method="linear")

common_index = elev.index.intersection(ndo.index)
ndo = ndo.loc[common_index]
elev = elev.loc[common_index]

start = pd.to_datetime("1930-01-01")
end = pd.to_datetime("2020-09-01")

# Load Parameters from Fitting
log10beta = params["log10gbeta"]
npow = params["npow"]
area_coef = params["area_coef"]
energy_coef = params["energy_coef"]
beta0 = params["b0"]
beta1 = params["b1"]
filter_k0 = params["filter_k0"]
filt_coefs = np.array(params["afilt"]) * 1e-3  # Convert to numpy array and scale
filter_dt = pd.Timedelta(params["filter_dt"])
so = params["so"]
sb = params["sb"]

mrzecest = ec_est(
    ndo,
    elev,
    start,
    end,
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
)

# Plot EC
if True:
    # Plot Predicted versus Observed
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # Plot EC
    ax1.plot(mrz_ec.index, mrz_ec.values, label="Observed", color="blue")
    ax1.plot(mrzecest.index, mrzecest.values, label="Estimated", color="red")
    ax1.set_ylabel("EC (uS/cm)")
    ax1.legend(loc="upper left")
    ax1.grid()

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.set_xlabel("")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
