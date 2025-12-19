"""Example: run MRZ EC estimator from YAML model config and plot results (Bokeh).

This is intentionally a *script* (no CLI): edit constants below.

It mirrors the *data + prep* approach used in fitting_example.py:
  - NDO: daily values treated as period averages; conservative upsample via vtools.rhistinterp
  - Stage: 15-min (already filled) and padded to support the estimator's filters
  - Observed EC: used only for comparison in plots

Notes on Bokeh:
  - In Bokeh >= 3.6 (per migration notes), RangeTool is active by default and is no longer
    a "multi GestureTool", so do NOT set toolbar.active_multi to the RangeTool.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from bokeh.io import output_file, save, show
from bokeh.layouts import column
from bokeh.models import RangeTool
from bokeh.plotting import figure

from vtools import rhistinterp, hours

from mrzecest.ec_boundary import ec_est_yaml


# -----------------
# User-edit settings
# -----------------
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MODEL_YAML = HERE / "model.yaml"

# Evaluation window (inclusive endpoints for slicing)
EVAL_START = pd.Timestamp("2006-01-01")
EVAL_END = pd.Timestamp("2025-01-01")

# Padding for stage around the evaluation window to support any internal filtering
# (fitting_example.py uses 9d after discovering edge effects)
STAGE_PAD = pd.Timedelta("9d")

# Plot output: set WRITE_HTML=True to save an HTML file; otherwise opens a browser tab
WRITE_HTML = False
HTML_OUT = HERE / "ec_est_example.html"


def _read_inputs_15min() -> tuple[pd.Series, pd.Series, pd.Series]:
    """Read NDO, stage, and observed EC using the same approach as fitting_example.py."""

    # --- NDO: daily -> PeriodIndex -> conservative upsample to 15-min
    ndo = pd.read_csv(
        DATA_DIR / "dsm2_ndo_hist.csv",
        header=0,
        index_col=0,
        parse_dates=["datetime"],
    ).iloc[:, 0]

    ndo = ndo.asfreq("d")
    ndo = ndo.to_period("d")
    ndo15 = rhistinterp(ndo, "15min", lowbound=-2000.0)
    ndo15 = ndo15.asfreq("15min")


    # --- Stage (already filled)
    elev = pd.read_csv(
        DATA_DIR / "dms_mrz_elev_filled.csv",
        header=0,
        index_col=0,
        parse_dates=["datetime"],
    ).iloc[:, 0]
    elev = elev.asfreq("15min")

    # --- Observed EC (for comparison only)
    obs_ec = pd.read_csv(
        DATA_DIR / "dms_mrz@upper_ec.csv",
        header=0,
        index_col=0,
        parse_dates=["datetime"],
        comment="#",
    ).iloc[:, 0]

    # fitting_example.py does a short interpolation then clips to >= 201
    obs_ec15 = obs_ec.resample("15min").interpolate(limit=4)
    obs_ec15 = obs_ec15.clip(lower=201).resample("15min").asfreq("15min")

    return ndo15, elev, obs_ec15


def _validate_timebase(name: str, s: pd.Series, freq: str) -> None:
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex, got {type(s.index)}")
    if s.index.freq is None:
        # Pandas often drops .freq; enforce via inferred or explicit asfreq in upstream prep
        inferred = pd.infer_freq(s.index)
        if inferred != freq:
            raise ValueError(
                f"{name} index has no .freq and inferred_freq={inferred!r}, expected {freq!r}"
            )
    # also ensure uniform spacing
    deltas = s.index.to_series().diff().dropna()
    expected = pd.Timedelta(freq)
    if not (deltas == expected).all():
        bad = deltas[deltas != expected].iloc[:5]
        raise ValueError(
            f"{name} is not uniformly spaced at {freq}. First mismatches:\n{bad}"
        )


def _plot_bokeh(
    *,
    ec_obs: pd.Series,
    ec_est: pd.Series,
    elev: pd.Series,
    ndo: pd.Series,
    title: str = "MRZ EC estimate",
):
    """Three stacked time-series panels + a RangeTool scroller."""

    # Align x-range limits to the evaluation window
    first = ec_est.index[0]
    last = ec_est.index[-1]

    def _panel(name: str, y_label: str, series_dict: dict[str, pd.Series], y_range=None):
        fig_kwargs = dict(
            height=260,
            width=1200,
            tools="xpan,wheel_zoom,box_zoom,reset,save,hover",
            x_axis_type="datetime",
            x_axis_location="above",
            background_fill_color="#efefef",
            x_range=(first, last),
            title=name,
        )
        if y_range is not None:
            fig_kwargs["y_range"] = y_range

        p = figure(**fig_kwargs)
        colors = ["black", "firebrick", "steelblue", "seagreen", "darkorange"]
        for i, (k, s) in enumerate(series_dict.items()):
            p.line(s.index, s.values, line_width=2, color=colors[i % len(colors)], legend_label=k)
        p.yaxis.axis_label = y_label
        p.legend.click_policy = "hide"
        return p

    p_ec = _panel(
        f"{title}: EC",
        "EC (uS/cm)",
        {"obs": ec_obs, "est": ec_est},
        y_range=(0, 55000),
    )
    p_elev = _panel("Stage", "Stage", {"elev": elev})
    p_ndo = _panel("NDO", "NDO", {"ndo": ndo})

    # Link x ranges
    p_elev.x_range = p_ec.x_range
    p_ndo.x_range = p_ec.x_range

    # Scroller (use EC panel's y-range for context)
    select = figure(
        title="Drag the selection box to change the time window",
        height=140,
        width=1200,
        y_range=p_ec.y_range,
        x_axis_type="datetime",
        y_axis_type=None,
        tools="",
        toolbar_location=None,
        background_fill_color="#efefef",
    )

    # Plot both obs+est in the scroller for context
    select.line(ec_obs.index, ec_obs.values, line_width=1.5, legend_label="obs")
    select.line(ec_est.index, ec_est.values, line_width=1.5, legend_label="est")
    select.ygrid.grid_line_color = None

    range_tool = RangeTool(x_range=p_ec.x_range)
    range_tool.overlay.fill_alpha = 0.2
    select.add_tools(range_tool)

    # IMPORTANT: Do NOT set select.toolbar.active_multi = range_tool
    # RangeTool is active by default in newer Bokeh and is no longer a multi GestureTool.

    layout = column(p_ec, p_elev, p_ndo, select)
    if WRITE_HTML:
        output_file(str(HTML_OUT), mode="inline", title="ec_est example")
        save(layout)
    else:
        show(layout)


def main() -> None:

    if not MODEL_YAML.exists():
        raise FileNotFoundError("model.yaml not found. Run fitting_example.py first.")

    ndo15, elev15, obs_ec15 = _read_inputs_15min()

    # Slice like fitting_example (and keep padding for elev)
    ndo_in = ndo15.loc[EVAL_START:EVAL_END]
    elev_in = elev15.loc[EVAL_START - STAGE_PAD : EVAL_END + STAGE_PAD]

    # Estimator expects timebase consistency; keep strict
    _validate_timebase("ndo_in", ndo_in, "15min")
    _validate_timebase("elev_in", elev_in, "15min")

    ec_est = ec_est_yaml(ndo_in, elev_in, str(MODEL_YAML))
    print("ndo_in:", ndo_in.index[0], ndo_in.index[-1])
    print("elev_in:", elev_in.index[0], elev_in.index[-1])
    print("ec_est:", ec_est.index[0], ec_est.index[-1])


    # Plot comparisons only over eval window
    ec_est = ec_est.loc[EVAL_START:EVAL_END]
    ec_obs = obs_ec15.loc[EVAL_START:EVAL_END]
    elev_plot = elev15.loc[EVAL_START:EVAL_END]


    _plot_bokeh(ec_obs=ec_obs, ec_est=ec_est, elev=elev_plot, ndo=ndo_in)


if __name__ == "__main__":
    os.chdir(str(HERE))
    main()
