from mrzecest.ec_boundary import *
from mrzecest.fitting_util import *
from vtools import rhistinterp

import pandas as pd

import matplotlib.pyplot as plt

from bokeh.io import show
from bokeh.layouts import column
from bokeh.models import RangeTool
from bokeh.plotting import figure, show, output_file, save
from bokeh.palettes import Magma, Inferno, Plasma, Viridis, Cividis, Colorblind, Bokeh


def main():
    config = "ec_est_config.yaml"
    config = parse_config(config)

    ndo = pd.read_csv(
        "./data/hist_ndo.csv", header=0, index_col=0, parse_dates=["datetime"]
    )
    ndo = ndo.asfreq("d")
    ndo = ndo.to_period()
    ndo = rhistinterp(ndo, "h", lowbound=-2000.0)
    elev = pd.read_csv(
        "./data/mrz_hist_stage.csv", header=0, index_col=0, parse_dates=["datetime"]
    )
    elev = elev.asfreq("h")
    obs_ec = pd.read_csv(
        "./data/mrz_hist_ec.csv", header=0, index_col=0, parse_dates=["datetime"]
    )
    obs_ec = obs_ec.asfreq("h")
    obs_ec = obs_ec.interpolate(limit=4)

    # align the ndo and elev dataframes
    common_index = elev.index.intersection(ndo.index)
    ndo = ndo.loc[common_index]
    elev = elev.loc[common_index]

    start = pd.to_datetime(config["start"])
    end = pd.to_datetime(config["end"])

    # parameters from estimation
    log10beta = 10.217  # x[0] from ec_boundary_fit_gee.py printout
    npow = 0.461  # x[1] from ec_boundary_fit_gee.py printout
    area_coef = -6127433509.04  # x[2] from ec_boundary_fit_gee.py printout
    energy_coef = 1495.91  # x[3] from ec_boundary_fit_gee.py printout
    beta0 = 1.6828  # from const coef result
    beta1 = -23.0735 * 1e-3  # from gnpow coef result
    filter_k0 = 6  # from fitting_config.yaml
    filt_coefs = (
        np.array(
            [
                0.111,
                0.896,
                -0.606,
                0.678,
                -0.745,
                -0.416,
                -0.046,
                1.161,
                0.321,
                -1.069,
                0.515,
                -0.965,
                0.576,
            ]
        )
        * 1e-3
    )  # z{n} from output coefs
    filter_dt = pd.Timedelta("3h")  # from fitting_config.yaml
    so = 20000.0  # hardwired in ec_boundary_fit_gee.py
    sb = 200.0  # hardwired in ec_boundary_fit_gee.py

    # ec_est(ndo, elev, start, end, area_coef,energy_coef, log10beta, beta0, beta1, npow, filter_k0, filt_coefs, filter_dt, so, sb))
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
    obs_ec = obs_ec.loc[start:end]

    # for debugging ---------------------------------------------------------------------
    plt_dicts = {"EC (uS/cm)": {"Historic Input": obs_ec, "Estimated EC": mrzecest}}

    plot_dicts(
        plt_dicts, "EC (uS/cm)"
    )  # ------------------------------------------------

    print("debug")


def plot_dicts(plot_dicts, range_dict_key, save_filename=None, date_range=None):
    """generic plot function

    Parameters
    ----------
    plot_dicts: dict

        dictionary of dictionaries. each dictionary uses they key as the plot pane title and has dataframes within it whose keys correspond to the legend label. Ex: plot_dicts = {'Stage':{'Original': stage_df, 'SLR': slr_stage_df}, 'Flow':{'Net Delta Outflow':ndo_df}}

    range_dict_key: str

        name of plot panel to use as range slider. Needs to be a key that exists in plot_dicts (ex: 'Stage')

    """

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    # default colors same as sns.color_palette("tab10", n_colors=10).as_hex() # '#d62728' is red

    plt_elem_names = list(plot_dicts.keys())

    h_plt = int(1000 / len(plt_elem_names))

    for key, pdict in plot_dicts.items():
        keys = list(pdict.keys())

        min_y, max_y = np.inf, -np.inf
        if date_range is not None:
            for k, df in pdict.items():
                pdict[k] = pd.Series(df.loc[date_range[0] : date_range[1]])
        for k, df in pdict.items():
            if min(df.values) < min_y:
                min_y = min(df.values)
            if max(df.values) > max_y:
                max_y = max(df.values)

        max_y = 55000
        # get the first DateTime in the dataframe index
        first = pdict[keys[0]].index[0]
        # get the last DateTime in the dataframe index
        last = pdict[keys[0]].index[-1]

        plt_elem = figure(
            height=h_plt,
            width=1200,
            tools="xpan,wheel_zoom,box_zoom,reset,save,hover",
            x_axis_type="datetime",
            x_axis_location="above",
            background_fill_color="#efefef",
            x_range=(first, last),
            y_range=(min_y - (0.05 * min_y), max_y + (0.05 * max_y)),
            title=key,
        )

        for k, df in pdict.items():
            plt_elem.line(
                df.index,
                df.loc[:],
                color=colors[keys.index(k)],
                legend_label=k,
                line_width=3,
            )
        if key == plt_elem_names[0]:
            plt_elems = [plt_elem]
        else:
            plt_elems.append(plt_elem)

    # set x_ranges to match
    for key, pdict in plot_dicts.items():
        if key != plt_elem_names[0]:
            plt_elems[plt_elem_names.index(key)].x_range = plt_elems[0].x_range

    # create scroller with range_dict_key
    select = figure(
        title="Drag the middle and edges of the selection box to change the range",
        height=130,
        width=1200,
        y_range=plt_elems[plt_elem_names.index(range_dict_key)].y_range,
        x_axis_type="datetime",
        y_axis_type=None,
        tools="",
        toolbar_location=None,
        background_fill_color="#efefef",
    )

    range_tool = RangeTool(x_range=plt_elems[0].x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    keys = list(plot_dicts[range_dict_key].keys())
    for key, df in plot_dicts[range_dict_key].items():
        select.line(
            df.index, df.loc[:], color=colors[keys.index(key)], legend_label=key
        )

    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)

    plot_final = column(*(plt_elems + [select]))
    if save_filename is not None:
        output_file(save_filename, mode="inline")
        save(plot_final)
    else:
        show(plot_final)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
