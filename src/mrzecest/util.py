import pandas as pd
import numpy as np

import datetime as dt
import os
import sys

from schimpy import schism_yaml
from schimpy.prepare_schism import process_output_dir, check_nested_match, item_exist

from vtools.functions.filter import cosine_lanczos

from pydelmod.create_ann_inputs import get_dss_data
import pyhecdss

import matplotlib.pyplot as plt

from bokeh.io import show
from bokeh.layouts import column
from bokeh.models import RangeTool
from bokeh.plotting import figure, show, output_file, save
from bokeh.palettes import Magma, Inferno, Plasma, Viridis, Cividis, Colorblind, Bokeh


def calc_dcu(dcd_filename, write_outfile=None):
    """calculate Delta Consumptive Use from DICULFILE inputs from DSM2 setup
    
    Parameters
    ----------
    dcd_filename: str | Path 
    
        name of boundary flow DSM2 input file in DSS format

    write_outfile: str | Path | None
    
        name of output file to write Net Delta Ouflow (df_ndo) out to

    Returns
    -------
    cu_total_dcd : pd.DataFrame
        Data with calculation of Delta Consumptive Use from boundary inputs
    
    """
    
    div_seep_dcd_c_part_dss_filename_dict = {'DIV-FLOW': dcd_filename, 'SEEP-FLOW': dcd_filename}
    drain_dcd_c_part_dss_filename_dict = {'DRAIN-FLOW': dcd_filename}

    df_div_seep_dcd = get_dss_data(div_seep_dcd_c_part_dss_filename_dict, 'c_part', filter_b_part_numeric=True)
    df_drain_dcd = get_dss_data(drain_dcd_c_part_dss_filename_dict, 'c_part', filter_b_part_numeric=True)

    df_div_seep_dcd['dcd_divseep_total']=df_div_seep_dcd[df_div_seep_dcd.columns].sum(axis=1)
    df_drain_dcd['dcd_drain_total']=df_drain_dcd[df_drain_dcd.columns].sum(axis=1)

    cu_total_dcd = pd.merge(df_div_seep_dcd, df_drain_dcd, how='left', left_index=True, right_index=True)

    cu_total_dcd['cu_total'] = cu_total_dcd['dcd_divseep_total'] - cu_total_dcd['dcd_drain_total']

    cu_total_dcd = cu_total_dcd[['cu_total']]

    if write_outfile:
        cu_total_dcd.to_csv(write_outfile, index=True, float_format="%.2f")

    return cu_total_dcd

def calc_filtered(wse_ts, 
                  unit='ft', cutoff_period='40H', padtype='odd', cutoff_frequency=None):
    """calculate filtered tidal signal (subtide) from a stage dataframe
    
    Parameters
    ----------
    wse_ts: pd.DataFrame
    
        dataframe with stage (and only stage) column

    Returns
    -------
    df_filt : pd.DataFrame
        Data with calculation of filtered tidal signal (subtide)
    
    """

    if isinstance(wse_ts.index, pd.core.indexes.datetimes.DatetimeIndex):
        print('timeseries is inst-val, converting to per-aver')
        wse_ts.index = wse_ts.index.to_period()

    # convert to feet
    if unit=="m":
        wse_ts = wse_ts * 3.28084  # ft/m 

    df_filt = cosine_lanczos(wse_ts.copy(), cutoff_period=cutoff_period, padtype=padtype, cutoff_frequency=cutoff_frequency)
    df_filt.columns = ['filtered']

    return df_filt

def calc_ndo(flow_filename, dcd_filename, write_outfile=None):
    """calculate Net Delta Outflow using the BNDRYINPUT and DICULFILE inputs from DSM2 setup
    
    Parameters
    ----------
    flow_filename: str | Path 
    
        name of boundary flow DSM2 input file in DSS format

    dcd_filename: str | Path 
    
        name of boundary flow DSM2 input file in DSS format

    write_outfile: str | Path | None
    
        name of output file to write Net Delta Ouflow (df_ndo) out to

    Returns
    -------
    df_ndo : pd.DataFrame
        Data with calculation of Net Delta Outflow from boundary inputs
    
    """

    cu_total_dcd = calc_dcu(dcd_filename)

    b_part_dss_filename_dict = {'RSAC155': flow_filename, 
                                'RSAN112': flow_filename,
                                'BYOLO040': flow_filename, 
                                'RMKL070': flow_filename, 
                                'RCSM075': flow_filename, 
                                'RCAL009': flow_filename, 
                                'SLBAR002': flow_filename, 
                                'CHSWP003': flow_filename, 
                                'CHDMC004': flow_filename, 
                                'CHVCT001': flow_filename, 
                                'ROLD034': flow_filename, 
                                'CHCCC006': flow_filename}
    df_ndo = get_dss_data(b_part_dss_filename_dict, 'b_part')
    df_ndo = pd.merge(df_ndo, cu_total_dcd, how='left', left_index=True, right_index=True)

    positive_flows = ['RSAC155', 'BYOLO040', 'RMKL070', 'RCSM075', 'RCAL009']
    negative_flows = ['SLBAR002', 'CHSWP003', 'CHDMC004', 'CHVCT001', 'ROLD034', 'CHCCC006', 'cu_total']

    df_ndo['ndo'] = df_ndo[positive_flows].sum(axis=1) - df_ndo[negative_flows].sum(axis=1)
    df_ndo = df_ndo[['ndo']]
    
    if write_outfile:
        df_ndo.to_csv(write_outfile, index=True, float_format="%.2f")

    return df_ndo

def plot_dicts(plot_dicts, range_dict_key, 
                  save_filename=None, date_range=None):
    """generic plot function
    
    Parameters
    ----------
    plot_dicts: dict
    
        dictionary of dictionaries. each dictionary uses they key as the plot pane title and has dataframes within it whose keys correspond to the legend label. Ex: plot_dicts = {'Stage':{'Original': stage_df, 'SLR': slr_stage_df}, 'Flow':{'Net Delta Outflow':ndo_df}}

    range_dict_key: str
    
        name of plot panel to use as range slider. Needs to be a key that exists in plot_dicts (ex: 'Stage')
    
    """
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # default colors same as sns.color_palette("tab10", n_colors=10).as_hex() # '#d62728' is red

    plt_elem_names = list(plot_dicts.keys())

    for key, pdict in plot_dicts.items():
        keys = list(pdict.keys())
        
        min_y, max_y = 9e99, -9e99
        if date_range is not None:
            for k, df in pdict.items():
                pdict[k] = df.loc[date_range[0]:date_range[1]]
        for k, df in pdict.items():
            if min(df.values)[0] < min_y: min_y = min(df.values)[0]
            if max(df.values)[0] > max_y: max_y = max(df.values)[0]

        # get the first DateTime in the dataframe index
        first = pdict[keys[0]].index[0]
        # get the last DateTime in the dataframe index
        last = pdict[keys[0]].index[-1]


        plt_elem = figure(height=300, width=1200, tools = "xpan,wheel_zoom,box_zoom,reset,save,hover",
                          x_axis_type="datetime", x_axis_location="above",
                          background_fill_color="#efefef", 
                          x_range=(first, last), y_range=(min_y-(.05*min_y), max_y+(.05*max_y)),
                          title=key)

        for k, df in pdict.items():
            plt_elem.line(df.index, df.iloc[:,0], color=colors[keys.index(k)], legend_label=k)
        if key==plt_elem_names[0]:
            plt_elems = [plt_elem]
        else:
            plt_elems.append(plt_elem)
    
    # set x_ranges to match
    for key, pdict in plot_dicts.items():
        if key != plt_elem_names[0]:
            plt_elems[plt_elem_names.index(key)].x_range = plt_elems[0].x_range

    # create scroller with range_dict_key  
    select = figure(title="Drag the middle and edges of the selection box to change the range",
                    height=130, width=1200, y_range=plt_elems[plt_elem_names.index(range_dict_key)].y_range,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")

    range_tool = RangeTool(x_range=plt_elems[0].x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    keys = list(plot_dicts[range_dict_key].keys())
    for key, df in plot_dicts[range_dict_key].items():
        select.line(df.index, df.iloc[:,0], color=colors[keys.index(key)], legend_label=key)

    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)

    plot_final = column(*(plt_elems + [select]))
    if save_filename is not None:
        output_file(save_filename, mode='inline')
        save(plot_final)
    else:
        show(plot_final)

def gcalc(ndo,
          beta=1.5e10,
          g0=None):
    """ Calculates antecedent outflow from a stream of ndo

    Parameters
    ----------
        ndo: pd.DataFrame

            a regular time series. Must be 15MIN, 1HOUR. Thus, NDO has been interpolated first.

        beta: float
        
            g-model parameter in cubic feet ((cfs/s)*s)  1.5e9 - R. Denton [cf]
        
        g0: float
        
            initial condition. If g0 is not given it is equal to ndo at the first time step.
            
    Returns
    -------
        g: pd.DataFrame
          
          a regular time series, same sampling rate as input with the same start time as ndo
    """

    ti = ndo.index.freq
    if not ((ti == pd.Timedelta("15MIN")) | (ti == pd.Timedelta("1HOUR"))):
        raise("NDO time step must be 15MIN or 1HOUR.")
    dt = 1
    nstep=len(ndo)
    if ti == pd.Timedelta("15MIN"):
        # dt = 0.25 # hours 
        dt = 15*60 # [s]

    solu_df = ndo.copy()
    solu_df.columns = ['ndo']
    solu_df['g'] = np.nan
    
    div2dt = 2 * beta / dt # units?

    # Set initial condition
    if g0 is None: g0 = solu_df['ndo'].iloc[0]
    solu_df['g'].iloc[0] = g0

    # solve implicitly with trapezoidal method
    for i in range(1, len(solu_df)):
        qterm = solu_df['ndo'].iloc[i] - div2dt
        qpast = solu_df['ndo'].iloc[i - 1]
        gpast = solu_df['g'].iloc[i - 1]

        solu_df['g'].iloc[i] = 0.5 * (qterm + np.sqrt(qterm**2 - 4 * (gpast**2 - gpast * (qpast + div2dt))))

        g = solu_df[['g']]

    return g


def ecest(stage, ndo, g, 
          so = 32000, sb = 200, 
          npow1 = 0.770179, 
          b0 = 1.37e-2, b1 = -6.43e-5, Dt = "3HOUR", k0 = -1,
          min_ec = 200,
          c = [1.59e-4, -1.28e-5, 6.96e-6, 4.83e-5, -7.67e-5, 6.93e-5, -3.85e-5]):
        #   c = [7.30E-05,-1.00E-05,-3.00E-05,1.70E-06,-1.00E-04,4.50E-05,-1.00E-04]):
    """ Estimate 15min EC at the boundary.

    Parameters
    ----------
        stage: pd.DataFrame

            TODO: BOUNDARY (NOT ASTRONOMICAL - ISSUE?) tide estimate. Only 15min data are acceptable

        ndo: pd.DataFrame

            ndo     ndo estimate -- e.g., from CALSIM
        
        g0: float
        
            initial condition. If g0 is not given it is equal to ndo at the first time step.
            
        min_ec: float
        
            Minimum output EC
            
    Returns
    -------
        ec: pd.DataFrame
          
            a regular time series, same sampling rate as input with the same start time as ndo which estimates EC

        gval: float

            TODO: DATA HERE
    """

    if ndo.index.freq == pd.Timedelta("1DAY"):
        ndo = ndo.resample('15T').interpolate(method='linear')
    elif ndo.index.freq != pd.Timedelta("15MIN"):
        raise("ndo must be a one day or 15 minute series")
    if not stage.index.freq == pd.Timedelta("15MIN"):
        raise("stage must be an hourly or 15 minute series")
    if ndo.index.freq != stage.index.freq:
        raise("stage and ndo must have the same window")

    if ndo.isna().any().any():
        raise(f"missing data not allowed in ndo. First missing data at index: {np.where(ndo.isna())}")
    if stage.isna().any().any():
        raise(f"missing data not allowed in stage. First missing data at index: {np.where(stage.isna())}")

    newstart = ndo.index[0] - pd.Timedelta("21HOURS")
    newend = ndo.index[-1] - pd.Timedelta("3HOURS")
    if stage.index[0] > newstart:
        print(f"Stage record starts {stage.index[0]} and NDO starts {ndo.index[0]}")
        raise("stage record must begin at least 21 hours before ndo")
    if newend > stage.index[-1]:
        raise("stage record must end no more than 3 hours before end of ndo")
    
    z = stage.loc[newstart:newend]
    
    g_df = g.copy()
    g_df.columns = ['g']
    g_df['ec'] = np.nan

    z.columns = ['z']

    solu_df = pd.merge(g_df, z, how='left', left_index=True, right_index=True)
    # due to linear tidal filter need to start after filter period
    start_row = -int((k0 - len(c)-1) * pd.Timedelta(Dt) / solu_df.index.freq) 

    dstep = pd.Timedelta(Dt)/pd.Timedelta(solu_df.index.freq) # number of rows for each Dt

    # Calculate EC
    for i in range(start_row, len(solu_df)-1): 
        gval = solu_df['g'].iloc[i]
        # sum the tidal filter terms
        lin_filt = sum(ck * solu_df['z'].loc[solu_df.index[i] + int((k0 - k) * dstep)] 
                                             for k, ck in enumerate(c))
        # use numpy convolution
        # ecfrac = b0 + b1 * gval**npow1 + gval * lin_filt # npow1 and b1 are our parameters to tweak
        ecfrac = b1 * gval**npow1 + gval * lin_filt # npow1 and b1 are our parameters to tweak

        # ln((s - sb)/(so-sb)) = ecfrac <- use this to parameterize

        solu_df['ec'].iloc[i] = max(min_ec, (np.exp(ecfrac)*(so-sb) + sb))

    ec = solu_df[['ec']]

    return ec

def read_24h_index(date_strs):
    adjusted_dates = []
    for date_str in date_strs:
        if '24:00' in date_str:
            # Replace '24:00' with '00:00' and increment the date by one day
            date_str = date_str.replace('24:00', '00:00')
            # Adjust the date format (increment day)
            date = pd.to_datetime(date_str, format='%d %b %y, %H:%M')
            # Increment the day by 1
            date += pd.Timedelta(days=1)
            adjusted_dates.append(date)
        else:
            # For regular times, just convert to datetime
            adjusted_dates.append(pd.to_datetime(date_str, format='%d %b %y, %H:%M'))

    return adjusted_dates