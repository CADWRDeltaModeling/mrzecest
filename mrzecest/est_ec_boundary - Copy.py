# script by Lily Tomkovic to calculate Net Delta Outflow using DSM2 boundary data
# takes config.inp file to get relevant inputs
# returns dataframe that can be written to DSS

import pandas as pd
import numpy as np

import datetime as dt
import os
import sys

from pydelmod.create_ann_inputs import get_dss_data
import pyhecdss

import config_python as config
from ec_functions import gcalc, ecest, calc_filtered, plot_dicts, read_24h_index

from gekko import GEKKO

def find_ecest_params(stage, ndo, ec_obs,
                      gbeta = 1.5e10, g0 = 5000,
                      so = 32000, sb = 200, 
                      npow1 = 0.770179, 
                      b0 = 1.37e-2, b1 = -6.43e-5, Dt = "3HOUR", k0 = -1,
                      min_ec = 200,
                      a = [1.59e-4, -1.28e-5, 6.96e-6, 4.83e-5, -7.67e-5, 6.93e-5, -3.85e-5]):
    
    # newstart = ndo.index[0] - pd.Timedelta("21HOURS")
    # newend = ndo.index[-1] - pd.Timedelta("3HOURS")
    # z = stage.loc[newstart:newend]
    
    ndo.columns = ['ndo']

    stage.columns = ['z0']
    ec_obs = ec_obs.resample('15T').interpolate(method='linear')
    ec_obs.index = ec_obs.index.to_period(freq='15T')
    ec_obs.columns = ['ec_obs']

    solu_df = pd.merge(ndo, stage, how='left', left_index=True, right_index=True)
    solu_df = pd.merge(solu_df, ec_obs, how='left', left_index=True, right_index=True)
    dstep = int(pd.Timedelta(Dt)/pd.Timedelta(ndo.index.freq)) # number of rows for each Dt

    for k in range(len(a)):
        solu_df[f'z{k+1}'] = solu_df['z0'].shift(k*dstep)

    solu_df = solu_df.dropna()
    solu_df = solu_df.iloc[:500,:]
    
    z = solu_df.filter(like='z').to_numpy()

    # Initialize GEKKO model
    temp_dir = r'./gekko_model'
    os.makedirs(temp_dir, exist_ok=True)
    m = GEKKO(remote=True, temp_dir=temp_dir)
    m.time = solu_df.index.astype('int64') / 10**9  # Convert to seconds (since PeriodIndex uses nanoseconds)
    
    # Set the model mode to dynamic if using time
    m.options.IMODE = 4 # Dynamic mode, as an example
    # m.options.MEAS_CHK = 0
    m.options.solver = 3  # set solver type 

    # input parameters
    s_obs_var = solu_df['ec_obs'].to_numpy().astype(float)
    q_param = m.Param(value=solu_df[['ndo']].to_numpy().astype(float))


    # Define a specific path for saving temporary files

    # Define variables
    g = m.Var(value=g0, lb=0)  # Single GEKKO variable over time
    beta = m.FV(value=gbeta, lb=0.1)  # Coefficient, lower bound .1
    beta0 = m.FV(value=b0)
    beta1 = m.FV(value=b1)
    n = m.FV(value=npow1, lb=0.1) # lower bound .1
    a_param = [m.FV(value=ak) for ak in a]  # Coefficients for lags

    # Enable parameters for optimization
    for param in [beta, beta0, beta1, n] + a_param:
        param.STATUS = 1

    # Define differential equation

    m.Equation(g.dt() == (g * (q_param - g) / beta))

    # Define output s
    so = so  # Example constant
    sb = sb  # Example constant

    # Summation term for z_k
    z_sum = m.Intermediate(m.sum([ak * z[:, k] for k, ak in enumerate(a_param)]))

    # Define the model for s
    s = m.Intermediate(m.exp(beta0 + beta1 * g**n + g * z_sum) * (so - sb) + sb)
    print("s_obs_var:", s_obs_var.shape, s_obs_var)
    print("z array:", z.shape, z)
    print("z_sum:", z_sum)
    print("Intermediate s:", m.exp(beta0 + beta1 * g**n + g * z_sum) * (so - sb) + sb)

    # Define error for each observation
    error = [m.log((s - so) / (so - sb)) - s_obs_var[i] for i in range(len(s_obs_var))]
    print([e**2 for e in error])  # Check the squared errors list
    # Define the objective as the sum of squared errors
    m.Obj(m.sum([e**2 for e in error]))

    # Solve the problem
    os.environ['GEKKO_TMPDIR'] = temp_dir
    m.solve(disp=True, debug=2)

    # Extract results
    g_sol = g.value
    beta_sol = beta.value[0]
    beta0_sol = beta0.value[0]
    beta1_sol = beta1.value[0]
    n_sol = n.value[0]
    a_sol = [ai.value[0] for ai in a_param]

    print("Solution:")
    print(f"g: {g_sol}")
    print(f"beta: {beta_sol}, beta0: {beta0_sol}, beta1: {beta1_sol}, n: {n_sol}")
    print(f"a: {a_sol}")


def estimate_ec_from_ndo_stage(df_ndo, df_mtz_stage, twind,
                               write_dss=None, ec_est_filename=None,
                               storage_area = 34923.48,
                               gbeta = 1.5e10, g0 = 5000, 
                               so = 32000, sb = 200, 
                               npow1 = 0.770179, 
                               b0= 1.37e-2, b1 = -6.43e-5, Dt = "3HOUR", k0 = -1,
                               min_ec=200,
                               c = [1.59e-4, -1.28e-5, 6.96e-6, 4.83e-5, -7.67e-5, 6.93e-5, -3.85e-5],
                               hist_ec_df = None):
    """take Net Delta Outflow and Stage at Martinez and determine the estimated EC at Martinez
    
    Parameters
    ----------
    g0: float 
    
        initial value of g (antecedent outflow) for the beginning of the first year. This is pretty arbitrary and makes little difference

    df_ndo : pd.DataFrame 
        Net Delta Outflow with timeseries index and "ndo" column

    Returns
    -------
    ec : pd.DataFrame
        Data with estimate of EC at Martinez for DSM2 simulation
    
    """

    twindbuf = [twind[0] - pd.DateOffset(months=1),
                twind[1] + pd.DateOffset(months=1)]     # Conservative buffered period for retrieval
                                                        # so that after prelimiary operations (e.g. time average)
                                                        # time series will still span at least twind
    df_ndo = df_ndo.loc[twindbuf[0]:twindbuf[1]]
    df_mtz_stage = df_mtz_stage.loc[twindbuf[0]:twindbuf[1]]

    df_ndo15 = df_ndo.resample('15T').interpolate(method='linear')

    astro_stage_version = config.getAttr("ASTRO_STAGE_VERSION") # retrieve the vertical datum info for tidal boundary
    if 'NAVD' in astro_stage_version:
        df_mtz_stage = df_mtz_stage - 2.68

    filter_tide = calc_filtered(df_mtz_stage)   # filtered subtide
    # formerly: godin((mtzastro*mtzastro)**0.5)           # RMS energy of tide (used to predict filling and draining)
    time_delta = 1/int(pd.Timedelta('1day')/filter_tide.index.freq)
    filt_deriv = pd.DataFrame(index=filter_tide.index, 
                              data={'derivative':np.gradient(filter_tide.iloc[:,0], time_delta),
                                    'ndo':df_ndo15['ndo'].values}) # first difference/derivative [ft/day]
    # formerly: dastrorms = (  (astrorms-(astrorms>>1))*96. ).createSlice(twind) # first difference/derivative? using days/15min .shift(-/+1)
    filt_deriv['proxy_for_ec'] = filt_deriv['ndo'] - storage_area*(filt_deriv['derivative']) # Fitted quantity (change to change in subtidal flow) what is this for? Unit conversion? was 53411.1 then 34923.48

    fifteenflo2 = filt_deriv[['proxy_for_ec']].loc[twind[0]:twind[1]]
    
    short_filter_tide = (pd.merge(df_mtz_stage, filter_tide, how='left', left_index=True, right_index=True)
                     .assign(short_stage=lambda x: x.iloc[:, 0] - x.iloc[:, 1])
                     [['short_stage']])
    # short_filter_tide = calc_filtered(df_mtz_stage, cutoff_period="3HOUR")
    
    find_ecest_params(df_mtz_stage, fifteenflo2, hist_ec_df)
    # calculate g-model
    g = gcalc(fifteenflo2, gbeta, g0)

    # call to ec estimator. all parameters are included. 
    mtzecest = ecest(df_mtz_stage, fifteenflo2, g, 
                     so = so, sb = sb, npow1 = npow1, 
                     b0 = b0, b1 = b1, Dt = Dt, k0 = k0, min_ec = min_ec, c = c)

    # for debugging ---------------------------------------------------------------------
    plt_dicts = {'Tide (ft)': {'raw stage': df_mtz_stage,
                               'filtered': filter_tide,
                               'just tidal': short_filter_tide,
                               'derivative of filtered': filt_deriv[['derivative']]},
                 'NDO (cfs)': {'NDO in': df_ndo15,
                               'NDO with Tidal Effect': filt_deriv[['proxy_for_ec']],
                               'g': g},
                 'EC (uS/cm)': {'Historic Input': hist_ec_df,
                                'Estimated EC': mtzecest}
                                }

    plot_dicts(plt_dicts, 'NDO (cfs)')#------------------------------------------------
    
    print('Done')

    return mtzecest


def main(infile, write_dss, hist_ec_filename):

    config.setConfigVars(infile) # load the configuration parameters
    startdate=config.getAttr('START_DATE')
    enddate=config.getAttr('END_DATE')

    twind = pd.to_datetime([f'{startdate} 0000', f'{enddate} 0000'], format="%d%b%Y %H%M")  # Actual period to be estimated

    print(f"Calculating boundary salinity for the period {twind[0]} to {twind[0]}")

    os.chdir(os.path.dirname(infile))
    
    ec_est_filename = os.path.normpath(os.path.join(os.getcwd(),config.getAttr('MTZECFILE'))).replace("\\", "/")
    stage_bdy_filename = os.path.normpath(os.path.join(os.getcwd(),config.getAttr('STAGE_SOURCE_FILE'))).replace("\\", "/")
    flow_filename = os.path.normpath(os.path.join(os.getcwd(),config.getAttr('BNDRYINPUT'))).replace("\\", "/")
    dcd_filename = os.path.normpath(os.path.join(os.getcwd(),config.getAttr('DICUFILE'))).replace("\\", "/")

    # df_ndo = calc_ndo(flow_filename, dcd_filename, write_outfile="./ndo.csv")
    df_ndo = pd.read_table("../data/ndo.csv",
                           sep=',',
                           index_col=0)
    df_ndo.index = pd.to_datetime(df_ndo.index) # debugging - skip the calc step

    if not os.path.exists(ec_est_filename) and write_dss:
        dumdss = pyhecdss.DSSFile(ec_est_filename, create_new=True) # create the file if writing out to DSS

    b_part_dss_filename_dict={'RSAC054': stage_bdy_filename}
    b_part_c_part_dict={'RSAC054': 'STAGE'}
    df_mtz_stage = get_dss_data(b_part_dss_filename_dict, 'b_part', primary_part_c_part_dict=b_part_c_part_dict, daily_avg=False)

    hist_ec_df = pd.read_csv(hist_ec_filename,
                             sep=',',
                             index_col=0)
    hist_ec_df.index = read_24h_index(hist_ec_df.index) # debugging - skip the calc step

    estimate_ec_from_ndo_stage(df_ndo, df_mtz_stage, twind,
                               write_dss=write_dss, ec_est_filename=ec_est_filename, hist_ec_df=hist_ec_df)

if __name__ == '__main__':
    # infile = sys.argv[1] # infile should be the model's config.inp file

    infile = r'./dsm2_inputs/config_latinhypercube_85.inp'
    hist_ec_filename = r"../data/mtz_ec_hist_bc.csv" # location relative to inp file

    main(infile, False, hist_ec_filename)