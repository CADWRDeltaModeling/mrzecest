# script by Lily Tomkovic to calculate Net Delta Outflow using DSM2 boundary data
# takes config.inp file to get relevant inputs
# returns dataframe that can be written to DSS

import yaml
import pandas as pd
import numpy as np
import datetime as dt
import os
import sys


from ec_functions import gcalc, ecest, calc_filtered, plot_dicts

from gekko import GEKKO

with open("example.yaml") as stream:
    try:
        print(yaml.safe_load(stream))
    except yaml.YAMLE



def gcalc(ndo,
          beta=1.5e10,
          g0=None):
    """ Calculates antecedent outflow from a stream of ndo integrating using the trapezoidal method

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

    dt = 1
    nstep=len(ndo)
    if ti == pd.Timedelta("15MIN"):
        # dt = 0.25 # hours 
        dt = 900. # [s]
    elif ti == pd.Timedelta('1h")
        dt = 3600. # [s]
    else:
        raise ValueError("NDO time step must be 15MIN or 1HOUR. Please interpolate")
        
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



def find_ecest_params(elev, ndo, ec_obs,
                      gbeta = 1.5e10, g0 = 5000,
                      so = 32000, sb = 200, 
                      npow1 = 0.770179, 
                      b0 = 1.37e-2, b1 = -6.43e-5, Dt = "3HOUR", k0 = -1,
                      min_ec = 200,
                      a = [1.59e-4, -1.28e-5, 6.96e-6, 4.83e-5, -7.67e-5, 6.93e-5, -3.85e-5]):
    
    # newstart = ndo.index[0] - pd.Timedelta("21HOURS")
    # newend = ndo.index[-1] - pd.Timedelta("3HOURS")
    # z = elev.loc[newstart:newend]
    
    ndo.columns = ['ndo']

    elev.columns = ['z0']
    ec_obs = ec_obs.resample('15T').interpolate(method='linear')
    ec_obs.index = ec_obs.index.to_period(freq='15T')
    ec_obs.columns = ['ec_obs']

    solu_df = pd.merge(ndo, elev, how='left', left_index=True, right_index=True)
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


def parse_config(yml):
    with open(yml) as stream:
        try:
            print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)


def est_mrzec_model(config, df_ndo=None, df_mtz_elev=None): 
    """Estimate model based on data and a config file of parameter guesses and bounds
       take Net Delta Outflow and elev at Martinez and determine the estimated EC at Martinez
    
    Parameters
    ----------
    config : name of yaml file with contents similar to example folder
    
        Includes specification of the filter and fixed ge params

    df_ndo : pd.DataFrame 
        Net Delta Outflow time series with DateTime index and ndo column

    df_elev: pd.DataFrame
        Elevation used for estimate. May be used for tidal and/or g correction if elev_correction=True in config

    Returns
    -------
    params : pd.DataFrame
        DataFrame with "name" and "value" columns   
    """
    
    # test if it is yaml structure or yaml file and parse accordingly    
    config = parse_config if isinstance(config, str) else config

    raise NotImplementedError("Focusing on fit first")

    return mtzecest
