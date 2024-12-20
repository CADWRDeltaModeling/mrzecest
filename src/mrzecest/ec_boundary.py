""" Calculate g-model based estimate of Martinez boundary.
"""

import yaml
import pandas as pd
import numpy as np
import datetime as dt
from vtools import *
import os

import numba

from mrzecest.fitting_util import parse_config

def gcalc(ndo,
          log10beta=10.1,
          g0=None):
    """ Calculates antecedent outflow from a stream of ndo integrating using the trapezoidal method

    Parameters
    ----------
        ndo: pd.DataFrame
            a regular time series. Must be 15MIN, 1HOUR. Thus, NDO has been interpolated first.

        log10beta: float
            log10 of g-model parameter, which pre-log is in units ((cfs/s)*s).   
            Values of 1.5e9 - R. Denton [cf] to 1.5e10 Ateljevich used previously, 
            So a log range of 8.5-11 probably covers the range that needs to be explored
            in parameter fitting. 
        
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
    elif ti == pd.Timedelta("1h"):
        dt = 3600. # [s]
    else:
        raise ValueError("NDO time step must be 15MIN or 1HOUR. Please interpolate")

    beta = 10.**log10beta

    g =  ndo.copy()
    g.columns = ['g']
    g.iloc[:] = np.nan
    g = g.squeeze()

    # Set initial condition
    if g0 is None: 
        g0 = ndo.iat[0, 0]
    
    # solve implicitly with trapezoidal method
    # using g_kernel to accelerate, which requires
    # pandas conversion to/from numpy 
    print("integrating")
    g.iloc[:] = g_kernel(ndo.squeeze().to_numpy(),beta,g0,dt)    
    print("done")
    return g

@numba.jit
def g_kernel(ndo,beta,g0,dt):
    """numpy based integration kernel for g(t) 
       using trapezoidal method and numba."""
    div2dt = 2. * beta / dt # units?
    g = np.empty(len(ndo),dtype=float)
    g[0] = g0
    ntime = len(g)
    qpast = ndo[0]
    gpast = qpast 
    for i in np.arange(1, ntime):
        q = ndo[i] 
        qterm = q - div2dt
        gnew = 0.5 * (qterm + np.sqrt(qterm**2 - 4 * (gpast**2 - gpast * (qpast + div2dt))))
        g[i] = gnew
        gpast = gnew
        qpast = q
    return g

def z_sum_term(z,filter_k0,filt_coefs,filter_dt):

    filter_len = len(filt_coefs)
    df_freq = pd.Timedelta(z.index.freq)
    d_step = int(filter_dt/df_freq) # number of rows for each Dt
    
    z.columns = ['elev_tidal']

    for k in range(0,filter_len):
        z[f'z{k}'] = z['elev_tidal'].shift( -int((filter_k0 - k)*d_step) )

    z = z.dropna()
    z_sum = pd.Series(index=z.index,
                      data=np.nansum(z.iloc[:, :len(filt_coefs)].values * filt_coefs, axis=1))

    return z_sum

def ec_est(ndo, elev,
           start, end,
           storage_area,
           log10beta,
           beta1, npow, filter_k0,
           filt_coefs, filter_dt,
           so, sb):
    """ Estimates EC given the net delta outflow and tidal elevation

    Parameters
    ----------
        ndo: pd.DataFrame
            A regular time series. Must be 15MIN, 1HOUR.

        elev: pd.DataFrame
            A regular time series. Must be 15MIN, 1HOUR.
        
        filter_k0: float
            Filter parameter k0.
        
        filt_coefs: list or array
            Filter coefficients.
        
        filter_dt: pd.Timedelta
            Filter time step.

    Returns
    -------
        ec: pd.DataFrame
            A regular time series, same sampling rate as input with.
    """
    assert ndo.index.equals(elev.index)

    # Apply a cosine Lanczos filter (low-pass) to the elevation dataframe
    elev_filt = cosine_lanczos(elev, cutoff_period='40H', padtype='odd')
    elev_tidal = elev.copy() - elev_filt  # isolate tidal part for z_sum term

    # calculate subtidal effects on ndo
    time_delta = 1/int(pd.Timedelta('1day')/elev_filt.index.freq) # dt term to be used for estimating derivative of tide
    filt_deriv = pd.DataFrame(index=elev_filt.index, 
                              data={'derivative':np.gradient(elev_filt.values.flatten(), time_delta), # first difference/derivative [ft/day]
                                    'ndo':ndo['ndo'].values}) 
    
    filt_deriv['ndo_w_subtide'] = filt_deriv['ndo'] - storage_area*(filt_deriv['derivative']) # compute subtidal effect on NDO
    ndo_w_subtide = filt_deriv[['ndo_w_subtide']]
    
    # calculate g-model results
    g = gcalc(ndo_w_subtide, log10beta=log10beta)
    
    # calculate lagged z_df term for
    z_sum = z_sum_term(elev_tidal, filter_k0, filt_coefs, filter_dt)

    g = g.loc[start:end]    
    z_sum = z_sum.loc[start:end]    
    ec = z_sum.copy()
    ec[:] = np.nan

    # Calculate EC
    # using ec_kernel to accelerate, which requires
    # pandas conversion to/from numpy 
    print("solving for ec")
    ec.iloc[:] = ec_kernel(g.squeeze().to_numpy(),z_sum.squeeze().to_numpy(), beta1, npow, so, sb)
    print("done")

    return ec

@numba.jit
def ec_kernel(g, z_sum, beta1, npow, so, sb):
    """numpy based kernel for ec(t) using numba."""
    
    ec = np.empty(len(g),dtype=float)
    ntime = len(g)

    for i in np.arange(1, ntime):
        ecfrac = beta1 * g[i]**npow + g[i] * z_sum[i] # npow1 and b1 are our parameters to tweak

        ec[i] = np.exp(ecfrac)*(so-sb) + sb # solving for s term
    
    return ec

def ec_config(config):

    if isinstance(config,str): 
        config = parse_config(config)
    elif isinstance(config,dict):
        config = config
    else:
        raise ValueError(f"config input needs to be either a config.yaml file or a dictionary with the correct setup")
    
    ndo = pd.read_csv(config['ndo_file'], sep=',', index_col=0, parse_dates=['datetime'])
    elev = pd.read_csv(config['mrz_elev_file'], sep=',', index_col=0, parse_dates=['datetime'])
    ndo = ndo.resample('15T').interpolate(method='linear') # make sure ndo is on 15minute period
    elev = elev.resample('15T').interpolate(method='linear') # make sure ndo is on 15minute period

    # align the ndo and elev dataframes
    common_index = elev.index.intersection(ndo.index)
    ndo = ndo.loc[common_index]
    elev = elev.loc[common_index]

    start = pd.to_datetime(config['start'])
    end = pd.to_datetime(config['end'])

    storage_area = config['storage_area']

    try:
        log10beta = float(config['param']['log10gbeta'][0])
    except:
        log10beta = float(config['param']['log10gbeta'])
    
    try:
        beta1 = float(config['param']['b1'][0])
    except:
        beta1 = float(config['param']['b1'])

    try:
        npow = float(config['param']['npow'][0])
    except:
        npow = float(config['param']['npow'])

    filter_k0 = float(config['filter_setup']['k0'])
    filt_coefs = [float(ak) for ak in config['filter_setup']['afilt']]
    filter_dt = pd.Timedelta(config['filter_setup']['dt'])
    so = float(config['so'])
    sb = float(config['sb'])

    return ndo, elev, start, end, storage_area, log10beta, beta1, npow, filter_k0, filt_coefs, filter_dt, so, sb