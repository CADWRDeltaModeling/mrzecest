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
          g0=5000.):
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
    ndo = ndo.dropna()

    g =  ndo.copy()
    g.columns = ['g']
    g.iloc[:] = np.nan
    g = g.squeeze()

    # Set initial condition
    if g0 is None: 
        g0 = ndo['ndo'].iloc[0]
    
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

def ndo_mod(ndo,d_elev_filt,area_coef,energy,energy_coef):
    ndo_mod = ndo.squeeze() + area_coef*d_elev_filt.squeeze() + energy_coef*energy.squeeze()
    ndo_mod.name = 'ndo'

    return ndo_mod.to_frame()


def z_sum_term(z,filter_k0,filt_coefs,filter_dt):

    filter_len = len(filt_coefs)
    df_freq = pd.Timedelta(z.index.freq)
    d_step = int(filter_dt/df_freq) # number of rows for each Dt
    
    z.columns = ['elev_tidal']

    for k in range(0,filter_len):
        z[f'z{k}'] = z['elev_tidal'].shift( -int((filter_k0 - k)*d_step) )

    z = z.dropna()
    z_sum = pd.Series(index=z.index,
                      data=np.nansum(z.iloc[:, 1:(len(filt_coefs)+1)].values * filt_coefs, axis=1))

    return z_sum

# default_ec_params = {'area_coef' : 0.,
#                      'energy_coef': 0.,
#                      'log10beta':10.1}

def ec_est(ndo, elev, start, end,
           area_coef,energy_coef,
           log10beta,
           beta0, beta1, npow, filter_k0,
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
    
    # Determine which index has the finer frequency
    if elev.index.freq is not None and ndo.index.freq is not None:
        if elev.index.freq < ndo.index.freq:
            ndo = ndo.resample(elev.index.freq).interpolate(method='linear')
        else:
            elev = elev.resample(ndo.index.freq).interpolate(method='linear')

    overlapping_index = ndo.index.intersection(elev.index)
    ndo = ndo.loc[overlapping_index]
    elev = elev.loc[overlapping_index]

    assert(ndo.index.equals(elev.index))

    # Apply a cosine Lanczos filter (low-pass) to the elevation dataframe
    elev_filt = cosine_lanczos(elev, cutoff_period='40H', padtype='odd')
    elev_tidal = elev.copy() - elev_filt  # isolate tidal part for z_sum term
    energy = cosine_lanczos(elev_tidal*elev_tidal, cutoff_period='40H', padtype='odd')

    # calculate subtidal effects on ndo
    offset = elev_filt.index.freq
    
    two_dtsec = 2.*pd.Timedelta(offset, unit=offset.freqstr.lower()).total_seconds() # dt term to be used for estimating derivative of tide
    d_elev_filt = (elev_filt.shift(-1) - elev_filt.shift(1)) / two_dtsec
    d_elev_filt = d_elev_filt.dropna()
    
    ndomod = ndo_mod(ndo,d_elev_filt,area_coef,energy,energy_coef) 
    ndomod = ndomod.dropna()
    ndomod = ndomod.loc[ndomod[ndomod['ndo'] >= 0].index[0]:]
    
    # calculate g-model results
    g = gcalc(ndomod, log10beta=log10beta)
    
    # calculate lagged z_df term for
    z_sum = z_sum_term(elev_tidal, filter_k0, filt_coefs, filter_dt)

    g = g.loc[(start-pd.Timedelta('30d')):(end+pd.Timedelta('30d'))]    
    z_sum = z_sum.loc[(start-pd.Timedelta('30d')):(end+pd.Timedelta('30d'))]    
    ec = z_sum.copy()
    ec[:] = np.nan

    # Calculate EC
    # using ec_kernel to accelerate, which requires
    # pandas conversion to/from numpy 
    print("solving for ec")
    ec.iloc[:] = ec_kernel(g.squeeze().to_numpy(),z_sum.squeeze().to_numpy(), beta0, beta1, npow, so, sb)
    print("done")

    ec = ec.loc[start:end]

    return ec

@numba.jit
def ec_kernel(g, z_sum, beta0, beta1, npow, so, sb):
    """numpy based kernel for ec(t) using numba."""
    
    ec = np.empty(len(g),dtype=float)
    ntime = len(g)

    for i in np.arange(1, ntime):
        # TODO: FIX TO BE CONSISTENT WIHT GEE.py
        ecfrac = beta0 + beta1 * g[i]**npow + g[i]**npow * z_sum[i] # npow1 and b1 are our parameters to tweak
        ec[i] = np.exp(ecfrac)*(so-sb) + sb # solving for s term
    
    return ec
