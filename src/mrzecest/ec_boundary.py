""" Calculate g-model based estimate of Martinez boundary.
"""

import yaml
import pandas as pd
import numpy as np
import datetime as dt
from vtools import *

#from gekko import GEKKO
import numba

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
        g0 = ndo.iat[0]
    
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

def ec_est(ndo,beta,g0,filt_coefs,filter_dt,elev):
    """ Estimate EC"""
    pass
