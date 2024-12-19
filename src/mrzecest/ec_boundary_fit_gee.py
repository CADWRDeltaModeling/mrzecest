""" Estimate the Martinez EC model using the 
    conditional optimization approach in the 2001 Annual Report Chapter 11.
    An outer, generic optimization searches over beta (time scale)
    parameter of gmodel and npow, the power "n" sometimes called
    a "shape factor". Conditional on those variables, the rest
    of the fit can be performed using Generalized Estimating Equations,
    which are appropriate given the log link and non-normal errors. 

    As a side note, this fit will by definition always be worse than 
    least squares if the basis of comparison is also squared squared 
    error. The point in the 2001 article is that this is a poor basis
    of comparison for this variable. 

"""
import pandas as pd
import numpy as np
from vtools import *
from mrzecest.ec_boundary import gcalc

import statsmodels.api as sm
import statsmodels.formula.api as smf

def subfit_mrzecest_gee(x,parms,data):
    """ Conditional fit"""
    beta = x[0]

def outer_fit(x,data):
    log10beta = x[0]
    npow = x[1]
    ndo = data.ndo
    ec_obs = data.ec_obs
    g0 = 5000.
    # external parameters are used for g()
    g = gcalc(data.ndo,log10beta=log10beta,g0=g0)
    gnpow = g**npow

    # now gather the linear components. As is usual for R-style models
    # beta0 (intercept is assumed in model so nothing gathered
    # beta1 requres g*npow which is called gnpow
    # ak k = 1 ... n requires g times lagged values of stage. These
    #                appear in data as columns called z0,z1,z2,z3.
    # 
    sb = 200. # Todo: move to inputs
    y = data.ec_obs - sb
    zcols = [col for col in data.columns if col.lower.startswith("z")]
    preds = pd.concat(gpow,data[zcols].mul(g,axis=0))

    cov_struct = sm.cov_struct.Autoregressive()
    family = sm.families.Gamma()
    mod = smf.gee("y ~ age + trt + base", "subject", data, cov_struct=cov_struct, family=family) 
res = mod.fit()
    # now formulate the 
    error = gnpow.squeeze() - ec_obs.squeeze()
    return (error*error).sum(axis=None)


def fit_mrzecest_gee(config, elev=None, ndo=None, ec_obs=None):
    
    if isinstance(config,str): 
        config = parse_config(config)

    # Now do bounds/nan check
    start = pd.to_datetime(config['fit_start'])
    end = pd.to_datetime(config['fit_end'])

    elev_filt = cosine_lanczos(elev,'40h')
    elev_tidal = elev.copy() - elev_filt  # isolate tidal part

    # The outer optimization is only in terms of log10beta and npow
    # The other variables are fit globally using GEE in a way that
    # does not require a starting guess.    
    try:
        log10beta = float(config['param']['log10gbeta'][0]
    except:
        log10beta = float(config['param']['log10gbeta']
    
    try:
        npow = float(config['param']['npow'][0]
    except:
        npow = float(config['param']['npow'][0]

    # Prepare fixed items and data
    filter_dt = pd.Timedelta(config['filter_setup']['dt'])
    filter_len = int(config['filter_setup']['filter_length'])
    filter_k0 = int(config['filter_setup']['k0'])
    ndofreq = pd.Timedelta(ndo.index.freq)
    dstep = int(filter_dt/ndofreq) # number of rows for each Dt
    solu_df = pd.concat((ec_obs,ndo,elev_filt,elev_tidal),axis=1) 
    solu_df.columns = ["ec_obs","ndo","elev_filt","elev_tidal"]   
    for k in range(0,filter_len):
        solu_df[f'z{k}'] = solu_df['elev_tidal'].shift( (filter_k0 - k)*dstep)

    # Do bounds/nan check after all the filteration is done.
    start = pd.to_datetime(config['fit_start'])
    end = pd.to_datetime(config['fit_end'])
    solu_df = solu_df.loc[start:end,:]    
    solu_df.to_csv("test.csv",index=True,header=True,float_format="%.3f")

    x0 = [10.1,0.75,40000]   # log10beta, npow, area correction

    res = scipy.minimize(outer_fit_gee,x0,args=solu_df)
    # set up filter


