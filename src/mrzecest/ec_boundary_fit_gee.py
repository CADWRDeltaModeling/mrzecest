"""Estimate the Martinez EC model using the
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
import matplotlib.pyplot as plt
from mrzecest.ec_boundary import gcalc, ndo_mod
from mrzecest.fitting_util import parse_config

import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families.links import Log
from statsmodels.genmod.families import Gamma
import scipy


calls = 0


def outer_fit(x, data, return_coefs=False):
    global calls
    calls = calls + 1
    print(f"Entering outer fit with x = {x}")
    log10beta = x[0]
    npow = x[1]
    area_coef = x[2] * 3600 * 1000000.0
    energy_coef = x[3] * 1000

    ndo = data.ndo
    ec_obs = data.ec_obs
    g0 = 5000.0
    # external parameters are used for g()
    ndomod = ndo_mod(data.ndo, data.d_elev_filt, area_coef, data.energy, energy_coef)
    g = gcalc(ndomod, log10beta=log10beta, g0=g0)

    gnpow = g.pow(npow)

    # now gather the linear components. As is usual for R-style models
    # beta0 (intercept is assumed in model so nothing gathered
    # beta1 requres g*npow which is called gnpow
    # ak k = 1 ... n requires g times lagged values of stage. These
    #                appear in data as columns called z0,z1,z2,z3.
    #
    use_ols = True
    sb = 200.0  # Todo: move to inputs
    so = 20000.0
    y = data.ec_obs
    y = y.clip(lower=1e-10)

    if use_ols:
        y = y.clip(lower=sb + 1e-10)
        y[:] = np.log((y - sb) / (so - sb))
        ysmall = y < -2.0
    else:
        y = y.clip(lower=sb + 1e-5)
        ysmall = y < 1000.0
        y[:] = (y - sb) / (so - sb)

    zcols = [col for col in data.columns if col.lower().startswith("z")]
    preds = pd.concat((0.001 * gnpow, 0.001 * data[zcols].mul(gnpow, axis=0)), axis=1)
    preds.columns = ["gnpow"] + zcols
    print("NaN check found null: ", preds.isnull().any(axis=None))

    X = sm.add_constant(preds)
    y = y.loc[~ysmall]
    X = X.loc[~ysmall, :]

    print("Condition Number of design matrix:", np.linalg.cond(X))
    if calls > 600000:
        fig, ax = plt.subplots(1)
        ax.plot(preds.index, preds.values)
        ax.legend(preds.columns)
        plt.show()
    if use_ols:
        print("Using OLS")
        mod = sm.OLS(y, X)
    else:
        print("Creating GEE")
        cov_struct = sm.cov_struct.Autoregressive()
        family = sm.families.Gamma(link=Log())
        group = np.ones_like(y)
        mod = GEE(
            y, X, groups=group, family=family, cov_struct=cov_struct
        )  # sm.cov_struct.Independence())
    print("Fitting model")
    try:
        result = mod.fit()
        print(result.summary())  # this has the coefficients that I require
    except Exception as e:
        print("Error during fitting:", e)
        raise

    predictions = result.fittedvalues
    rss = np.sum((y - predictions) ** 2)

    ypred = y.copy().to_frame()
    ypred.columns = ["data"]
    print(ypred)
    ypred["fit"] = np.nan
    ypred.loc[:, "fit"] = predictions
    print(ypred)

    if (calls > 1000) | return_coefs:
        fig, ax = plt.subplots(1)
        ax.plot(ypred.index, ypred.values)
        ax.legend(["data", "fit"])
        plt.show()

    if use_ols:
        out = rss
        print(f"RSS {rss}")
    else:
        res.scale = 1.0
        out = res.qic()
        print(f"RSS {rss} QIC {qic}")
    if return_coefs:
        return x, result.params, ypred  # qic[0]
    else:
        return out  # qic[0]


def fit_mrzecest_gee(config, elev=None, ndo=None, ec_obs=None):

    print("Entering fit routine")
    if isinstance(config, str):
        config = parse_config(config)

    # Now do bounds/nan check
    start = pd.to_datetime(config["fit_start"])
    end = pd.to_datetime(config["fit_end"])

    elev_filt = cosine_lanczos(elev, "40h")
    offset = elev.index.freq
    two_dtsec = 2.0 * pd.Timedelta(offset, unit=offset.freqstr.lower()).total_seconds()
    print(two_dtsec)
    d_elev_filt = (elev_filt.shift(-1) - elev_filt.shift(1)) / two_dtsec
    elev_tidal = elev.copy() - elev_filt  # isolate tidal part
    energy = cosine_lanczos(elev_tidal * elev_tidal, "40h")

    # The outer optimization is only in terms of log10beta and npow
    # The other variables are fit globally using GEE in a way that
    # does not require a starting guess.
    try:
        log10beta = float(config["param"]["log10gbeta"][0])
    except:
        log10beta = float(config["param"]["log10gbeta"])

    try:
        npow = float(config["param"]["npow"][0])
    except:
        npow = float(config["param"]["npow"][0])

    # Prepare fixed items and data
    filter_dt = pd.Timedelta(config["filter_setup"]["dt"])
    filter_len = int(config["filter_setup"]["filter_length"])
    filter_k0 = int(config["filter_setup"]["k0"])
    ndofreq = pd.Timedelta(ndo.index.freq)
    dstep = int(filter_dt / ndofreq)  # number of rows for each Dt
    solu_df = pd.concat(
        (ec_obs, ndo, elev_filt, elev_tidal, d_elev_filt, energy), axis=1
    )
    solu_df.columns = [
        "ec_obs",
        "ndo",
        "elev_filt",
        "elev_tidal",
        "d_elev_filt",
        "energy",
    ]
    for k in range(0, filter_len):
        solu_df[f"z{k}"] = solu_df["elev_tidal"].shift(-(filter_k0 - k) * dstep)

    # Do bounds/nan check after all the filteration is done.
    start = pd.to_datetime(config["fit_start"])
    end = pd.to_datetime(config["fit_end"])
    solu_df = solu_df.loc[start:end, :]
    solu_df.to_csv("test.csv", index=True, header=True, float_format="%.3f")

    x0 = [10.1, 0.5, -1.0, 0.88]  # log10beta, npow, area correction

    res = scipy.optimize.minimize(outer_fit, x0, args=solu_df, tol=5e-3)
    print(res.success)
    print(res.message)
    x_res, coefs, ypred = outer_fit(res.x, solu_df, return_coefs=True)
    print(
        f"log10beta = {round(x_res[0],3)} npow = {round(x_res[1],3)} area_coef = {round(x_res[2]*3600*1000000.,3)} energy_coef = {round(x_res[3]*1000,3)}"
    )
    print(f"beta0 = {coefs['const']}")
    print(f"beta1 = 0.001*{coefs['gnpow']}")
    print(
        f"z coefs = 0.001*{[round(zval, 3) for zval in coefs[coefs.index.str.startswith('z')].values]}"
    )

    return x_res, coefs, ypred


def obj_nlls(x):

    log10beta, npow, area_coef, energy_coef = x

    x0_pass2 = (area_coef, energy_coef, log10beta, beta0, beta1, npow, filt_coefs)
    args = (ec_obs, ndo, elev, start, end, filter_k0, filter_dt, so, sb)
    res2 = scipy.optimize.mimize(full_ls_obj, x, args=args)


def full_ls_obj(x, ec_obs, ndo, elev, start, end, filter_k0, filter_dt, so, sb):
    (area_coef, energy_coef, log10beta, beta0, beta1, npow, filt_coefs) = x

    print(f"Entering outer fit with x = {x}")
    log10beta = x[0]
    npow = x[1]
    area_coef = x[2] * 1000000.0
    energy_coef = x[3] * 1000

    print("second pass objective")

    ec_fit = ec_est(
        ndo,
        elev,
        start,
        end,
        area_coef,
        energy_coef,
        log10beta,
        beta1,
        npow,
        filter_k0,
        filt_coefs,
        filter_dt,
        so,
        sb,
    )

    rss = np.sum((predictions - ec_obs) ** 2.0)
    return rss
