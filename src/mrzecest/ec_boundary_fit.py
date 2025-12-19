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
from mrzecest.fitting_util import parse_config, validate_fit_config

import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families.links import Log
from statsmodels.genmod.families import Gamma
import scipy


calls = 0


def outer_fit(x, data, return_coefs=False, plot_fits=False):
    global calls
    calls = calls + 1
    print(f"Entering outer fit with x = {x}")
    fit_spec = data.attrs.get("fit_spec", None)
    if fit_spec is None:
        raise ValueError("fit_spec missing from data.attrs; cannot interpret outer parameters")

    outer_params = fit_spec.get("outer_params")
    if not outer_params:
        raise ValueError("fit_spec.outer_params is empty")

    # Map optimizer vector -> canonical model parameter values.
    pvals = {}
    for i, p in enumerate(outer_params):
        key = p.get("key")
        if not key:
            raise ValueError(f"outer_params[{i}] missing 'key'")
        scale = p.get("scale", 1.0)
        pvals[key] = float(x[i]) * float(scale)

    try:
        log10beta = pvals["beta_log10"]
        npow = pvals["npow"]
        area_coef = pvals["area_coef"]
        energy_coef = pvals["energy_coef"]
    except KeyError as e:
        raise ValueError(f"fit_run.outer_params must include beta_log10, npow, area_coef, energy_coef; missing {e}")

    ndo = data.ndo
    ec_obs = data.ec_obs
    g0 = 1000.
    # external parameters are used for g()
    ndomod = ndo_mod(data.ndo, data.d_elev_filt, area_coef, data.energy, energy_coef).squeeze()
    g = gcalc(ndomod, log10beta=log10beta, g0=g0)    
    has_nan = g.squeeze().isna().any() 
    has_negative = (g.squeeze() < 0).any()
    if has_nan or has_negative:
        if ndomod.iat[0] < 0.:
            print("ndomod has negative values at start:", ndomod.iloc[0])
        raise ValueError(f"g has NaN: {has_nan} or negative values: {has_negative}")

    if False:
        ndomod_series = ndomod.squeeze()
        g_series = g.squeeze()
        fig, ax = plt.subplots()
        ax.plot(data.ndo.index, data.ndo.values, label="ndo")
        ax.plot(ndomod_series.index, ndomod_series.values, label="ndomod")
        ax.plot(g_series.index, g_series.values, label="g")
        ax.legend()
        ax.set_ylabel("Value")
        ax.set_title("ndomod vs g")
        fig.autofmt_xdate()
        ax.grid(True)
        plt.show()
    gnpow = g.pow(npow)

    # now gather the linear components. As is usual for R-style models
    # beta0 (intercept is assumed in model so nothing gathered
    # beta1 requres g*npow which is called gnpow
    # ak k = 1 ... n requires g times lagged values of stage. These
    #                appear in data as columns called z0,z1,z2,z3.
    #
    use_ols = True
    sb = float(fit_spec["sb"])
    so = float(fit_spec["so"])
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
    preds = pd.concat((0.001 * gnpow, 0.001 * data.loc[:,zcols].mul(gnpow, axis=0)), axis=1)
    preds.columns = ["gnpow"] + zcols
    print("NaN check found null: ", preds.isnull().any(axis=None))

    X = sm.add_constant(preds)
    y = y.loc[~ysmall]
    X = X.loc[~ysmall, :]

    mask = ~X.isnull().any(axis=1) & y.notna()
    X_clean = X[mask]
    y_clean = y[mask]
    print("Condition Number of design matrix:", np.linalg.cond(X_clean))
    if (calls > 600000) and plot_fits:
        fig, ax = plt.subplots(1)
        ax.plot(preds.index, preds.values)
        ax.legend(preds.columns)
        plt.show()
    if use_ols:
        print("Using OLS")
        mod = sm.OLS(y_clean, X_clean)
    else:
        print("Creating GEE")
        cov_struct = sm.cov_struct.Autoregressive()
        family = sm.families.Gamma(link=Log())
        group = np.ones_like(y_clean)
        mod = GEE(
            y_clean, X_clean, groups=group, family=family, cov_struct=cov_struct
        )  # sm.cov_struct.Independence())
    print("Fitting model")
    try:
        result = mod.fit()
        print(result.summary())  # this has the coefficients that I require
    except Exception as e:
        print("Error during fitting:", e)
        raise

    predictions = result.fittedvalues
    rss = np.sum((y_clean - predictions) ** 2)

    ypred = y_clean.copy().to_frame()
    ypred.columns = ["data"]
    ypred["fit"] = np.nan
    ypred.loc[:, "fit"] = predictions

    if ((calls > 1000) | return_coefs) and plot_fits:
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


def fit_mrz_ecest(config, elev=None, ndo=None, ec_obs=None, plot_fits=False):

    print("Entering fit routine")
    if isinstance(config, str):
        config = parse_config(config)

    # No legacy/backward constraints: validate and fail fast.
    validate_fit_config(config)

    fit_run = config.get("fit_run")
    if not isinstance(fit_run, dict):
        raise ValueError("fit_run is required and must be a mapping")

    fit_start_key = fit_run.get("start")
    fit_end_key = fit_run.get("end")
    if fit_start_key is None or fit_end_key is None:
        raise ValueError("fit_run must include 'start' and 'end'")
    start = pd.to_datetime(fit_start_key)
    end = pd.to_datetime(fit_end_key)
    solver = str(fit_run.get("solver", "powell")).lower()

    outer_params = fit_run.get("outer_params")
    if not isinstance(outer_params, (list, tuple)) or len(outer_params) == 0:
        raise ValueError("fit_run.outer_params is required and must be a non-empty list")

    # Build x0/bounds from outer_params. Keys must match canonical model param names.
    x0 = []
    bounds = []
    any_finite_bounds = False
    for i, p in enumerate(outer_params):
        if not isinstance(p, dict):
            raise TypeError(f"outer_params[{i}] must be a mapping, got {type(p)}")
        key = p.get("key")
        if not key:
            raise ValueError(f"outer_params[{i}] missing 'key'")
        if "x0" not in p:
            raise ValueError(f"outer_params[{i}] ({key}) missing 'x0'")
        if "bounds" not in p:
            raise ValueError(f"outer_params[{i}] ({key}) missing 'bounds'")
        x0_i = float(p["x0"])
        b = p["bounds"]
        if b is None or (isinstance(b, (list, tuple)) and len(b) == 2 and b[0] is None and b[1] is None):
            bnd = (None, None)
        else:
            if not (isinstance(b, (list, tuple)) and len(b) == 2):
                raise ValueError(f"outer_params[{i}] ({key}) bounds must be [lo, hi] or [null, null]")
            lo = None if b[0] is None else float(b[0])
            hi = None if b[1] is None else float(b[1])
            bnd = (lo, hi)
            if lo is not None or hi is not None:
                any_finite_bounds = True
        # Fail fast if x0 outside finite bounds
        lo, hi = bnd
        if lo is not None and x0_i < lo:
            raise ValueError(f"outer_params[{i}] ({key}) x0 {x0_i} < lower bound {lo}")
        if hi is not None and x0_i > hi:
            raise ValueError(f"outer_params[{i}] ({key}) x0 {x0_i} > upper bound {hi}")
        x0.append(x0_i)
        bounds.append(bnd)

    # Bounds must not be silently ignored.
    if any_finite_bounds and solver in {"nelder-mead", "nelder_mead", "neldermead"}:
        raise ValueError("Bounds were provided but solver is Nelder-Mead (bounds would be ignored). Use a bounded solver.")

    elev.index.freq = elev.index.inferred_freq
    ndo.index.freq = ndo.index.inferred_freq
    ec_obs.index.freq = ec_obs.index.inferred_freq

    elev_filt = cosine_lanczos(elev, "40h")
    offset = elev.index.freq
    two_dtsec = 2.0 * pd.Timedelta(offset, unit=offset.freqstr.lower()).total_seconds()
    print(f"two_dtsec = {two_dtsec}")
    d_elev_filt = (elev_filt.shift(-1) - elev_filt.shift(1)) / two_dtsec
    elev_tidal = elev.copy() - elev_filt  # isolate tidal part
    energy = cosine_lanczos(elev_tidal * elev_tidal, "40h")

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
    solu_df = solu_df.loc[start:end, :]

    # Attach fit spec (constants + parameter interpretation) to the data frame.
    solu_df.attrs["fit_spec"] = {
        "so": float(config["so"]),
        "sb": float(config["sb"]),
        "outer_params": outer_params,
    }
    # solu_df.to_csv("test.csv", index=True, header=True, float_format="%.3f")

    tol = float(fit_run.get("tol", 5e-3))
    res = scipy.optimize.minimize(
        outer_fit,
        x0,
        args=(solu_df,),
        tol=tol,
        method=("Nelder-Mead" if solver in ("nelder-mead", "nelder_mead", "neldermead") else solver),
        bounds=bounds,
    )
    print(res.success)
    print(res.message)
    x_res, coefs, ypred = outer_fit(
        res.x, solu_df, return_coefs=True, plot_fits=plot_fits
    )
    # Report parameters in canonical units using the same mapping as the objective.
    pvals = {}
    for i, p in enumerate(outer_params):
        pvals[p["key"]] = float(x_res[i]) * float(p.get("scale", 1.0))
    print(
        " ".join(
            [
                f"{k}={round(v,3)}"
                for k, v in (
                    ("beta_log10", pvals.get("beta_log10")),
                    ("npow", pvals.get("npow")),
                    ("area_coef", pvals.get("area_coef")),
                    ("energy_coef", pvals.get("energy_coef")),
                )
            ]
        )
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
        start=start,
        end=end,
    )

    rss = np.sum((predictions - ec_obs) ** 2.0)
    return rss
