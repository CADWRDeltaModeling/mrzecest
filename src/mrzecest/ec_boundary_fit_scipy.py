"""Estimate the Martinez EC model using scipy minimize and least squares."""

import yaml
import pandas as pd
import numpy as np
from vtools import *
from mrzecest.ec_boundary import gcalc


def subfit_mrzecest(x, data):
    log10beta = x[0]
    npow = x[1]
    ndo = data.ndo
    ec_obs = data.ec_obs
    g0 = 5000.0
    # external parameters are used for g()
    g = gcalc(data.ndo, log10beta=log10beta, g0=g0)
    gnpow = g**npow

    # now formulate the
    error = gnpow.squeeze() - ec_obs.squeeze()
    return (error * error).sum(axis=None)


def fit_mrzecest_scipy(config, elev=None, ndo=None, ec_obs=None):
    """This is based on our gekko code. I have not verified it"""
    if isinstance(config, str):
        config = parse_config(config)

    # Now do bounds/nan check
    start = pd.to_datetime(config["fit_start"])
    end = pd.to_datetime(config["fit_end"])

    elev_filt = cosine_lanczos(elev, "40h")
    elev_tidal = elev.copy() - elev_filt  # isolate tidal part

    log10beta, betalb, betaub = (
        float(config["param"]["log10gbeta"][0]),
        float(config["param"]["log10gbeta"][1]),
        float(config["param"]["log10gbeta"][2]),
    )

    b0, b0_lb, b0_ub = (
        float(config["param"]["b0"][0]),
        float(config["param"]["b0"][1]),
        float(config["param"]["b0"][2]),
    )
    b1, b1_lb, b1_ub = (
        float(config["param"]["b1"][0]),
        float(config["param"]["b1"][1]),
        float(config["param"]["b1"][2]),
    )
    npow, npow_lb, npow_ub = (
        float(config["param"]["npow"][0]),
        float(config["param"]["npow"][1]),
        float(config["param"]["npow"][2]),
    )

    filter_dt = pd.Timedelta(config["filter_setup"]["dt"])
    filter_len = int(config["filter_setup"]["filter_length"])
    filter_k0 = int(config["filter_setup"]["k0"])
    ndofreq = pd.Timedelta(ndo.index.freq)
    dstep = int(filter_dt / ndofreq)  # number of rows for each Dt
    solu_df = pd.concat((ec_obs, ndo, elev_filt, elev_tidal), axis=1)
    solu_df.columns = ["ec_obs", "ndo", "elev_filt", "elev_tidal"]
    for k in range(0, filter_len):
        solu_df[f"z{k}"] = solu_df["elev_tidal"].shift((filter_k0 - k) * dstep)

    # Now do bounds/nan check
    start = pd.to_datetime(config["fit_start"])
    end = pd.to_datetime(config["fit_end"])
    solu_df = solu_df.loc[start:end, :]
    solu_df.to_csv("test.csv", index=True, header=True, float_format="%.3f")

    # now do the fit.


def fit_mrzecest(config, elev=None, ndo=None, ec_obs=None):

    if isinstance(config, str):
        config = parse_config(config)

    elev_filt = cosine_lanczos(elev, "40h")
    elev_tidal = elev.copy() - elev_filt  # isolate tidal part

    # set up filter

    filter_dt = pd.Timedelta(config["filter_setup"]["dt"])
    filter_len = int(config["filter_setup"]["filter_length"])
    filter_k0 = int(config["filter_setup"]["k0"])
    ndofreq = pd.Timedelta(ndo.index.freq)
    dstep = int(filter_dt / ndofreq)  # number of rows for each Dt
    # assert dstep == 12, "Erase this later"

    solu_df = pd.concat((ec_obs, ndo, elev_filt, elev_tidal), axis=1)
    solu_df.columns = ["ec_obs", "ndo", "elev_filt", "elev_tidal"]

    a = config["param"]["afilt"][0]
    a_lb = config["param"]["afilt"][1]
    a_ub = config["param"]["afilt"][2]
    print(a)
    print(a_lb)
    print(filter_len)
    assert len(a) == filter_len
    assert len(a_ub) == filter_len

    # add lagged values
    for k in range(len(a)):
        solu_df[f"z{k+1}"] = solu_df["elev_tidal"].shift((filter_k0 - k) * dstep)

    # Now do bounds/nan check
    start = pd.to_datetime(config["fit_start"])
    end = pd.to_datetime(config["fit_end"])
    solu_df = solu_df.loc[start:end, :]

    # centering: causal   # must be 'centered' or 'causal'

    # Initialize GEKKO model
    # temp_dir = r'./gekko_model'
    # os.makedirs(temp_dir, exist_ok=True)
    m = GEKKO(remote=True)
    elapsed = datetime_elapsed(solu_df).astype("int64")
    m.time = elapsed.index.to_numpy()

    # Set the model mode to dynamic if using time
    m.options.IMODE = 4  # Dynamic mode, as an example
    # m.options.MEAS_CHK = 0
    m.options.DIAGLEVEL = 3  # set solver type

    # input parameters
    s_obs_var = solu_df["ec_obs"].to_numpy().astype(float)
    q_param = m.Param(value=solu_df[["ndo"]].to_numpy().astype(float))

    z = solu_df.filter(like="z").to_numpy()
    z_params = [m.Param(value=z[:, k]) for k in range(filter_len)]

    # Define a specific path for saving temporary files

    # Define variables
    g0 = solu_df.at[start, "ndo"]
    log10beta, betalb, betaub = (
        float(config["param"]["log10gbeta"][0]),
        float(config["param"]["log10gbeta"][1]),
        float(config["param"]["log10gbeta"][2]),
    )

    print("g0", g0)
    b0, b0_lb, b0_ub = (
        float(config["param"]["b0"][0]),
        float(config["param"]["b0"][1]),
        float(config["param"]["b0"][2]),
    )
    b1, b1_lb, b1_ub = (
        float(config["param"]["b1"][0]),
        float(config["param"]["b1"][1]),
        float(config["param"]["b1"][2]),
    )
    npow, npow_lb, npow_ub = (
        float(config["param"]["npow"][0]),
        float(config["param"]["npow"][1]),
        float(config["param"]["npow"][2]),
    )

    g = m.Var(value=g0, lb=-10000.0, ub=1000000)  # Single GEKKO variable over time
    log10beta = m.FV(
        value=log10beta, lb=betalb, ub=betaub
    )  # Coefficient, lower bound .1
    beta0 = m.FV(value=b0, lb=b0_lb, ub=b0_ub)
    beta1 = m.FV(value=b1, lb=b1_lb, ub=b1_ub)
    npow = m.FV(value=npow, lb=npow_lb, ub=npow_ub)  # lower bound .1
    a_param = [m.FV(value=ak) for ak in a]  # Coefficients for lags

    # Enable parameters for optimization
    for param in [log10beta, beta0, beta1, npow] + a_param:
        param.STATUS = 1

    # Define differential equation

    m.Equation(g.dt() == (g * (q_param - g) / 10.0**log10beta))

    # Define output s
    so = m.Const(value=float(config["so"]))  # Example constant
    sb = m.Const(value=float(config["sb"]))  # Example constant
    ndata = m.Const(value=solu_df.shape[0])

    # Summation term for z_k
    # z_sum = m.Intermediate(m.sum([ak * z[:, k] for k, ak in enumerate(a_param)]))
    z_sum = m.Intermediate(
        m.sum([ak * z_param for ak, z_param in zip(a_param, z_params)])
    )

    # Define the model for s
    s = m.Intermediate(m.exp(beta0 + beta1 * g**npow + g * 1.0) * (so - sb) + sb)
    # print("s_obs_var:", s_obs_var.shape, s_obs_var)
    # print("z array:", z.shape, z)
    # print("z_sum:", z_sum)
    # print("qparam length",len(q_param.value))

    # Define error for each observation
    # error = [m.log((s - so) / (so - sb)) - s_obs_var[i] for i in range(len(s_obs_var))]
    # print([e**2 for e in error])  # Check the squared errors list
    # Define the objective as the sum of squared errors

    observed_data = solu_df.ec_obs.to_numpy()
    s_obs = m.Param(value=observed_data)
    error = m.Intermediate((s - s_obs) ** 2)
    print(solu_df.shape)
    print("Objective test:", type((g - s_obs) ** 2))
    squared_errors = [(s - s_obs) ** 2 for _ in m.time]
    m.Obj(m.sum(squared_errors))

    # Solve the problem
    m.solve(disp=True)

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
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise
    return config


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
