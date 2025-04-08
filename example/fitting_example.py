from mrzecest.ec_boundary_fit_gee import *
from mrzecest.ec_boundary import *
from mrzecest.fitting_util import *
from vtools import rhistinterp
import pandas as pd
import matplotlib.pyplot as plt


def main():
    config = "fitting_config.yaml"
    ndo = pd.read_csv(
        "../data/hist_ndo.csv", header=0, index_col=0, parse_dates=["datetime"]
    )
    ndo = ndo.asfreq("d")
    ndo = ndo.to_period()
    ndo15 = rhistinterp(ndo, "h", lowbound=-2000.0)
    elev = pd.read_csv(
        "../data/mrz_hist_stage.csv", header=0, index_col=0, parse_dates=["datetime"]
    )
    elev = elev.asfreq("h")
    obs_ec = pd.read_csv(
        "../data/mrz_hist_ec.csv", header=0, index_col=0, parse_dates=["datetime"]
    )
    obs_ec = obs_ec.asfreq("h")
    obs_ec15 = obs_ec.interpolate(limit=4)
    # fig, (ax0,ax1,ax2) = plt.subplots(3,sharex=True)
    # ax0.plot(ndo15.index,ndo15.values)
    # ax1.plot(elev.index, elev.values)
    # ax2.plot(obs_ec.index,obs_ec.values)
    x_res, coefs, pred_df = fit_mrzecest_gee(
        config, elev=elev, ndo=ndo15, ec_obs=obs_ec15
    )

    log10beta = x_res[0]  # x[0] from ec_boundary_fit_gee.py printout
    npow = x_res[1]  # x[1] from ec_boundary_fit_gee.py printout
    area_coef = x_res[2] * 3600 * 1000000.0  # x[2] from ec_boundary_fit_gee.py printout
    energy_coef = x_res[3] * 1000.0  # x[3] from ec_boundary_fit_gee.py printout
    beta0 = coefs["const"]  # from const coef result
    beta1 = coefs["gnpow"] * 0.001  # from gnpow coef result
    filter_k0 = 6  # from fitting_config.yaml
    filt_coefs = [
        ak * 1e-3 for ak in coefs[coefs.index.str.startswith("z")].values
    ]  # z{n} from output coefs
    filter_dt = pd.Timedelta("3h")  # from fitting_config.yaml
    so = 20000.0  # hardwired in ec_boundary_fit_gee.py
    sb = 200.0  # hardwired in ec_boundary_fit_gee.py

    # align the ndo and elev dataframes
    common_index = elev.index.intersection(ndo15.index)
    ndo15 = ndo15.loc[common_index]
    elev = elev.loc[common_index]

    start = pd.to_datetime("2006-01-01")
    end = pd.to_datetime("2007-01-01")

    # ec_est(ndo, elev, start, end, area_coef,energy_coef, log10beta, beta0, beta1, npow, filter_k0, filt_coefs, filter_dt, so, sb))
    mrzecest = ec_est(
        ndo15,
        elev,
        start,
        end,
        area_coef,
        energy_coef,
        log10beta,
        beta0,
        beta1,
        npow,
        filter_k0,
        filt_coefs,
        filter_dt,
        so,
        sb,
    )
    obs_ec = obs_ec.loc[start:end]
    pred_df = pred_df.loc[start:end]
    pred_df.columns = ["obs", "fit"]
    pred_df = np.exp(pred_df) * (so - sb) + sb  # untransfrm

    fig, ax = plt.subplots(1)
    ax.plot(obs_ec.index, obs_ec.values)
    ax.plot(mrzecest.index, mrzecest.values)
    ax.legend(["obs", "est"])
    plt.show()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
