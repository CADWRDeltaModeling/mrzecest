from mrzecest.ec_boundary import *
from vtools import rhistinterp
import pandas as pd
import matplotlib.pyplot as plt

def main():
     config = "fitting_config.yaml"
     ndo = pd.read_csv("../data/hist_ndo.csv",header=0,index_col=0,parse_dates=["datetime"])
     ndo = ndo.asfreq('d')
     ndo = ndo.to_period()
     ndo15 = rhistinterp(ndo,'15min',lowbound=-2000.)
     elev = pd.read_csv("../data/mrz_hist_stage.csv",header=0,index_col=0,parse_dates=["datetime"])
     elev = elev.asfreq('15min')
     obs_ec = pd.read_csv("../data/mrz_hist_ec.csv",header=0,index_col=0,parse_dates=["datetime"])
     obs_ec = obs_ec.asfreq('15min')
     obs_ec15 = obs_ec.interpolate(limit=4)     
     #fig, (ax0,ax1,ax2) = plt.subplots(3,sharex=True)
     #ax0.plot(ndo15.index,ndo15.values)
     #ax1.plot(elev.index, elev.values)
     #ax2.plot(obs_ec.index,obs_ec.values)
     fit_mrzecest(config, elev=elev, ndo=ndo15, ec_obs=obs_ec15)
     


if __name__ == "__main__":
    main()
