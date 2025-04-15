import pandas as pd
import matplotlib.pyplot as plt

import os

os.chdir(os.path.dirname(__file__))

mrz_elev = pd.read_csv("mrz_hist_stage.csv", index_col=[0], parse_dates=[0])

fig, ax = plt.subplots(1)
ax.plot(mrz_elev.index, mrz_elev.values)
ax.set_xlabel("")  # X-axis label
ax.set_ylabel("ft")  # Y-axis label
plt.show()

mrz_ec = pd.read_csv("mrz_hist_ec.csv", index_col=[0], parse_dates=[0])

fig, ax = plt.subplots(1)
ax.plot(mrz_ec.index, mrz_ec.values)
ax.set_xlabel("")  # X-axis label
ax.set_ylabel("uS/cm")  # Y-axis label
plt.show()
