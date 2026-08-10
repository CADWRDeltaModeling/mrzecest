"""Amplitude--salinity diagnostic for the EC-kernel tidal coupling.

This is the falsifiable test of whether the tidal-amplitude coupling has the
right *rate/shape* (as opposed to the trivially-agreed endpoints, where tidal
amplitude -> 0 as the estuary freshens).

Background
----------
For a small tidal swing the modeled EC tidal amplitude is

    dEC ~ (EC - sb) * dA,   A = g**n_tide * sum_k a_k z_k,

so the *EC* tidal amplitude scales like

    amp(g) ~ (so - sb) * exp(b0 + b1 * g**n_mean) * g**n_tide
           = [decaying in g]  x  [growing in g].

That product is NOT monotone: it has an interior peak at

    g* = ( n_tide / (-b1 * n_mean) )**(1/n_mean),

i.e. the model already predicts tidal amplitude that rises from the marine end,
peaks near the front, and falls toward fresh. So there is no contradiction in the
form; the question is whether the *placement/width/rate* of that bump matches data.

What this script does (read-only, uses model.yaml)
--------------------------------------------------
1. Prints the model's analytic amp(g) curve and g*.
2. Builds the *empirical* amplitude--salinity relationship:
     - x-axis : subtidal (low-pass) EC  -> "how salty is it"
     - y-axis : intra-tidal-day EC range -> tidal amplitude
   for BOTH observed and modeled EC, over a multi-year span.
3. Colors observed points by tidal energy (spring/neap) to expose the
   spring--neap dependence.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from vtools import cosine_lanczos, hours
from ndo_chooser import get_ndo

from mrzecest.ec_boundary import ec_est_yaml
import logging


MODEL_YAML = "model.yaml"
PAD = pd.Timedelta("9d")
SPAN_START = pd.Timestamp("2004-01-01")
SPAN_END = pd.Timestamp("2016-12-31")
ROLL_N = 101  # ~25.25 h at 15 min: one lunar day
EC_BINS = np.arange(0.0, 46000.0, 2500.0)


def _analytic_amp_curve():
    m = yaml.safe_load(open(MODEL_YAML))
    b0 = m["inner_linear"]["b0"]
    b1 = m["inner_linear"]["b1"]
    nm = m["g_model"]["npow"]
    g_thr_tide = m["g_model"]["g_thr_tide"]
    width_frac_tide = m["g_model"].get("width_frac_tide", 0.6)
    width_tide = max(1e-6, width_frac_tide * g_thr_tide)
    so = m["constants"]["so_uS_cm"]
    sb = m["constants"]["sb_uS_cm"]
    print(
        f"b0={b0:.4f} b1={b1:.6f} n_mean={nm:.3f} g_thr_tide={g_thr_tide:.0f} "
        f"width_frac_tide={width_frac_tide:.2f} so={so:.0f} sb={sb:.0f}"
    )
    ec_at = lambda g: (so - sb) * np.exp(b0 + b1 * g**nm) + sb
    gate = lambda g: 1.0 / (1.0 + np.exp(-(g - g_thr_tide) / width_tide))
    print("  analytic model amp(g) ~ (EC-sb)*W(g)  [relative units]:")
    for g in [1000, 3000, 6000, 10000, 20000, 40000, 80000, 150000]:
        amp = (so - sb) * np.exp(b0 + b1 * g**nm) * gate(g)
        print(f"    g={g:9.0f}  EC_mean={ec_at(g):7.0f}  W={gate(g):5.3f}  rel_amp={amp:12.1f}")
    return sb, so


def _amp_sub(ec: pd.Series):
    """Intra-tidal-day range and subtidal (mean) via a centered lunar-day window."""
    roll = ec.rolling(ROLL_N, center=True, min_periods=60)
    sub = roll.mean()
    rng = roll.max() - roll.min()
    return rng, sub


def _binned_median(x, y, bins):
    idx = np.digitize(x.to_numpy(), bins)
    centers, meds = [], []
    for b in range(1, len(bins)):
        sel = idx == b
        if sel.sum() >= 20:
            centers.append(0.5 * (bins[b - 1] + bins[b]))
            meds.append(float(np.nanmedian(y.to_numpy()[sel])))
    return np.array(centers), np.array(meds)


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    _analytic_amp_curve()

    # --- load inputs ---
    ndo = get_ndo("dayflow", "15min")
    elev = pd.read_csv(
        "./data/dms_mrz_elev_filled.csv", header=0, index_col=0, parse_dates=["datetime"]
    ).asfreq("15min").squeeze("columns")
    obs = pd.read_csv(
        "./data/dms_mrz@upper_ec.csv", header=0, index_col=0, parse_dates=["datetime"], comment="#"
    ).squeeze()

    ndo_in = ndo.loc[SPAN_START:SPAN_END]
    elev_in = elev.loc[SPAN_START - PAD : SPAN_END + PAD]

    # --- modeled EC over the span ---
    ec_model = ec_est_yaml(ndo_in, elev_in, MODEL_YAML).astype(float)

    # --- observed EC on a regular grid (keep gaps as NaN) ---
    obs15 = obs.resample("15min").asfreq().loc[SPAN_START:SPAN_END]

    # --- tidal energy (spring/neap), model definition ---
    elev_lp = cosine_lanczos(elev_in, "40h")
    elev_tidal = elev_in - elev_lp
    energy = cosine_lanczos(elev_tidal * elev_tidal, "40h").loc[SPAN_START:SPAN_END]

    # --- amplitude & subtidal for obs and model ---
    obs_rng, obs_sub = _amp_sub(obs15)
    mod_rng, mod_sub = _amp_sub(ec_model)

    # --- daily reduction to a readable cloud ---
    daily = pd.DataFrame(
        {
            "obs_rng": obs_rng,
            "obs_sub": obs_sub,
            "mod_rng": mod_rng,
            "mod_sub": mod_sub,
            "energy": energy.reindex(ec_model.index),
        }
    ).resample("1D").median()

    obs_d = daily[["obs_rng", "obs_sub", "energy"]].dropna()
    mod_d = daily[["mod_rng", "mod_sub"]].dropna()

    # --- binned-median amplitude vs subtidal EC (the curve) ---
    cx_o, cy_o = _binned_median(obs_d["obs_sub"], obs_d["obs_rng"], EC_BINS)
    cx_m, cy_m = _binned_median(mod_d["mod_sub"], mod_d["mod_rng"], EC_BINS)

    print("\nbinned median tidal amplitude (uS/cm) vs subtidal EC (uS/cm):")
    print("  subtidalEC     obs_amp    model_amp")
    allc = sorted(set(np.round(cx_o).astype(int)) | set(np.round(cx_m).astype(int)))
    o_map = {int(round(c)): v for c, v in zip(cx_o, cy_o)}
    m_map = {int(round(c)): v for c, v in zip(cx_m, cy_m)}
    for c in allc:
        print(f"  {c:9d}   {o_map.get(c, float('nan')):9.0f}   {m_map.get(c, float('nan')):9.0f}")

    # --- plots ---
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))

    axA.plot(cx_o, cy_o, "o-", color="0.2", label="observed")
    axA.plot(cx_m, cy_m, "s-", color="C1", label="model")
    axA.set_xlabel("subtidal EC (uS/cm)")
    axA.set_ylabel("intra-tidal-day EC range (uS/cm)")
    axA.set_title("Amplitude vs salinity (binned median)")
    axA.legend()

    sc = axB.scatter(
        obs_d["obs_sub"], obs_d["obs_rng"], c=obs_d["energy"], s=8, cmap="viridis"
    )
    axB.set_xlabel("subtidal EC (uS/cm)")
    axB.set_ylabel("observed intra-tidal-day EC range (uS/cm)")
    axB.set_title("Observed amplitude, colored by tidal energy (spring/neap)")
    fig.colorbar(sc, ax=axB, label="tidal energy")

    fig.tight_layout()
    fig.savefig("amplitude_salinity.png", dpi=110)
    print("\nsaved figure -> amplitude_salinity.png")
    plt.show()


if __name__ == "__main__":
    main()
