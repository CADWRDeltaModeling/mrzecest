

# mrzecest  
## Martinez Electrical Conductivity (EC) Estimator

The estimator uses a few things, derived from [Chapter 11 of the 22nd Annual Progress Report](https://sitesreservoirproject.riptideweb.com/references/REF23/Volume%202/App05A1_Surface_H2O_Model_Assum/Ateljevich_2001_Chapter_11_Improving_Salinity_Estimates.pdf):
The formulation is revised here.

`mrzecest` estimates electrical conductivity (EC) at Martinez using Net Delta Outflow (NDO) and tidal information.  
The formulation follows the conceptual framework developed in Chapter 11 of the 2001 Annual Progress Report, expressed in a modern, internally consistent form suitable for fitting and operational evaluation.

The estimator combines:
- **Antecedent transport dynamics** (the Denton *g-model*),
- **Tidal phasing effects** represented through a lagged elevation filter,
- **Energy-dependent stratification effects** that modulate effective transport,
- A **log-linear EC kernel** with physically interpretable bounds.

The intent is not to reproduce EC point-by-point, but to capture the dominant transport and tidal controls governing salt intrusion at the seaward boundary of the Delta.

---

## Overview of the Algorithm

At a high level, the estimator proceeds in four stages:

1. **Construct physically meaningful features** from NDO and water level.
2. **Evolve antecedent transport** using a nonlinear storage model.
3. **Apply stratification-dependent corrections** tied to tidal energy and transport regime.
4. **Map transport and tidal phase to EC** through a bounded log-linear kernel.

Each step is described below.

---

## Inputs and Preprocessing

### Required Inputs

- **Net Delta Outflow (NDO)**  
  A regular time series (15-minute or hourly) representing Delta outflow in cfs. Common sources are Dayflow, DSM2 input 
  or SCHISM input. 

- **Water surface elevation at Martinez**  
  A regular time series at the same temporal resolution as NDO, with sufficient padding to support filtering.

No internal interpolation or gap filling is performed. Inputs must be complete and uniformly sampled.

---

### Feature Construction

All filtering and derived quantities are built once using a common routine:

- **Subtidal elevation**  
  A cosine–Lanczos low-pass filter applied to water level.

- **Tidal elevation**  
  The residual between raw elevation and subtidal elevation.

- **Tidal energy proxy**  
  A low-pass filtered version of squared tidal elevation.

- **Time derivative of subtidal elevation**  
  A centered finite difference, used to represent effective surface-area exchange.

- **Lagged tidal matrix**  
  A set of phase-shifted tidal elevations:

$$
z_k(t) = z\bigl(t + (k_0 - k)\,\Delta t\bigr)
$$

These are **not summed initially**; the summation happens inside the EC kernel.

This design ensures strict time alignment and eliminates ambiguity about where filtering occurs.

---

## Modified Net Delta Outflow

The estimator works with a *modified* outflow signal:

$$
q_{\text{base}}(t) = \text{NDO}(t) + c_{\text{area}} \,\frac{d}{dt}\langle z \rangle
$$

where:
- $\langle z \rangle$ is subtidal elevation,
- $c_{\text{area}}$ is a fitted coefficient representing effective estuarine surface area.

At this stage, no tidal-energy correction is applied. This “base” signal defines the transport regime.

---

## Antecedent Transport (g-Model)

Transport memory is represented by the Denton *g-model*:

$$
\frac{dg}{dt} = \frac{g\,(q - g)}{\beta}
$$

This equation describes how antecedent transport $g(t)$ evolves toward the current forcing $q(t)$ with a characteristic time scale controlled by $\beta$.

In discrete time, the model is solved using an implicit trapezoidal (Crank–Nicolson) scheme, which is stable and preserves positivity for realistic parameter ranges. 
The model can survive period of negative outflow, which is fairly common both in the field and in the model input when things like the filling and draining (area-based)
correction or energy is considered. In this case simply delaying or anticipating the start of integration by a week usually does the trick. 
Once the model gets Cranking it usually traverses periods of negative q very well, producing positive g (the continuum equation does not have the negative g problem,
that is just an artifact of integration). It is possible with a transformation to fix even the initial values, but we haven't prioritized this. 
Do NOT floor the modeified inflow -- it is a natural reaction, but it is significant and not correct.  

The result is a smooth, history-aware transport proxy that responds gradually to changes in NDO.

---

## Stratification and Energy Effects

### Motivation

Observed EC behavior shows that transport efficiency changes with tidal energy and flow regime. Specifically, at Martinez salinity
intrusion "explodes" during very weak neap tides when flow is modest and the longitudinal gradient is near Martinez. During these
periods, the time scale of transport is practically zero. In 3D we usually see very large (e.g. 10psu in practical salinity units) 
surface-bottom stratification at local sites. The stratification builds from tide cycle to tide cycle, an incipient form of 
what Stacey and Monismith called "runaway stratification". 

Rather than applying tidal energy as a direct additive term, the estimator adjusts **effective transport** in a regime-aware manner.

The term "front" below refers both to the physical environment conditions (salinity front is passing Martinez) and to the way the 
effect is modeled as an adjstment to $g_{\text{thr}}$, with the adjustment activated for low-medium values of salinity and then 
ramped away at values above 20ms/cm. For higher values of salinity, the dominant transport mechanisms are variants
of tidal stirring and the jet-sink balance at Benicia. Salinity intrusion is driven more by spring tide at that point, and the
effect is hard to incorporate because of colinearity with other factors such as subtidal filling and draining.

---

### Transport Regime Weighting

A **soft frontal weight** is constructed from the base transport:

$$
w_{\text{front}}(t)
= \frac{1}{1 + \exp\!\left[-\frac{g_{\text{base}}(t) - g_{\text{thr}}}{\alpha\,g_{\text{thr}}}\right]}
$$

- $g_{\text{thr}}$ is the transport level corresponding to a reference EC (e.g. 20 mS/cm) under mean tidal conditions.
- The transition width scales with $g_{\text{thr}}$, making the weighting dimensionless and robust.

This smoothly distinguishes background conditions from strong frontal exchange.

---

### Energy Weighting

Tidal energy modulates the strength of stratification effects:

$$
w_{\text{energy}}(t) = \frac{1}{1 + E(t)/E_{\text{ref}}}
$$

Low-energy conditions allow stronger stratification influence; high-energy conditions suppress it.

---

### Stratification Correction

The final correction applied to transport is:

$$
\Delta q(t) = c_{\text{energy}} \; w_{\text{front}}(t)\; w_{\text{energy}}(t)
$$

The **effective forcing** driving the g-model becomes:

$$
q(t) = q_{\text{base}}(t) - \Delta q(t)
$$

The g-model is then recomputed using this corrected forcing.

---

## EC Kernel

Electrical conductivity is modeled with a bounded log-linear kernel. The tidal
term is scaled by a **bounded logistic gate** $W(g)\in[0,1]$ (this replaces an
earlier, unbounded $g^{n_{\text{tide}}}$ factor — see *Lessons Learned*):

$$
\ln\!\left(\frac{EC(t) - S_b}{S_o - S_b}\right)
= \beta_0 + \beta_1 \, g(t)^{n_{\text{mean}}}
+ W\!\big(g(t)\big) \sum_{k} a_k \, z_k(t)
$$

$$
W(g) = \sigma\!\left(\frac{g - g_{\text{thr}}^{\text{tide}}}{\text{width\_frac\_tide}\;\cdot\;g_{\text{thr}}^{\text{tide}}}\right),
\qquad \sigma(x) = \frac{1}{1 + e^{-x}}
$$

where:
- $S_o$ is ocean EC,
- $S_b$ is background (river) EC,
- $\beta_0$ and $\beta_1$ are fitted mean-term coefficients,
- $n_{\text{mean}}$ controls sensitivity of the mean EC response to transport,
- $g_{\text{thr}}^{\text{tide}}$ and $\text{width\_frac\_tide}$ set the location and sharpness of the tidal gate,
- $a_k$ are lagged tidal coefficients.

Linearizing the exponential shows the observable EC tidal amplitude is
approximately $(EC - S_b)\,W(g)\times(\text{tidal swing})$: it **vanishes toward
fresh** through the $(EC - S_b)$ prefactor, and holds a **bounded plateau at high
salinity** because $W(g)$ is bounded and rises with $g$ (falls with salinity).

Note: the exponential form guarantees a *lower* bound $EC \ge S_b$ (a floor); it
does **not** by itself impose the upper bound $EC \le S_o$. In practice the fitted
coefficients keep the exponent negative.

---

## Model Fitting

Fitting is performed in two nested steps:

1. **Outer optimization**  
   Searches over physically meaningful nonlinear parameters:
   - $\log_{10}\beta$
   - $n_{\text{mean}}$
   - tidal-gate threshold $g_{\text{thr}}^{\text{tide}}$ and fractional width
   - area coefficient
   - energy coefficient
   - (optionally) ocean salinity

2. **Inner conditional fit**  
   Given $g(t)$, the EC kernel is fit using a Gamma GLM with a log link.

This separation keeps the optimization stable and interpretable, while allowing flexible nonlinear structure.

---

## Getting Started

### Fitting a Model

```python
from mrzecest.ec_boundary_fit import fit_mrz_ecest
from mrzecest.fitting_util import build_model_from_fit, write_model_yaml

x_res, coefs, ypred, front_spec = fit_mrz_ecest(
    "fitting_config.yaml",
    elev=elev_series,
    ndo=ndo_series,
    ec_obs=observed_ec
)

model = build_model_from_fit(
    "fitting_config.yaml",
    x_res,
    coefs,
    front_spec=front_spec
)

write_model_yaml(model, "model.yaml")
```

---

## Lessons Learned

Notes accumulated while revising the tidal formulation. These record *why* the
current form looks the way it does, and which alternatives were tried and rejected.

### Tidal term: bounded gate, not an unbounded power
- The original tidal factor $g^{n_{\text{tide}}}$ is **unbounded**. Pushed toward the
  values needed to place the amplitude correctly, it drove the inner Gamma–IRLS
  fit to the saturation rails (predicted ratio $\to 0/1$), producing
  `NaN`/`inf` weights and "estimation infeasible" failures. Normalizing
  ($ (g/g_{\text{ref}})^{n_{\text{tide}}} $) fixed the *scaling* but not the IRLS
  infeasibility.
- Replacing it with a **bounded logistic gate** $W(g)=\sigma((g-g_{\text{thr}}^{\text{tide}})/(\text{width\_frac\_tide}\cdot g_{\text{thr}}^{\text{tide}}))$
  keeps the inner GLM feasible everywhere, keeps the tidal design column linear
  (so the two-stage GLM still applies), and is physically interpretable: the
  threshold sets the salinity of the amplitude "knee" (~EC 10 mS/cm ↔ $g\approx20{,}000$ cfs),
  and a **fitted** width controls how flat the high-salinity plateau is.

### The observable tidal amplitude is multiplicative
- Linearizing the kernel, the EC tidal amplitude is $\approx (EC-S_b)\,W(g)\times$(tidal swing).
  It **must vanish as the estuary freshens** through the $(EC-S_b)$ prefactor.
- Consequence: an **un-gated additive** tidal term is unphysical — it would keep
  oscillating on fully fresh water. Tried and **rejected**.
- The fresh-side rise is handled by $(EC-S_b)$; the high-salinity roll-off (rarely
  reached at Martinez) is handled by $W(g)$. With $n_{\text{tide}}>0$-style rising
  behavior the amplitude is a broad interior bump, **not** monotone — there was no
  "contradiction" in the original sign.

### Observed amplitude–salinity shape (from the diagnostic)
- Tidal EC amplitude rises steeply to subtidal EC $\approx$ 10 mS/cm, then a broad
  **high plateau** (~14–16 mS/cm) with a slight decline; the fully-marine roll-off
  is essentially **unobserved** (subtidal EC tops out ~31 mS/cm; marine saturation
  is rare, fresh saturation is common). So we fit the rise + plateau and do **not**
  impose a marine roll-off in range. A single logistic gate with fitted width
  reproduces this; the gentle mid-hump is a minor, cosmetic residual.

### Objective function matters
- The inner **Gamma deviance (log link)** is a *relative-error* measure dominated by
  low-EC points, so it under-weights high-EC amplitude — leaving a ~15% low bias on
  the plateau. **RMSE / least-squares** value high-EC absolute error.
- Up-weighting the GLM toward high EC (`var_weights = y**weight_power`) was tried to
  lift the plateau. It **destabilized the outer fit** (the search re-solved and
  collapsed `energy_coef`) and worsened the rising limb without lifting the plateau —
  **reverted** (`weight_power = 0`; knob retained). The ~2 mS/cm plateau bias is
  accepted as cosmetic.
- With the improved gate, the GLM and least-squares objectives **reconciled**
  (per-sample deviance became ~equal) — a good sign the *form* improved rather than
  one objective being traded for another.

### Fitting window / data
- Fit on **2004–2016** (spans both wet and dry extremes → broad EC coverage,
  including the fresh end the relative-error criterion needs) generalizes forward
  better than fitting the saltier, narrower 2018–2024 window.
- **DSM2** historical NDO ends 2018-01; use **Dayflow** (through ~2024-10) for a
  genuine multi-year out-of-sample window.
- Fixing $\beta_0=0$ anchors $S_b$ as the asymptotic-low EC. The least-squares
  "touch-up" honors this (`fit_run.inner.b0.fix`).
