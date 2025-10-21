# mrzecest
## Martinez EC Estimator

### Math of it All

The estimator uses a few things, derived from [Chapter 11 of the 22nd Annual Progress Report](https://sitesreservoirproject.riptideweb.com/references/REF23/Volume%202/App05A1_Surface_H2O_Model_Assum/Ateljevich_2001_Chapter_11_Improving_Salinity_Estimates.pdf):

#### NDOI

The Net Delta Outflow Index is calculated by:

$$
\text{NDOI} = \text{San Joaquin River} + \text{Sacramento River (I St)} + \text{Yolo Bypass} + \text{Calaveras} + \text{Cosumnes} + \text{Mokelumne} - \text{Northbay Aqueduct} - \text{Exports} - \text{Consumptive Use}
$$

This is an imperfect specification given the diversity of Consumptive Use calculations. So take this with a grain of salt.

NDOI is then modified by the tidal energy and tidal filter in the **ndo_mod** function.

$$
NDO_{mod} = q = NDOI + c_{area} \langle z \rangle + c_{energy} \langle (z - \langle z \rangle )^2 \rangle
$$

where $\langle z \rangle$ is cosine-lancsoz filtered tidal data at Martinez (aka subtide), $c_{area}$ is the area coeficient for the estuary, $c_{energy}$ is the tidal energy coefficient, and $\langle (z - \langle z \rangle )^2 \rangle$ is tidal energy at Martinez.

#### G-Model

The G model, due to Denton, evolves "antecedent outflow" `g(t)` using the following equation:
$$
\frac{dg}{dt} = \frac{g(t)(q(t) - g(t))}{\beta}
$$

Which can be discretized using a Crank Nicolson scheme (which tends to be sufficiently positivity preserving)
$$
\frac{g_{n+1}-g_n}{\Delta t} = \frac{1}{2}\left[\frac{g_n(q_n-g_n)}{\beta}+\frac{g_{n+1}\left(q_{n+1}-g_{n+1}\right)}{\beta}\right]
$$

where $q$ is the adjusted Net Delta Outflow Index or $NDO_{mod}$. After solving for $g_{n+1}$:

$$
g_{n+1}=\frac{\left(q_{n+1}+\frac{2\beta}{\Delta t}\right) \pm \sqrt{\left(q_{n+1}+\frac{2\beta}{\Delta t}\right)^2-4\left[g_n \left(q_n + \frac{2\beta}{\Delta t}\right)\right]}}{2}
$$

this is solved in ec_boundary.py using the **gcalc** function.


#### EC Formula

$$
\ln\left( \frac{S(t) - S_b}{S_0 - S_b} \right) = \beta_0 + \beta_1 \ g(t)^n + g(t)^n \sum_{k=0}^{n_k-1} a_k \ z_{t + k_0 \Delta t - k \Delta t}
$$

where $S$ is salinity, $S(t)$ is salinity at the current timestep, $S_0$ is the ocean salinity, and $S_b$ is river salinity. $\beta_1$ is the g-model weight (unrelated to $\beta$ in the g-model formulation), $n$ is the n-power coefficient, and $a_k$ is the $k^{th}$ coefficient for the $z$ value at that timestep. The values of $k$ generate the time offsets for this term. The EC formula is solved in the **ec_kernel** function. It is also used in the **outer_fit** in `ec_boundary_fit_gee.py` in building the *preds* variable.

##### $Z_{sum}$ term

The $z_{sum}$ term is represented by:

$$
\sum_{k=0}^{n_k-1} a_k \, z_{t + k_0 \Delta t - k \Delta t}
$$

This essentially takes into account which point in the tidal cycle you are in. This is resolved in the **z_sum_term** function. To set up the parameters for $z_{sum}$, you can specify the $a_{k}$ coefficients as an array in *filt_coefs*, the timestep in *filter_dt*, and the $k_0$ value in *filter_k0*. As an example:

If $k_0=6$, $dt=3$ hours, $a=[0.014\cdot10^{-3}, 0.87\cdot10^{-3}, -0.734\cdot10^{-3}, 0.596\cdot10^{-3}, -0.89\cdot10^{-3}, -0.527\cdot10^{-3}]$ then the filter_len is 6 and the $z_{sum}$ term would look like:

$$
z_{sum} = 
a_1\cdot z_{t+k_0 \Delta t - 0 \Delta t} + 
a_2\cdot z_{t+k_0 \Delta t - 1 \Delta t} + 
a_3\cdot z_{t+k_0 \Delta t - 2 \Delta t} + \\
a_4\cdot z_{t+k_0 \Delta t - 3 \Delta t} + 
a_5\cdot z_{t+k_0 \Delta t - 4 \Delta t} + 
a_6\cdot z_{t+k_0 \Delta t - 5 \Delta t}
\\ = 
0.014\cdot10^{-3}\cdot z_{(t+6 \cdot 3) h} + 
0.87\cdot10^{-3}\cdot z_{(t+6 \cdot 3 - 1 \cdot 3) h} + 
-0.734\cdot10^{-3}\cdot z_{(t+6 \cdot 3 - 2 \cdot 3) h} + \\
0.596\cdot10^{-3}\cdot z_{(t+6 \cdot 3 - 3 \cdot 3) h} + 
-0.89\cdot10^{-3}\cdot z_{(t+6 \cdot 3 - 4 \cdot 3) h} + 
-0.527\cdot10^{-3}\cdot z_{(t+6 \cdot 3 - 5 \cdot 3) h}
\\ = 
0.014\cdot10^{-3}\cdot z_{t+18 ~h} + 
0.87\cdot10^{-3}\cdot z_{t+15 ~h} + 
-0.734\cdot10^{-3}\cdot z_{t+12 ~h} + \\
0.596\cdot10^{-3}\cdot z_{t+9 ~h} +
-0.89\cdot10^{-3}\cdot z_{t+6 ~h} + 
-0.527\cdot10^{-3}\cdot z_{t+3 ~h}
$$

so that each value of z has a summed lag term that's used in the EC formula.
