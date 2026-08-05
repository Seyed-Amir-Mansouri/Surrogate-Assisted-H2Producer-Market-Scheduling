# Formulation

Two separate mathematical models live in this project, used together but solved
completely separately:

1. **Training** (`price_model/`) — per-zone gradient-boosted regression, always fit on
   the **full year, all 20 CORE-region zones** (`inputs/hourly_balance_{elec,h2}.csv`).
2. **Optimization** (`optimize_h2_producer.py`) — a small linear program for **one
   country's Hydrogen Producer, one zone, one day at a time**, that reuses Project 3's
   own Hydrogen Producer physics but replaces the real network coupling with a price
   signal read off the model trained in (1).

§3 documents the validation done so far (DE00, day 5) comparing the optimizer's solved
schedule against the actual historical Hydrogen Producer schedule, including a real,
unresolved discrepancy worth knowing about before trusting this for anything further.

---

## §1 Training — `price_model`

### 1.1 Per-zone regression problem

For each zone $z$ and commodity (electricity/hydrogen), a separate model
$\hat f_z: \mathbb{R}^{n_z} \to \mathbb{R}$ is fit to predict price from a feature vector:

$$\hat p_{z,h} = \hat f_z(x_{z,h}), \qquad x_{z,h} \in \mathbb{R}^{n_z}$$

Training rows are restricted to that zone's **active hours** ($\text{demand}_{z,h} > 0$;
electricity additionally drops hours where $\text{price}_{z,h} > 500$ EUR/MWh — extreme
VOLL/shedding spikes). A zone needs $\geq 200$ active-hour samples (`MIN_SAMPLES`) or it
gets no model at all. Every zone is trained independently — **there is no joint/network
consistency enforced across zones** at training time (see §2.3 for why that matters).

### 1.2 Feature vector $x_{z,h}$

Base features (`price_model/config.py`), current as of 2026-08-04:

| Commodity | Base features |
|---|---|
| Electricity | `demand, residual_load, wind, solar, month, season, hour` |
| Hydrogen | `h2_demand, month, season, hour` |

$\text{residual\_load} = \text{demand} - \text{wind} - \text{solar}$ (merit-order net
demand).

**Changelog** (both dropped at user request, retrain-verified each time — see
conversation history for the exact numbers):
- 2026-08-04: dropped `ens`/`dumped` (electricity) and `dumped`/`hns` (hydrogen) —
  dispatch *outcomes*, not exogenous inputs; mean CV R² unaffected (0.977/0.999 both
  ways) since these carried ~0 importance in this 20-zone dataset already.
- 2026-08-04: dropped `elec_price` (hydrogen) — the *same zone's* own electricity price
  at the same hour (1:1 zone mapping, no country-level aggregation needed). An ablation
  across all 13 zones showed it does essentially nothing for 11 of them (already
  dominated by a correlated neighbour's own H2 price) but costs **SI00/HR00 ~0.06 mean
  CV R² each** — the two zones with no strong correlated-neighbour fallback (mean across
  all zones: 0.999 → 0.990). Dropped anyway, accepting that cost, for a smaller and more
  defensibly-exogenous feature list (`elec_price` is only exogenous under the
  price-taker framing in §2.3, not unconditionally).
- 2026-08-04: **dropped `residual_load` (electricity)**, **added `smr` (hydrogen)**, at
  user request — full per-zone before/after CV R² in conversation history, headline:
  - Electricity mean CV R² 0.977 → 0.971 (−0.006). Costs concentrate almost entirely in
    the zones with no strong correlated-neighbour price to fall back on: **FR15 −0.077**
    (0.966→0.889 — FR15 has zero declared electricity interconnection in this dataset,
    so `residual_load` was carrying real, otherwise-unrecoverable signal for it), plus
    smaller hits to PL00 (−0.008), RO00 (−0.004), CZ00 (−0.003), HR00 (−0.002). Every
    zone with a dominant correlated-neighbour price (`price_X` importance ≫ everything
    else) is unaffected (Δ ∈ [−0.0005, +0.0007]) — makes sense, since
    `residual_load = demand − wind − solar` was redundant with `demand`/`wind`/`solar`
    (already in the feature list) for those zones anyway; only where the model actually
    needed the merit-order signal on its own does removing the pre-computed column cost
    anything (gradient-boosted trees can *approximately* reconstruct a linear
    combination via multiple splits, but not as cleanly as being handed the column
    directly).
  - Hydrogen mean CV R² 0.990 → 0.991 (+0.002, net positive). Nearly all of the gain is
    **HR00: +0.023** (0.939→0.962, `smr` becomes HR00's #2 feature by importance) — the
    other weak-neighbour zone, **SI00, is completely unaffected** (0.936→0.936 exactly)
    despite the same "no strong correlated-neighbour fallback" profile as HR00, implying
    SI00's price isn't meaningfully driven by SMR output specifically. Every
    neighbour-price-dominated zone is unaffected (Δ = 0.0000–0.0001).

Each zone's base list is extended per-zone (`price_model/neighbors.py`) with:
- $\text{demand\_system\_total}_h = \sum_{z'} \text{demand}_{z',h}$ (and the net/residual-load
  version for electricity) — same value for every zone, a given hour.
- $\text{neighbor\_demand}_{z,z',h}$ for every $z'$ physically interconnected to $z$
  (`inputs/elec_adjacency.json` / `h2_adjacency.json`, built from Project 3's
  `networks_2030.parquet` line topology), plus their sum
  $\text{neighbor\_demand\_total}_{z,h}$.
- $\text{price}_{z',h}$ for the **top-5 system-wide price-correlated** zones $z'$
  (regardless of physical interconnection) — computed once via
  $\text{corr}(z,z') = \text{Corr}_h(\text{price}_{z,h},\, \text{price}_{z',h})$ and the
  5 highest kept.

The effective feature count $n_z$ therefore differs per zone (declared neighbours range
0–9 for electricity, 0–7 for hydrogen).

### 1.3 Model: histogram gradient-boosted regression trees

`sklearn.ensemble.HistGradientBoostingRegressor`, identical hyperparameters for every
zone:

$$\hat f_z(x) = \sum_{m=1}^{M} \nu \cdot h_m(x), \qquad M = 300,\ \nu = 0.06$$

Built additively (standard gradient boosting): $F_0(x) = \bar y$; for $m=1..M$, fit tree
$h_m$ to the negative gradient of the loss w.r.t. the current ensemble's predictions,
restricted to $\leq 20$ leaves per tree, add it in scaled by the learning rate $\nu$:

$$F_m(x) = F_{m-1}(x) + \nu\, h_m(x)$$

**Loss function**: squared error (L2), scikit-learn's default —
$L(y,F) = \tfrac{1}{2}(y-F)^2$ — with $\text{l2\_regularization}=1.0$ added as a penalty on
leaf output values during each tree's split search (shrinks leaf values toward 0,
standard regularized-GBM leaf formula $w_j^\star = -G_j/(H_j+\lambda)$ for leaf $j$'s
accumulated gradient/hessian $G_j, H_j$).

### 1.4 Cross-validation and reported accuracy

5-fold **shuffled** K-fold, out-of-fold predictions via `cross_val_predict`:

$$R^2 = 1 - \frac{\sum_h (y_h - \hat y_h^{\text{oof}})^2}{\sum_h (y_h - \bar y)^2}, \qquad
\text{RMSE} = \sqrt{\frac{1}{H}\sum_h (y_h - \hat y_h^{\text{oof}})^2}$$

The **final** model used everywhere else (API, this optimizer) is refit on **all** of a
zone's active-hour data after CV scoring — CV is only used to get an honest accuracy
estimate, not to select the deployed model.

Permutation importance (3 repeats, `sklearn.inspection.permutation_importance`) is
computed on the final fitted model, for diagnostics only — not part of the training
objective.

### 1.5 Prediction-time feature fallback (`price_model/api.py`)

$$x_{z,h}^{(k)} = \begin{cases} \text{caller-supplied value} & \text{if given} \\
\text{median}_z^{(k)} & \text{otherwise} \end{cases}$$

where $\text{median}_z^{(k)}$ is that zone's own training-data median for feature $k$,
stored in the trained bundle. `residual_load` is recomputed from `demand`/`wind`/`solar`
if not explicitly overridden. This optimizer does **not** use this fallback path — every
feature is supplied from real historical data (§2.4).

### 1.6 Headline accuracy (declared-neighbor-price feature set — see conversation
history for the full per-zone table)

| | GBM mean CV R² | Linear mean CV R² |
|---|---:|---:|
| Electricity (16 zones) | 0.977 | 0.904 |
| Hydrogen (13 zones) | 0.999 | 1.000 |

---

## §2 Optimization — `optimize_h2_producer.py`

### 2.1 Scope

Solves **one country's Hydrogen Producer, attached to one zone, over one day (24h)** at a
time — e.g. `--zone DE00 --day 5`. This is deliberately much smaller than training:
training always stays whole-year/all-zones (§1), but a single-day/single-zone solve is
enough to test whether the trained proxy is usable as a market-price signal for
optimizing one country's own asset.

### 2.2 Reused physics: the Hydrogen Producer

The internal unit — wind + PV + battery + electrolyser + H2 tank + flexible downstream
H2 demand — and its **sizing** are taken directly from Project 3's own model
(`economic_dispatch/model.py::_h2_producer_sizing` / `_build_h2_producer`, vendored
locally, copied from Project 3), so the
comparison in §3 is apples-to-apples on physics and capacities. Sizing for a country $c$
is rank-assigned against its full-year reference H2 load (§18.2 of Project 3's own
Formulation.md) — e.g. for DE (`electrolyser_mw`, `wind_mw`, `pv_mw`, `battery_mw/mwh`,
`tank_mw/mwh`) = $(40,\ 7.5,\ 5.0,\ 10/20,\ 12.5/300)$, confirmed to match Project 3's own
documented verified spread exactly.

**Decision variables**, hour $h \in \{0,\dots,23\}$ (all $\geq 0$ unless noted):
$p^{wind}_h,\, p^{pv}_h,\, p^{batt,dis}_h,\, p^{batt,ch}_h,\, \text{soc}^{batt}_h,\,
p^{ely}_h,\, p^{tank,dis}_h,\, p^{tank,ch}_h,\, \text{soc}^{tank}_h,\, p^{ely,ren}_h \geq 0$;
free-sign bounded $x^{grid}_h \in [-40, 40]$, $x^{h2}_h \in [-20, 20]$ MW
(`h2_producer_grid/h2_connection_mw`); flexible demand
$d_h \in [\text{Base}-\text{Flex},\ \text{Base}+\text{Flex}]$; horizon-total
$g^{buy}, g^{sell} \geq 0$.

**Constraints** (identical to Project 3's own, just re-derived over $H=24$ instead of
$H=8736$ — see the full LaTeX in the earlier conversation transcript, reproduced here in
short form):

$$p^{wind}_h+p^{pv}_h+p^{batt,dis}_h-p^{batt,ch}_h-p^{ely}_h-x^{grid}_h=0 \quad\text{(elec. balance)}$$
$$\eta_{ely}\,p^{ely}_h+p^{tank,dis}_h-p^{tank,ch}_h-d_h-x^{h2}_h=0 \quad\text{(H2 balance)}$$
$$p^{ely,ren}_h\le p^{wind}_h+p^{pv}_h,\qquad p^{ely,ren}_h\le p^{ely}_h$$
$$g^{buy}\le{\textstyle\sum_h}p^{ely}_h-{\textstyle\sum_h}p^{ely,ren}_h,\qquad
g^{sell}\le{\textstyle\sum_h}(p^{wind}_h+p^{pv}_h)-{\textstyle\sum_h}p^{ely,ren}_h$$
$$\text{soc}_h-\text{soc}_{h-1}-\eta\,ch_h+dis_h=0\ \ (\text{battery \& tank, cyclic }\text{soc}_{23}\ge\text{soc}_0)$$
$${\textstyle\sum_h} d_h = 24\cdot\text{Base} \quad\text{(demand conservation, THIS DAY only — see §2.5)}$$
$$\eta_{ely}\big({\textstyle\sum_h}p^{ely,ren}_h+g^{buy}\big)\ge 0.42\,{\textstyle\sum_h}d_h \quad\text{(RED III quota, THIS DAY only — see §2.5)}$$

### 2.3 The key modeling choice: price-taker objective, not network coupling

Project 3's own model adds $x^{grid}_h$/$x^{h2}_h$ **into the host zone's real nodal
balance constraint**, so their implied price is whatever the joint LP's dual comes out
to. This optimizer does not build that network at all — instead:

$$\min\ \sum_h\Big[-\hat p^{elec}_{z,h}\,x^{grid}_h - \hat p^{h2}_{z,h}\,x^{h2}_h\Big]
+ c_{sto}\!\!\sum_h(\cdots) + p_{gc}(g^{buy}-g^{sell})$$

$\hat p^{elec}_{z,h}$/$\hat p^{h2}_{z,h}$ are **fixed parameters**, not variables — the
trained proxy's own price prediction (§1), not a real network dual. This is only valid
under a **price-taker assumption**: the Producer's volume (5–40 MW) must be negligible
next to the host zone's own demand (thousands of MW) so its trading doesn't actually move
the zone's price. That's true here by construction (verified in conversation: DE00's own
demand is ~13 GW).

### 2.4 Where $\hat p_{z,h}$ comes from (`proxy_price_series`)

For each hour of the chosen day, this pulls that zone's **real recorded** feature values
(demand, wind, solar, calendar, and — critically — **real recorded neighbour prices for
that same hour**, not medians) from `inputs/{elec,h2}_samples.parquet`, builds exactly the
feature vector the shipped model was trained on
(`bundle["zones"][zone]["features"]`), and calls the trained model directly
(`multivariate.predict`). Using *real* neighbour prices here is legitimate specifically
**because this is a backtest over already-realized history**, not a forward what-if
scenario — there is no circularity, since nothing here is unknown at call time (contrast
with the general concern raised earlier in conversation, that neighbour prices are
endogenous to the joint system for a genuine forward/what-if optimization).

### 2.5 Known limitation: single-day horizon scope vs. the year-scoped ground truth

The **actual** schedule being compared against in §3 came from a *full-year* joint solve,
where `demand_conservation` and the RED III `quota` constraint are scoped to the whole
8736-hour year. This optimizer scopes the exact same constraints to just the 24 hours
being solved (this is not a bug — it's exactly what Project 3's own CLI would build if
you ran `run_dispatch.py --h2-producer --start-day 5 --end-day 5`). The consequence,
confirmed by direct diagnostic (§3.2): the quota can force the electrolyser to run *on a
day it otherwise wouldn't need to*, if that's the only day in scope to satisfy a
day-scoped 42% target — a distortion the real year-scoped solve doesn't have, since it
can bank compliance across all 364 days. **This is a real modeling limitation of solving
day-by-day, not a proxy-price accuracy problem**, and it should be kept in mind for any
future day-level solve, not just DE00/day 5.

---

## §3 Validation: DE00, day 5

### 3.1 Headline comparison (quota=0.42, the faithful default)

| Variable | R² | MAE | RMSE | actual mean | optimized mean |
|---|---:|---:|---:|---:|---:|
| H2 Producer wind (MW) | 1.000 | 0.000 | 0.000 | 2.264 | 2.264 |
| H2 Producer pv (MW) | 1.000 | 0.000 | 0.000 | 0.212 | 0.212 |
| H2 Producer battery discharge (MW) | 0.596 | 0.450 | 2.048 | 1.217 | 0.833 |
| H2 Producer battery charge (-) (MW) | 0.477 | 1.031 | 2.385 | −1.322 | −0.906 |
| H2 Producer electrolyser load (-) (MW) | n/a* | 13.440 | 22.788 | 0.000 | −13.440 |
| H2 Producer grid exchange (MW) | −19.45 | 13.474 | 22.284 | 2.371 | −11.036 |
| H2 Producer electrolyser production (MW) | n/a* | 9.139 | 15.496 | 0.000 | 9.139 |
| H2 Producer tank discharge (MW) | n/a* | 2.083 | 5.103 | 0.000 | 2.083 |
| H2 Producer tank charge (-) (MW) | n/a* | 2.083 | 5.103 | 0.000 | −2.083 |
| H2 Producer downstream demand (-) (MW) | n/a* | 3.488 | 4.641 | −20.000 | −21.760 |
| H2 Producer pipeline exchange (MW) | n/a* | 7.379 | 10.722 | −20.000 | −12.621 |

\* R² undefined where the actual series is exactly constant (variance-free denominator).

**Wind/PV: essentially exact** (weather-availability-bounded, no real decision to get
wrong). **Battery: moderate** (R²≈0.5–0.6). **Electrolyser, grid exchange, pipeline
exchange: large divergence** — the actual schedule ran the electrolyser at **exactly
zero** all day, met its entire 20 MW downstream demand via pipeline import instead; the
optimizer ran the electrolyser hard (mean −13.4 MW).

### 3.2 Diagnostic: disabling the quota isolates most of the electrolyser divergence

Re-solving the same day with `h2_producer_renewable_h2_quota=0`:

| | quota=0.42 (default) | quota=0 |
|---|---:|---:|
| electrolyser load mean (MW) | −13.440 | −2.588 |
| pipeline exchange mean (MW) | −12.621 | **−20.000 (exact match to actual)** |

Confirms §2.5: most of the electrolyser/pipeline mismatch is the single-day-scoped quota
forcing activity the year-scoped real solve didn't need *on this particular day*. A
residual **20.0 vs. 21.76 MW** downstream-demand baseline offset remains unexplained by
the quota and looks like a config/sizing drift between this run's `RunConfig` defaults
(`economic_dispatch/config.py`, vendored locally) and whatever produced the committed
`hourly_balance_*.csv` — worth checking against the exact CLI flags used for that
original run before trusting day-level electrolyser dispatch numbers further.

### 3.3 Single-day takeaway

The proxy-priced standalone optimizer reproduces the **weather-bound and battery**
behavior of the real system well, but the single-day horizon mis-scopes the RED III
quota relative to the real (year-scoped) solve, corrupting electrolyser/pipeline
dispatch on any individual day (§3.2).

### 3.4 Full-year validation (`--start-day 1 --end-day 364`, same horizon as the ground truth)

Solving the *entire year* in one LP (113,570 variables, 52,422 constraints, HiGHS: 4.5s)
scopes `demand_conservation`/`red3_quota`/cyclic-storage to the same 8736-hour horizon
the real joint solve uses — removing the §2.5/§3.2 artifact entirely, since nothing here
is scoped differently from the ground truth anymore:

| Variable | R² | MAE | RMSE | actual mean | optimized mean |
|---|---:|---:|---:|---:|---:|
| H2 Producer wind (MW) | 1.000 | 0.000 | 0.000 | 2.150 | 2.150 |
| H2 Producer pv (MW) | 1.000 | 0.000 | 0.000 | 0.811 | 0.811 |
| H2 Producer battery discharge (MW) | 0.413 | 0.572 | 2.373 | 1.083 | 1.090 |
| H2 Producer battery charge (-) (MW) | 0.236 | 0.818 | 2.462 | −1.177 | −1.185 |
| H2 Producer electrolyser load (-) (MW) | **0.815** | 2.254 | 7.968 | −14.630 | −13.440 |
| H2 Producer grid exchange (MW) | **0.788** | 3.456 | 8.667 | −11.762 | −10.573 |
| H2 Producer electrolyser production (MW) | **0.815** | 1.533 | 5.418 | 9.948 | 9.139 |
| H2 Producer tank discharge (MW) | **0.881** | 0.337 | 1.927 | 4.021 | 4.043 |
| H2 Producer tank charge (-) (MW) | **0.949** | 0.162 | 1.261 | −4.021 | −4.043 |
| H2 Producer downstream demand (-) (MW) | **0.995** | 0.017 | 0.335 | −21.760 | −21.760 |
| H2 Producer pipeline exchange (MW) | 0.668 | 1.940 | 5.879 | −11.812 | −12.621 |

Confirms both open questions from §3.1–3.2 at once:

- **The RED III quota diagnosis was right**: electrolyser load R² jumps from
  undefined/wildly-off at day-scope to **0.815** at year-scope, grid exchange from −19.45
  to **0.788**, tank discharge/charge to 0.88–0.95.
- **The "20.0 vs 21.76 MW" baseline offset was never a config drift** — it was ordinary
  day-to-day demand flexing (`prod_demand` is allowed ±20% of capacity per hour, only
  conserved *on average* over the horizon). At year scope, actual and optimized downstream
  demand means match to three decimal places (−21.760 exactly), and R²=0.995.

Battery (R²≈0.24–0.41) and pipeline exchange (R²=0.668) remain the weakest links — plausibly
because battery arbitrage is highly sensitive to hour-to-hour price *fluctuations* the
proxy doesn't need to get exactly right to score well on point-price R² (see §1.6), and
pipeline exchange absorbs whatever the other seven variables didn't already explain.

### 3.5 Diagnostic: fixing battery/tank to actual doesn't explain the remaining gap

`optimize_h2_producer.solve(..., fix_storage=True)` fixes battery and H2-tank
charge/discharge to their **actual** historical values (instead of leaving them free)
and re-optimizes everything else against the same proxy price signal — isolating whether
storage-arbitrage error (§3.4's weakest fit) was bleeding into grid/pipeline
exchange and electrolyser dispatch through the balance equations.

(Implementation note: replaying real dis/ch rounded to 3 decimals in the source CSV
accumulates ~0.03 MWh of drift in the recomputed SoC over 8736 hours — enough to trip
the LP's *exact* `[0, capacity]` bound and the cyclic-closure check on many hours where
the real battery sits at a boundary. Fixed by widening the SoC bound by that same slack
and dropping the cyclic-closure re-imposition under `fix_storage` — we're replaying an
already-realized trajectory, not asking the LP to find a fresh one.)

| Variable | Free-storage R² | Storage fixed to actual R² | Δ |
|---|---:|---:|---:|
| Electrolyser load | 0.815 | 0.820 | +0.005 |
| Grid exchange | 0.788 | 0.826 | +0.038 |
| Electrolyser production | 0.815 | 0.820 | +0.005 |
| Pipeline exchange | 0.668 | 0.726 | +0.058 |
| Downstream demand | 0.995 | 0.996 | +0.001 |

Battery/tank trivially hit R²=1.000 (fixed equal to actual by construction — not a real
result). **The interesting finding is how little the rest moved.** Pinning storage to its
true behavior recovers only +0.04–0.06 R² on grid/pipeline exchange and essentially
nothing on the electrolyser — storage-arbitrage error was not the dominant source of the
remaining gap. The likelier explanation: the proxy's point-prediction accuracy (R²
0.977–1.000, §1.6) isn't the same thing as being a good *decision* signal — a price
series can score very well on point accuracy while still occasionally sitting on the
wrong side of the electrolyser's on/off economic threshold, and dispatch decisions are
far more sensitive to that than an R² metric penalizes. This is worth keeping in mind
generally for §2.3's price-taker design: accurate price *level* prediction does not
guarantee accurate price-driven *decisions*.

### 3.6 Overall takeaway

At the horizon it was actually built for, the proxy-priced standalone optimizer
reproduces the real Hydrogen Producer's electrolyser, tank, and downstream-demand
behavior well (R² 0.81–0.995), wind/PV exactly (R²=1.000, by construction — availability-bound,
not a real decision), and is noticeably weaker on battery arbitrage and net pipeline
exchange. The single-day version (§3.1–3.3) is only reliable for weather-bound and
demand-conservation variables — its electrolyser/quota-linked numbers should not be
trusted at day granularity. §3.5's diagnostic narrows down *why* the remaining
0.17–0.33 of R² is missing on electrolyser/grid/pipeline: not storage-arbitrage error
bleeding through the balance equations (fixing storage to actual barely moves those
numbers), but more likely the proxy's price *level* accuracy not translating into
matching *decisions* at the electrolyser's on/off economic margin.
