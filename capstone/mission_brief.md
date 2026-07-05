# Guardian of the Grid — Mission Brief

> **Capstone · Module 09 · TSDynamics workshop, MPI-PKS Dresden**
> Everything you learned this week, on one desk, against one clock.

---

## The situation

It's 02:14. You are the dynamical-systems analyst on call for a regional grid operator.

Generator **G4** has been misbehaving. For days its rotor frequency has been
wandering in slow, irregular swings — never quite settling, never quite blowing up.
The night-shift engineers are nervous, because **last week a sister unit tripped
into a full blackout** during a routine load increase. No fault was found. No alarm
fired. The lights just went out.

Two data logs land on your desk. Management wants three questions answered **before
the morning load ramp** — and they want the answers defensible, not vibes.

1. **Is G4's frequency wobble deterministic** — a controllable nonlinear instability
   you can engineer against — **or is it just random noise** you have to live with?
2. **How much safety margin does a synchronized generator actually have** — and what
   does "losing sync" *look like* in state space?
3. **Could the sister unit's blackout have been seen coming**, and where *exactly*
   is the point of no return?

You have the whole TSDynamics toolkit. Go find out.

---

## The physics (why a pendulum guards the grid)

A single generator tied to a large grid is described by the **swing equation** — the
single-machine-infinite-bus model, which is *mathematically a driven, damped
pendulum*:

$$\ddot{\delta} \;=\; P \;+\; A\cos(\Omega t)\;-\;\alpha\,\dot{\delta}\;-\;K\sin(\delta)$$

- $\delta$ — the **rotor angle** relative to the grid (how far the machine leans).
- $\dot{\delta}$ — the **frequency deviation** from nominal (the SCADA channel you're handed).
- $P$ — the **mechanical load setpoint** (how hard you're driving the machine).
- $K$ — the **grid coupling / synchronizing torque** (how tightly the grid holds it).
- $\alpha$ — **damping**; $A,\Omega$ — a periodic demand modulation.

"Staying synchronized" means the pendulum settles into a stable hanging angle.
"Losing sync" — a **pole slip** — is the pendulum going *over the top* and running
away: a blackout. This is the canonical complex-systems tipping problem, and it
touches every reference this library itself cites: Menck basin stability,
Halekotte–Feudel resilience, Datseris basins.

---

## The evidence: two data files

Both live in [`capstone/data/`](data/). The headers carry multi-line `#` comments,
so **load them with pandas** (`np.loadtxt` will choke):

```python
import pandas as pd
df = pd.read_csv("../capstone/data/scada_stream.csv", comment="#")
t, freq = df["t"].values, df["freq_dev"].values
```

### `scada_stream.csv` — the wobble (Act I)

A single monitored channel from G4, in the erratic regime.

| column | units | meaning |
|---|---|---|
| `t` | s | time (uniform, `dt = 0.20 s`) |
| `freq_dev` | rad/s | rotor **frequency deviation** from grid nominal |

≈ 4000 samples. One noisy-looking wiggle. Your job in Act I is to decide **what it
really is**.

### `stress_test.csv` — the blackout (Act III)

The full state log from the slow load ramp that ended in a pole slip.

| column | units | meaning |
|---|---|---|
| `t` | s | time (uniform, `dt = 0.10 s`) |
| `load_P` | p.u. | **load setpoint**, slowly raised from ≈ 0.55 upward |
| `angle` | rad | rotor angle $\delta$ |
| `freq_dev` | rad/s | frequency deviation $\dot{\delta}$ |

The operator crept the load up while demand fluctuated. Somewhere in this log, the
machine crossed a line it could not come back from, and the run ends in runaway.
**Where was the line, and could you have called it in advance?**

> The exact machine parameters that generated these logs are sealed. That's the
> point — you're going to *recover* the story from the data and the model, the way
> you would on a real console.

---

## Your three tasks

Think of it as three acts. Each ends in a one-line verdict you'd defend to
management.

### 🔬 Act I — Diagnosis: *signal or noise?* &nbsp;<sub>(Modules 04–05)</sub>

You have one channel. Interrogate it.

- **Reconstruct the hidden attractor** from the single `freq_dev` series by delay
  embedding (`optimal_delay`, `embedding_dimension`, `embed`). Plot it. Is it a
  structured object or a fuzzball?
- **Measure its dimension** (`correlation_dimension`). A *low* value is a smoking gun:
  few degrees of freedom.
- **Measure sensitivity to initial conditions** (`lyapunov_from_data`). Inspect the
  divergence curve, pick your scaling region, and read off the exponent. Positive?
- **Put it on trial** against random noise: a surrogate test
  (`surrogate_test`, `statistic="prediction_error"`, IAAFT). Does the deterministic
  signature survive when you scramble the nonlinear structure but keep the spectrum?
- **Corroborate** with recurrence quantification (`rqa`): how *deterministic* is the
  recurrence plot?

> **Deliver a verdict:** is the G4 wobble irreducible randomness, or low-dimensional
> deterministic chaos you can actually engineer against?

### 🗺️ Act II — The model & the map of safety &nbsp;<sub>(Modules 02, 06–07)</sub>

Adopt the swing-equation model (`grid_model.GridNode`) and use it to draw the map no
telemetry channel can give you.

- **Sanity-check the model** against Act I: integrate it in the modulated regime,
  confirm a positive Lyapunov exponent (`lyapunov_spectrum`), and show a
  `StroboscopicMap` section that *looks like* the attractor you reconstructed from
  real data.
- **Find the equilibria** of the autonomous machine (`fixed_points` over a `Box`):
  the **stable synchronized state** and the **unstable saddle** that guards it. Read
  their eigenvalues — is sync a spiral or a node?
- **Paint the basins** (`basins_of_attraction` over a `Grid`): colour the set of
  states that recover into synchrony versus the set that runs away into blackout.
  This image *is* the safety map.
- **Quantify the margin** — Menck **basin stability** (`basin_fractions`): what
  fraction of a fault kick still comes home? And **resilience** — how far is the
  operating point from the edge of the cliff?

> **Deliver a verdict:** how much punishment can a synchronized generator absorb
> before it desynchronizes — and what does the failure look like geometrically?

### 🚨 Act III — The warning & the point of no return &nbsp;<sub>(Module 07 + EWS)</sub>

Now the sister unit. Reconstruct the blackout from `stress_test.csv` and the model.

- **See the runaway**: plot `angle` and `load_P` against time. The collapse is the
  spike at the end.
- **Find the early-warning signals.** Detrend the angle by the quasi-static operating
  point (`delta_star(P)` from `grid_model`), then track a **rolling variance** and a
  **rolling lag-1 autocorrelation** as the load climbs. Do they rise? (This is
  *critical slowing down* — the system's memory getting longer as it approaches the
  edge.) Corroborate with `windowed_rqa`. Mark the moment of collapse.
- **Explain the mechanism** with the model: track the synchronized state's **leading
  eigenvalue** as a function of load $P$ (`leading_eigenvalue`, or `fixed_points` on
  the overdamped machine). Watch its magnitude decay toward zero — recovery time
  diverging. *That* is why the variance rose.
- **Pin the point of no return**: run a `continuation` of the synchronized attractor
  in $P$, then `tipping_points`. Where does the stable state **vanish** in a
  saddle-node fold? Map that critical load back to a wall-clock time in the stress log.

> **Deliver a final verdict:** was the blackout foreseeable? When did the warning
> start, where is the fold, and what would you tell the operator to do differently?
> (Stretch: show that stronger coupling $K$ or damping $\alpha$ pushes the fold out.)

---

## How to start

Open the scaffolded notebook:

```
notebooks/09_capstone_student.ipynb
```

It loads and plots the raw data for you, restates the story, and lays out each act
with prompts and `# TODO` markers. Every cell runs clean out of the box — you fill in
the analysis. Work top to bottom; each act ends with a markdown cell for **your**
verdict.

The shared swing-equation model lives in [`capstone/grid_model.py`](grid_model.py)
and is imported for you (`GridNode`, `NoisyGridNode`, `delta_star`, `critical_load`,
`leading_eigenvalue`). You do **not** need to reimplement the physics — you need to
*analyse* it.

> **Stuck, or want to compare?** A complete, executed, fully worked walkthrough —
> every plot, every number, and the final verdict — lives in
> **`notebooks/09_capstone_solution.ipynb`**. Try each act yourself first; the
> solution is there to check against, not to copy.

---

**The grid is counting on you. Find the line before the morning ramp finds it first.**
