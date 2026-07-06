<h1 align="center">⚡ TSDynamics — the Grand Tour</h1>
<p align="center"><b>A single-notebook, hands-on tour of the <a href="https://github.com/El3ssar/TSDynamics">TSDynamics</a> library</b><br>
<i>Max Planck Institute for the Physics of Complex Systems · Dresden</i></p>

<p align="center">
  <a href="https://mybinder.org/v2/gh/El3ssar/tsdynamics-workshop/express?urlpath=lab/tree/workshop_express.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Launch Binder"></a>
  <a href="https://colab.research.google.com/github/El3ssar/tsdynamics-workshop/blob/express/workshop_express.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
  <img src="https://img.shields.io/badge/python-%E2%89%A53.12-blue.svg" alt="Python >=3.12">
  <img src="https://img.shields.io/badge/time-~75%20min-orange.svg" alt="~75 min">
</p>

<p align="center"><img src="assets/img/hero.png" alt="A strange attractor" width="70%"></p>

> **Just click [Launch Binder](https://mybinder.org/v2/gh/El3ssar/tsdynamics-workshop/express?urlpath=lab/tree/workshop_express.ipynb)** (or *Open in Colab*) and start — one notebook, nothing to install.

---

## What is this?

One self-contained notebook — [`workshop_express.ipynb`](workshop_express.ipynb) — that tours the
**TSDynamics** library the way it's meant to be used: its own `viz` and analysis layers throughout, not raw
matplotlib. It assumes you know nonlinear dynamics but have never touched the library. Read a cell, run it,
tweak the **▶ Try it** cells.

Across seven parts you will:

1. **integrate** flows and maps, then **compose, theme, and save** figures with the `viz` layer,
2. sweep a **bifurcation diagram** and let the library locate the period-doublings for you,
3. reduce a flow to a **2-D Poincaré section** (Hénon–Heiles) showing *order and chaos coexisting*,
4. **quantify** chaos — Lyapunov spectrum, Kaplan–Yorke dimension, and the 0–1 test,
5. **build your own system** from scratch in six lines,
6. watch a **Gray–Scott reaction–diffusion field** self-organize into a labyrinth *on camera*, and
7. **solve a real problem** — *is this measured signal predictable?* — end to end: embedding, correlation
   dimension, a Lyapunov predictability horizon, and a surrogate test that separates chaos from noise.

## Start in 30 seconds

| | How |
|---|---|
| ☁️ **Binder** | [**Launch**](https://mybinder.org/v2/gh/El3ssar/tsdynamics-workshop/express?urlpath=lab/tree/workshop_express.ipynb) — one click, nothing to install |
| ☁️ **Colab** | [**Open in Colab**](https://colab.research.google.com/github/El3ssar/tsdynamics-workshop/blob/express/workshop_express.ipynb) (Python ≥ 3.12) — the notebook installs the library on first run |
| 💻 **Local** | `pip install "tsdynamics[viz,interactive]==5.3.1" jupyterlab matplotlib plotly imageio-ffmpeg` then open the notebook |

## Want the deep dive?

This is the single-sitting tour. The **full multi-notebook workshop** — custom DDE/SDE models, stroboscopic
maps, fixed points & stability, basins of attraction, tipping points, and a power-grid capstone — lives on the
**[`main` branch](https://github.com/El3ssar/tsdynamics-workshop/tree/main)**.

## License

© 2026 Daniel Estevez, [MIT License](LICENSE). Made for the community at MPI-PKS, Dresden.
