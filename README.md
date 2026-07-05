<h1 align="center">⚡ TSDynamics in an Hour</h1>
<p align="center"><b>A single-notebook, ~60-minute hands-on tour of the <a href="https://github.com/El3ssar/TSDynamics">TSDynamics</a> library</b><br>
<i>Max Planck Institute for the Physics of Complex Systems · Dresden</i></p>

<p align="center">
  <a href="https://mybinder.org/v2/gh/El3ssar/tsdynamics-workshop/express?urlpath=lab/tree/workshop_express.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Launch Binder"></a>
  <a href="https://colab.research.google.com/github/El3ssar/tsdynamics-workshop/blob/express/workshop_express.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
  <img src="https://img.shields.io/badge/python-%E2%89%A53.12-blue.svg" alt="Python >=3.12">
  <img src="https://img.shields.io/badge/time-~60%20min-orange.svg" alt="~60 min">
</p>

<p align="center"><img src="assets/img/hero.png" alt="A strange attractor" width="70%"></p>

> **Just click [Launch Binder](https://mybinder.org/v2/gh/El3ssar/tsdynamics-workshop/express?urlpath=lab/tree/workshop_express.ipynb)** (or *Open in Colab*) and start — one notebook, nothing to install.

---

## What is this?

One self-contained notebook — [`workshop_express.ipynb`](workshop_express.ipynb) — that takes you through the
highlights of **TSDynamics** in about an hour. It assumes you know nonlinear dynamics but have never touched
the library. Read a cell, run it, and tweak the **▶ Try it** cells. Every step is a few lines and computes in
seconds.

By the end you will have:

1. **Integrated** flows and maps and plotted their attractors,
2. measured chaos with a **Lyapunov spectrum** and a fractal **dimension**,
3. drawn the **bifurcation diagram** of the road to chaos,
4. **built your own system** from scratch in six lines,
5. told **deterministic chaos from noise** in a measured signal (embedding + a surrogate test), and
6. **animated** an attractor to a GIF.

## Start in 30 seconds

| | How |
|---|---|
| ☁️ **Binder** | [**Launch**](https://mybinder.org/v2/gh/El3ssar/tsdynamics-workshop/express?urlpath=lab/tree/workshop_express.ipynb) — one click, nothing to install |
| ☁️ **Colab** | [**Open in Colab**](https://colab.research.google.com/github/El3ssar/tsdynamics-workshop/blob/express/workshop_express.ipynb) (Python ≥ 3.12) — the notebook installs the library itself on first run |
| 💻 **Local** | `pip install "tsdynamics[viz,interactive]==5.3.1" jupyterlab matplotlib plotly` then open the notebook |

## Want the deep dive?

This express notebook is the appetizer. The **full 9-notebook workshop** — custom DDE/SDE models,
Poincaré sections, fixed points & stability, basins of attraction, tipping points, and a power-grid
capstone — lives on the **[`main` branch](https://github.com/El3ssar/tsdynamics-workshop/tree/main)**.

## License

© 2026 Daniel Estevez, [MIT License](LICENSE). Made for the community at MPI-PKS, Dresden.
