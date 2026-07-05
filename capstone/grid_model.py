"""Shared dynamical model for the *Guardian of the Grid* capstone.

The physical picture: a single synchronous generator tied to a large ("infinite")
power grid.  Its rotor phase angle ``delta`` relative to the grid obeys the
**swing equation** — mathematically a *damped, driven pendulum* / the
single-machine-infinite-bus (SMIB) model:

    delta'  = omega
    omega'  = P + A*cos(Omega*t) - alpha*omega - K*sin(delta)

* ``P``     — net mechanical power fed to the rotor (the *load setpoint*; our control knob).
* ``K``     — maximum power the transmission line can carry (grid coupling strength).
* ``alpha`` — damping (mechanical + electrical losses / droop control).
* ``A``,``Omega`` — amplitude/frequency of a periodic load fluctuation (renewables, demand).

Physics the capstone leans on (all verified against tsdynamics 5.3.1):

* **Synchronized operation** = a stable fixed point  delta* = asin(P/K), omega = 0.
* **Saddle-node (fold) at P = K**: the stable node and the unstable saddle collide
  and annihilate — beyond it the machine cannot hold synchrony (*pole slip / blackout*).
* **Basin stability** (Menck et al. 2013): only part of state space returns to sync;
  a large enough fault throws the machine onto the *running* (desynchronized) solution.
* **Critical slowing down** near the fold: the restoring stiffness K*cos(delta*) -> 0,
  so recovery time diverges — the basis of early-warning signals.
* **Chaos** under strong periodic load modulation (A>0): the phase angle rattles
  chaotically (positive Lyapunov exponent) while still bounded.

This module is imported by the capstone *solution* notebook.  Participants model the
data with these same classes.
"""
from __future__ import annotations

import numpy as np
import symengine as se
import tsdynamics as ts

K_DEFAULT = 1.0


class GridNode(ts.ContinuousSystem):
    """Deterministic swing equation (driven damped pendulum), written as a 2-D flow.

    With ``A=0`` it is autonomous (used for fixed points, basins, continuation).
    With ``A>0`` the explicit-time forcing drives period-doubling and chaos.
    """

    params = {"P": 0.4, "alpha": 0.2, "K": K_DEFAULT, "A": 0.0, "Omega": 1.0}
    dim = 2
    variables = ("delta", "omega")
    default_ic = [0.0, 0.0]
    reference = "Swing equation / single-machine-infinite-bus; cf. Menck et al., Nat. Phys. 9, 89 (2013)"

    @staticmethod
    def _equations(y, t, *, P, alpha, K, A, Omega):
        delta, omega = y(0), y(1)
        return [
            omega,
            P + A * se.cos(Omega * t) - alpha * omega - K * se.sin(delta),
        ]


class NoisyGridNode(ts.StochasticSystem):
    """Swing equation with stochastic load/frequency fluctuations (diagonal Ito).

    Only the frequency equation is noise-driven (``sigma`` on ``omega``), the
    natural place for demand/renewable jitter.  Used for the early-warning-signal
    (critical slowing down) part of the capstone.
    """

    params = {"P": 0.6, "alpha": 0.7, "K": K_DEFAULT, "sigma": 0.05}
    dim = 2
    variables = ("delta", "omega")
    default_ic = [0.6, 0.0]

    @staticmethod
    def _drift(y, t, *, P, alpha, K, sigma):
        delta, omega = y(0), y(1)
        return [omega, P - alpha * omega - K * se.sin(delta)]

    @staticmethod
    def _diffusion(y, t, *, P, alpha, K, sigma):
        # no noise on the angle, constant noise on the frequency
        return [se.Integer(0) * y(0), sigma]


# ---------------------------------------------------------------------------
# Analytic helpers (the "textbook answers" the capstone can check against)
# ---------------------------------------------------------------------------
def delta_star(P: float, K: float = K_DEFAULT) -> float:
    """Stable synchronized operating angle delta* = asin(P/K) (nan past the fold)."""
    if abs(P) > K:
        return float("nan")
    return float(np.arcsin(P / K))


def critical_load(K: float = K_DEFAULT) -> float:
    """The saddle-node (fold) load: synchrony is lost for P > P_c = K."""
    return float(K)


def restoring_stiffness(P: float, K: float = K_DEFAULT) -> float:
    """Linear restoring stiffness kappa = K*cos(delta*) at the operating point.

    Vanishes at the fold -> the well flattens -> critical slowing down.
    """
    ds = delta_star(P, K)
    return float(K * np.cos(ds)) if np.isfinite(ds) else 0.0


def leading_eigenvalue(P: float, alpha: float, K: float = K_DEFAULT) -> complex:
    """Slowest eigenvalue of the synchronized node (its real part -> 0 at the fold)."""
    kappa = restoring_stiffness(P, K)
    disc = alpha * alpha - 4.0 * kappa
    root = np.sqrt(complex(disc))
    l1 = (-alpha + root) / 2.0
    l2 = (-alpha - root) / 2.0
    return l1 if l1.real >= l2.real else l2
