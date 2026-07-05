"""Generate the *Guardian of the Grid* capstone datasets (deterministic, seeded).

Run:  python capstone/generate_data.py
Outputs (committed so participants need no re-run):
  data/scada_stream.csv   — Act I: a single phase-angle sensor channel, monitored
                            regime.  Is it deterministic chaos or just noise?
  data/stress_test.csv    — Act III: a slow "load creep" experiment under load
                            noise that ends in a blackout.  Find the warning.
  data/_truth.json        — the ground-truth generating parameters (instructors only).

Nothing here is secret; it just keeps the notebooks tidy.  The "world" (stress test)
is generated with a transparent Euler–Maruyama loop — participants *analyse* it with
tsdynamics.  The monitored SCADA stream is a genuine tsdynamics trajectory.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

import tsdynamics as ts
from grid_model import GridNode, delta_star, critical_load

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Act I — the monitored SCADA stream: a bounded but CHAOTIC phase angle.
# Strong periodic load modulation (A) drives the driven pendulum into chaos.
# ---------------------------------------------------------------------------
SCADA = dict(P=0.20, alpha=0.20, K=1.0, A=0.90, Omega=0.60)


SCADA_DT = 0.20  # sensor cadence (s)


def make_scada(seed: int = 7) -> None:
    node = GridNode().with_params(**SCADA)
    # long run, drop transient, subsample to a realistic sensor cadence.
    # We record the FREQUENCY DEVIATION omega (what grid SCADA actually monitors) —
    # it is bounded and chaotic, whereas the raw rotor angle keeps winding up.
    traj = node.integrate(final_time=1000.0, dt=0.01, ic=[0.1, 0.0], method="rk45")
    t = traj.t
    omega = traj.y[:, 1]
    keep = t > 200.0                                  # drop transient
    t, omega = t[keep] - 200.0, omega[keep]
    step = int(round(SCADA_DT / 0.01))                # 0.01 -> 0.20 s cadence
    t, omega = t[::step], omega[::step]
    rng = np.random.default_rng(seed)
    freq = omega + 0.015 * np.std(omega) * rng.standard_normal(omega.shape)  # 1.5% sensor noise
    header = (
        f"# SCADA frequency-deviation telemetry, generator G4.  dt = {SCADA_DT} s.\n"
        "# Column: t (s), freq_dev (rad/s, rotor speed deviation from grid nominal).\n"
        "# Question: is this deterministic (low-dimensional chaos) or stochastic noise?\n"
    )
    with open(DATA / "scada_stream.csv", "w") as f:
        f.write(header)
        f.write("t,freq_dev\n")
        for ti, ai in zip(t, freq):
            f.write(f"{ti:.4f},{ai:.6f}\n")
    # sanity checks that Act I will succeed
    from tsdynamics.analysis.embedding import optimal_delay, embed
    tau = int(optimal_delay(freq).value)
    Y = np.asarray(embed(freq, dimension=4, delay=tau))
    d2 = float(ts.correlation_dimension(Y, radii=np.logspace(-1.0, 0.8, 20)).value)
    lam = float(ts.lyapunov_from_data(freq, dt=SCADA_DT, dimension=4, delay=tau,
                                      k_max=40, fit=(2, 10)).value)
    st = ts.surrogate_test(freq[:4000], statistic="prediction_error", method="iaaft", n=39, seed=1)
    print(f"[scada] n={len(freq)} tau={tau} D2={d2:.2f} lambda_fit={lam:+.3f} "
          f"surrogate_rejected={st.to_dict()['rejected']} (z={st.to_dict()['z_score']:+.1f})")


# ---------------------------------------------------------------------------
# Act III — the stress test: a slow load creep under load noise -> blackout.
# Overdamped so the leading eigenvalue -> 0 monotonically (clean critical slowing).
# Transparent Euler–Maruyama; P ramps quasi-statically from P0 to just past the fold.
# ---------------------------------------------------------------------------
STRESS = dict(alpha=0.75, K=1.0, sigma=0.045)
P0, P1 = 0.55, 1.05
T_TOTAL = 3000.0
DT = 0.01
SENSOR_STEP = 10  # 0.01 -> 0.10 s cadence


def make_stress(seed: int = 20) -> None:
    rng = np.random.default_rng(seed)
    alpha, K, sigma = STRESS["alpha"], STRESS["K"], STRESS["sigma"]
    n = int(T_TOTAL / DT)
    P_of_t = P0 + (P1 - P0) * np.arange(n) / n
    delta = np.empty(n)
    omega = np.empty(n)
    delta[0], omega[0] = delta_star(P0, K), 0.0
    sq = np.sqrt(DT)
    collapse_i = n - 1
    for i in range(1, n):
        P = P_of_t[i - 1]
        d, w = delta[i - 1], omega[i - 1]
        dw = P - alpha * w - K * np.sin(d)
        delta[i] = d + w * DT
        omega[i] = w + dw * DT + sigma * sq * rng.standard_normal()
        # pole slip: angle has run away past the barrier -> blackout
        if delta[i] - delta_star(min(P, 0.999), K) > np.pi:
            collapse_i = i
            break
    delta, omega, P_of_t = delta[: collapse_i + 1], omega[: collapse_i + 1], P_of_t[: collapse_i + 1]
    t = np.arange(len(delta)) * DT
    # sensor cadence + small measurement noise on the angle
    t, delta, omega, P_of_t = (a[::SENSOR_STEP] for a in (t, delta, omega, P_of_t))
    delta_meas = delta + 0.01 * rng.standard_normal(delta.shape)
    t_collapse = collapse_i * DT
    P_collapse = P0 + (P1 - P0) * collapse_i / n
    header = (
        "# Grid stress-test log, generator G4.  dt = 0.10 s.\n"
        "# The operator slowly raised the load setpoint P while demand fluctuated.\n"
        "# Columns: t (s), load_P (p.u.), angle (rad), freq_dev (rad/s).\n"
        f"# The run ends at a blackout (pole slip).  Fold load P_c = K = {critical_load(K):.3f}.\n"
    )
    with open(DATA / "stress_test.csv", "w") as f:
        f.write(header)
        f.write("t,load_P,angle,freq_dev\n")
        for row in zip(t, P_of_t, delta_meas, omega):
            f.write("%.3f,%.5f,%.6f,%.6f\n" % row)

    # verify the early-warning signal: rolling variance of detrended angle rises
    d = delta_meas.copy()
    # detrend by the quasi-static operating point
    trend = np.array([delta_star(min(P, 0.999), K) for P in P_of_t])
    resid = d - trend
    W = 400
    var = np.array([resid[max(0, i - W): i].var() if i >= 2 else 0.0 for i in range(len(resid))])
    early, late = var[len(var) // 5], var[int(len(var) * 0.85)]
    print(f"[stress] n={len(t)} collapse@t={t_collapse:.0f}s P={P_collapse:.3f}  "
          f"rolling-var early={early:.4f} -> late={late:.4f} (rise x{late/max(early,1e-9):.1f})")
    return dict(t_collapse=float(t_collapse), P_collapse=float(P_collapse))


def main() -> None:
    make_scada()
    stress_meta = make_stress()
    truth = {
        "scada": SCADA | {"note": "bounded chaotic driven pendulum; lambda1 ~ 0.13"},
        "stress": STRESS | {"P0": P0, "P1": P1, "T_total": T_TOTAL, **stress_meta,
                            "fold_load_Pc": critical_load(STRESS["K"])},
    }
    (DATA / "_truth.json").write_text(json.dumps(truth, indent=2))
    print("wrote", list(p.name for p in DATA.iterdir()))


if __name__ == "__main__":
    main()
