import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from scipy.integrate import quad

DATA_DIR  = Path(__file__).resolve().parent.parent / "data"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- 1. PARAMETERS ---------------------------------------------------------
ALPHA = -0.3 * 2 * np.pi          # anharmonicity, rad/ns  (= -2*pi*0.3 GHz)
TG    = 6.0                       # total gate time, ns
SIGMA = 1.5                       # Gaussian sigma, ns (truncated at +/-2 sigma)
NT    = 601                       # time samples on [0, TG]
times = np.linspace(0, TG, NT)

with open(DATA_DIR / "ibm_marrakesh_q0_metadata.json") as f:
    meta = json.load(f)
T1_us    = 319.70                          # offset-corrected fit from t1.py
T2_us    = meta["T2_fit_us"]               # 46.1 us
Tphi_us  = 1.0 / (1.0 / T2_us - 1.0 / (2.0 * T1_us))   # ~ 50 us

T1_ns   = T1_us   * 1e3
Tphi_ns = Tphi_us * 1e3

print("=" * 60)
print(f"alpha / 2*pi  = {ALPHA / (2*np.pi):.3f} GHz")
print(f"gate time TG  = {TG:.1f} ns,   sigma = {SIGMA:.1f} ns")
print(f"T1            = {T1_us:.2f} us  ({T1_ns:.0f} ns)")
print(f"T_phi         = {Tphi_us:.2f} us  ({Tphi_ns:.0f} ns)")
print(f"Decoherence over the {TG:.0f} ns gate:")
print(f"  exp(-TG/T1)    = {np.exp(-TG/T1_ns):.6f}")
print(f"  exp(-TG/T_phi) = {np.exp(-TG/Tphi_ns):.6f}")
print("(Both close to 1 -- the gate is fast compared to coherence times.)")
print("=" * 60)

# ---- 2. OPERATORS ON THE 3-LEVEL HILBERT SPACE -----------------------------
a       = qt.destroy(3)
adag    = a.dag()
n_op    = adag * a                          # number operator
Hx      = 0.5 * (a + adag)                  # I-quadrature drive operator
Hy      = 0.5j * (adag - a)                 # Q-quadrature drive operator (+sigma_y/2)

H0      = ALPHA * qt.fock_dm(3, 2)          # alpha * |2><2|, rotating frame
psi0    = qt.basis(3, 0)                    # start in |0>

P0      = qt.fock_dm(3, 0)
P1      = qt.fock_dm(3, 1)
P2      = qt.fock_dm(3, 2)
e_ops   = [P0, P1, P2]

# ---- 3. PULSE ENVELOPES ----------------------------------------------------
T0 = TG / 2.0                               # pulse center

def gauss(t, t0=T0, s=SIGMA):
    return np.exp(-((t - t0) ** 2) / (2.0 * s * s))

# Calibrate AMP so that integral(Omega_x dt) = pi over [0, TG].
# Under H = (Omega/2)(a + a^dag), this implements a pi rotation on |0>-|1>.
area, _ = quad(gauss, 0.0, TG)
AMP = np.pi / area
print(f"Pulse calibration: integral(Omega_x dt) = {AMP * area:.4f} rad  (target pi)")

def omega_x(t, args=None):
    return AMP * gauss(t)

def domega_x_dt(t, args=None):
    """Analytic derivative of the I-quadrature envelope."""
    s = SIGMA
    return -AMP * ((t - T0) / (s * s)) * gauss(t)

def omega_y_drag(t, args=None):
    return -domega_x_dt(t) / (2.0 * ALPHA)

def omega_y_zero(t, args=None):
    return 0.0

# ---- 4. SIMULATION DRIVER --------------------------------------------------
def simulate(use_drag, with_collapse):
    H = [H0, [Hx, omega_x], [Hy, omega_y_drag if use_drag else omega_y_zero]]
    c_ops = []
    if with_collapse:
        c_ops = [
            np.sqrt(1.0 / T1_ns)   * a,                 # amplitude damping
            np.sqrt(2.0 / Tphi_ns) * n_op,              # pure dephasing
        ]
    return qt.mesolve(H, psi0, times, c_ops=c_ops, e_ops=e_ops)

res_sq_u   = simulate(use_drag=False, with_collapse=False)
res_drag_u = simulate(use_drag=True,  with_collapse=False)
res_sq_o   = simulate(use_drag=False, with_collapse=True)
res_drag_o = simulate(use_drag=True,  with_collapse=True)

# ---- 5. REPORT FINAL POPULATIONS -------------------------------------------
def final(res, level):
    return float(res.expect[level][-1])

def report(label, res):
    p0, p1, p2 = final(res, 0), final(res, 1), final(res, 2)
    print(f"  {label:38s}  P0={p0:.5f}  P1={p1:.5f}  P2={p2:.6f}")

print()
print("Final populations after the gate")
print("-" * 60)
print("Unitary (no T1 / T_phi):")
report("square pulse", res_sq_u)
report("DRAG-corrected pulse", res_drag_u)
leak_sq, leak_drag = final(res_sq_u, 2), final(res_drag_u, 2)
factor = leak_sq / max(leak_drag, 1e-14)
print(f"  Leakage suppression (square / DRAG): {factor:8.1f} x")
print()
print("Open system (T1 + T_phi on the 3-level space):")
report("square pulse", res_sq_o)
report("DRAG-corrected pulse", res_drag_o)
print("=" * 60)

# ---- 6. PLOT 1 -- UNITARY COMPARISON ---------------------------------------
def plot_pair(res_left, res_right, suptitle, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, res, panel_title in [
        (axes[0], res_left,  "Square Gaussian X gate"),
        (axes[1], res_right, "DRAG-corrected X gate"),
    ]:
        ax.plot(times, res.expect[0], color="steelblue", linewidth=2.0,
                label=r"$P(|0\rangle)$")
        ax.plot(times, res.expect[1], color="crimson", linewidth=2.0,
                label=r"$P(|1\rangle)$")
        ax.plot(times, res.expect[2], color="darkorange", linewidth=2.0,
                linestyle="--", label=r"$P(|2\rangle)$ (leakage)")
        ax.set_title(panel_title, fontsize=11)
        ax.set_xlabel("Time (ns)")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="center right", fontsize=9, framealpha=0.92)
    axes[0].set_ylabel("Population")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")

plot_pair(
    res_sq_u, res_drag_u,
    suptitle=(
        f"Transmon X gate, alpha/2pi = {ALPHA/(2*np.pi):.2f} GHz, "
        f"TG = {TG:.0f} ns, sigma = {SIGMA:.1f} ns  (unitary)"
    ),
    save_path=PLOTS_DIR / "transmon_drag_comparison.png",
)

# ---- 7. PLOT 2 -- OPEN-SYSTEM COMPARISON (T1 + T_phi) ----------------------
plot_pair(
    res_sq_o, res_drag_o,
    suptitle=(
        f"Transmon X gate with collapse operators, "
        f"T1 = {T1_us:.1f} us, T_phi = {Tphi_us:.1f} us  (ibm_marrakesh Q0)"
    ),
    save_path=PLOTS_DIR / "transmon_drag_open_system.png",
)

plt.show()
