from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from scipy.optimize import curve_fit

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. PARAMETERS ────────────────────────────────────────────────────────────
OMEGA_R = 2 * np.pi * 1.0                  # rad / unit-time
T_MAX   = 5.0                              # 5 full periods
NPTS    = 500

times = np.linspace(0, T_MAX, NPTS)

# ── 2. HAMILTONIAN AND INITIAL STATE ─────────────────────────────────────────
H = 0.5 * OMEGA_R * qt.sigmax()

# Start in |0>, the ground state
psi0 = qt.basis(2, 0)

# ── 3. SOLVE THE SCHRODINGER EQUATION ────────────────────────────────────────
result = qt.sesolve(H, psi0, times, e_ops=[qt.sigmaz()])
sz_expect = result.expect[0]
p1 = (1 - sz_expect) / 2

# ── 4. FIT TO sin^2(omega * t / 2) ───────────────────────────────────────────
def rabi(t, A, omega, phi, C):
    return A * (1 - np.cos(omega * t + phi)) / 2 + C

popt, pcov = curve_fit(
    rabi, times, p1,
    p0=[1.0, OMEGA_R, 0.0, 0.0],
    bounds=([0.5, 0.1, -np.pi, -0.1], [1.5, 100, np.pi, 0.1]),
    maxfev=10000,
)
A_fit, omega_fit, phi_fit, C_fit = popt

print("="*60)
print(f"Input  Omega_R / (2*pi) = {OMEGA_R / (2 * np.pi):.6f}  (frequency)")
print(f"Fitted omega   / (2*pi) = {omega_fit / (2 * np.pi):.6f}")
print(f"Difference              = {abs(omega_fit - OMEGA_R) / OMEGA_R * 100:.4f}%")
print("="*60)

# ── 5. PLOT ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(times, p1, color="steelblue", linewidth=2, label="qt.sesolve result")
ax.plot(times, rabi(times, *popt), color="crimson", linewidth=1.2,
        linestyle="--", label=f"Fit: f = {omega_fit / (2 * np.pi):.4f}")
ax.axhline(0, color="gray", alpha=0.3, linewidth=0.6)
ax.axhline(1, color="gray", alpha=0.3, linewidth=0.6)
ax.set_xlabel("Time (unit periods)", fontsize=12)
ax.set_ylabel("P(|1>)", fontsize=12)
ax.set_title(
    f"Driven Rabi oscillation  (Omega_R / 2*pi = "
    f"{OMEGA_R / (2 * np.pi):.2f})", fontsize=13
)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()

save_path = OUTPUT_DIR / "driven_rabi.png"
plt.savefig(save_path, dpi=150)
plt.show()
print(f"Saved: {save_path}")

# ── 6. Bloch sphere trajectory ────────────────────────────────────────
trajectory = result_states.states

# Compute Bloch vector at every step
sx_traj = np.array([qt.expect(qt.sigmax(), s) for s in trajectory])
sy_traj = np.array([qt.expect(qt.sigmay(), s) for s in trajectory])
sz_traj = np.array([qt.expect(qt.sigmaz(), s) for s in trajectory])

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

# Bloch sphere wireframe (translucent)
u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
sphere_x = np.cos(u) * np.sin(v)
sphere_y = np.sin(u) * np.sin(v)
sphere_z = np.cos(v)
ax.plot_wireframe(sphere_x, sphere_y, sphere_z,
                  color="gray", alpha=0.18, linewidth=0.4)

# Cardinal axes through the sphere
ax.plot([-1, 1], [0, 0], [0, 0], color="black", linewidth=0.6, alpha=0.4)
ax.plot([0, 0], [-1, 1], [0, 0], color="black", linewidth=0.6, alpha=0.4)
ax.plot([0, 0], [0, 0], [-1, 1], color="black", linewidth=0.6, alpha=0.4)

# Pole / equator labels
ax.text(0,    0,    1.18, r"$|0\rangle$",   fontsize=11, ha="center")
ax.text(0,    0,   -1.22, r"$|1\rangle$",   fontsize=11, ha="center")
ax.text(1.20, 0,    0,    r"$|+\rangle$",   fontsize=11)
ax.text(0,    1.20, 0,    r"$|+i\rangle$",  fontsize=11)

# Trajectory line
ax.plot(sx_traj, sy_traj, sz_traj, color="crimson", linewidth=1.8,
        label="trajectory")

# Starting state marker (|0>, at the north pole)
ax.scatter([sx_traj[0]], [sy_traj[0]], [sz_traj[0]],
           color="darkblue", s=60, zorder=10, label="start")

ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_zlim(-1.3, 1.3)
ax.set_box_aspect((1, 1, 1))
ax.set_axis_off()
ax.set_title(f"Bloch trajectory under X drive  ({T_MAX:.0f} periods)",
             fontsize=12, pad=10)
ax.legend(loc="upper left", fontsize=10, framealpha=0.85)

save_path_bloch = OUTPUT_DIR / "driven_rabi_bloch_trajectory.png"
plt.savefig(save_path_bloch, dpi=150)
plt.show()
print(f"Saved: {save_path_bloch}")
