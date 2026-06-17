import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qutip as qt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

DATA_DIR  = Path(__file__).resolve().parent.parent / "data"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. LOAD THE REAL Q0 DATA ────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "ibm_marrakesh_q0_t1.csv")
with open(DATA_DIR / "ibm_marrakesh_q0_metadata.json") as f:
    meta = json.load(f)

T1_MEASURED  = meta["T1_fit_us"]                  # 315.1 us  (real fit)
T1_PUBLISHED = meta["backend_published_t1_us"]    # 344.5 us  (IBM calibration book)
print("="*60)
print(f"Q0 measured T1   = {T1_MEASURED:.2f} us")
print(f"Q0 published T1  = {T1_PUBLISHED:.2f} us")
print(f"Published / measured ratio = {T1_PUBLISHED / T1_MEASURED:.4f}")
print("="*60)

# ── 2. BUILD LINDBLAD COMPONENTS ─────────────────────────────────────────────
sigma_minus = qt.destroy(2)
H_free      = qt.qzero(2)         # zero Hamiltonian: no drive, free evolution
psi0        = qt.basis(2, 1)      # initial state |1>, matching the experiment

# Time grid spanning the real-experiment delay range (0 to 600 us)
times = np.linspace(0, 600, 300)

# ── 3. SOLVE THE MASTER EQUATION ─────────────────────────────────────────────
def run_lindblad(T1_us):
    gamma_1 = 1.0 / T1_us                          # decay rate in 1/us
    c_op    = np.sqrt(gamma_1) * sigma_minus       # the collapse operator
    res     = qt.mesolve(H_free, psi0, times,
                         c_ops=[c_op], e_ops=[qt.sigmaz()])
    sz = res.expect[0]
    return (1 - sz) / 2                            # convert <sigma_z> to P(|1>)

p1_sim_measured  = run_lindblad(T1_MEASURED)
p1_sim_published = run_lindblad(T1_PUBLISHED)

# ── 4. RECOVER T1 FROM THE SIMULATED CURVE  (sanity check) ───────────────────
def exp_decay(t, A, T1, C):
    return A * np.exp(-t / T1) + C

popt, _ = curve_fit(
    exp_decay, times, p1_sim_measured,
    p0=[1.0, T1_MEASURED, 0.0],
    bounds=([0.5, 1, 0], [1.5, 1000, 0.1]),
)
T1_recovered = popt[1]
print()
print(f"Input T1 (Lindblad rate):  {T1_MEASURED:.4f} us")
print(f"Recovered T1 (fit to sim): {T1_recovered:.4f} us")
print(f"Self-consistency: {abs(T1_recovered - T1_MEASURED) / T1_MEASURED * 100:.4f}% off")
print("(Should be << 1% if the Lindblad solver and curve fitter agree.)")

# ── 5. COMPARE SIMULATION TO REAL DATA ───────────────────────────────────────
sim_meas_at_data = interp1d(times, p1_sim_measured)(df["delay_us"])
sim_pub_at_data  = interp1d(times, p1_sim_published)(df["delay_us"])

rms_measured  = np.sqrt(((df["p1"] - sim_meas_at_data) ** 2).mean())
rms_published = np.sqrt(((df["p1"] - sim_pub_at_data ) ** 2).mean())

print()
print(f"RMS residual (data - sim, T1={T1_MEASURED:.1f}): {rms_measured:.4f}")
print(f"RMS residual (data - sim, T1={T1_PUBLISHED:.1f}): {rms_published:.4f}")
print(f"Which T1 explains the data better? "
      f"{'measured' if rms_measured < rms_published else 'published'} "
      f"(lower RMS wins)")

# ── 5b. REFIT WITH READOUT OFFSET ────────────────────────────────────────────
# The raw RMS comparison is biased by a readout assignment error floor.
# Fitting P(t) = A*exp(-t/T1) + C to the real data recovers an unbiased T1.

popt_data, _ = curve_fit(
    exp_decay, df["delay_us"], df["p1"],
    p0=[1.0, T1_MEASURED, 0.02],
    bounds=([0.5, 50, 0.0], [1.5, 1000, 0.15]),
)
A_fit, T1_data_fit, C_fit = popt_data
print()
print(f"Direct fit to real data (with offset):")
print(f"  A  = {A_fit:.4f}")
print(f"  T1 = {T1_data_fit:.2f} us")
print(f"  C  = {C_fit:.4f}  <-- readout floor")
print(f"  Readout err from metadata: {meta['backend_published_readout_err']:.4f}")

T1_CORRECTED = popt_data[1]   # 319.70 us: offset-corrected, unbiased
p1_sim_corrected = run_lindblad(T1_CORRECTED)

# ── 6. PLOT ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
ax.axhline(C_fit, color="gray", linewidth=1.2, linestyle=":",
           label=f"Readout floor C = {C_fit:.3f} (thermal + assignment error)")

ax.plot(times, p1_sim_corrected, color="darkorange", linewidth=2.4,
        label=f"qt.mesolve, T1 = {T1_CORRECTED:.1f} us (offset-corrected fit)")

# Real data points
ax.scatter(df["delay_us"], df["p1"],
           color="black", s=50, zorder=10, edgecolor="white", linewidth=0.8,
           label="ibm_marrakesh Q0 real data (15 delays, 1024 shots each)")

# Simulation with measured T1 (should match the data well)
ax.plot(times, p1_sim_measured, color="crimson", linewidth=2.2,
        label=f"qt.mesolve, T1 = {T1_MEASURED:.1f} us (measured fit)")

# Simulation with published T1 (slower decay, visibly above the data)
ax.plot(times, p1_sim_published, color="steelblue", linewidth=2.0,
        linestyle="--",
        label=f"qt.mesolve, T1 = {T1_PUBLISHED:.1f} us (IBM published)")

ax.set_xlabel("Delay (us)", fontsize=12)
ax.set_ylabel("P(|1>)", fontsize=12)
ax.set_title(
    "Lindblad T1 simulation vs real ibm_marrakesh Q0\n"
    "Offset-corrected T1 = 319.7 µs is the unbiased decay rate",
    fontsize=12,
)
ax.set_ylim(-0.02, 1.05)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
plt.tight_layout()

save_path = PLOTS_DIR / "t1_lindblad_simulator_vs_real.png"
plt.savefig(save_path, dpi=150)
plt.show()
print()
print(f"Saved: {save_path}")
