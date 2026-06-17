import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qutip as qt

DATA_DIR  = Path(__file__).resolve().parent.parent / "data"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. LOAD THE REAL Q0 DATA ────────────────────────────────────────────────
df_t2 = pd.read_csv(DATA_DIR / "ibm_marrakesh_q0_t2.csv")
with open(DATA_DIR / "ibm_marrakesh_q0_metadata.json") as f:
    meta = json.load(f)

T1_MEASURED  = meta["T1_fit_us"]                  # 315.1 us (real fit)
T1_PUBLISHED = meta["backend_published_t1_us"]    # 344.5 us (IBM calibration book)
T1_us   = 319.70                            # offset-corrected T1 from t1_lindblad.py
T2_us   = meta["T2_fit_us"]                 # 46.1 us
delta_f = meta["T2_oscillation_freq_mhz"]   # 0.05186 MHz

print("="*60)
print(f"Q0 measured T1   = {T1_MEASURED:.2f} us")
print(f"Q0 published T1  = {T1_PUBLISHED:.2f} us")
print(f"Published / measured ratio = {T1_PUBLISHED / T1_MEASURED:.4f}")
print("="*60)

Tphi_us = 1.0 / (1.0/T2_us - 1.0/(2.0*T1_us))
print(f"T2      = {T2_us:.2f} us")
print(f"T1      = {T1_us:.2f} us")
print(f"T_phi   = {Tphi_us:.2f} us  (derived: 1/T2 - 1/2T1)")

gamma_1   = 1.0 / T1_us
gamma_phi = 1.0 / (2.0 * Tphi_us)   # factor of 2 is exact, not approximate

c_ops = [
    np.sqrt(gamma_1)   * qt.destroy(2),   # amplitude damping (same form as t1_lindblad.py)
    np.sqrt(gamma_phi) * qt.sigmaz(),     # pure dephasing
]
psi0_ramsey = (qt.basis(2,0) + qt.basis(2,1)).unit()

e_ops = [qt.sigmax()]
delta_f_MHz = meta["T2_oscillation_freq_mhz"]   # 0.05186 MHz
H_ramsey = (np.pi * delta_f_MHz) * qt.sigmaz()  # units: rad/µs

times = np.linspace(0, 100, 500)

res = qt.mesolve(H_ramsey, psi0_ramsey, times, c_ops=c_ops, e_ops=e_ops)
sx  = res.expect[0]   # <sigma_x>(t)
