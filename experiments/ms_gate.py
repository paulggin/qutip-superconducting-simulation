# -*- coding: utf-8 -*-
"""
ms_gate.py
Molmer-Sorensen two-qubit gate on two trapped ions sharing one motional mode,
simulated in QuTiP. This is the trapped-ion counterpart to transmon_drag.py:
where that script handles single-qubit gate control on a superconducting
transmon, this one handles the entangling gate that IonQ and Quantinuum
hardware actually run.

The physics:
    Two ion qubits couple to a single shared motional mode (a harmonic
    oscillator, truncated to N_FOCK Fock states). A bichromatic laser field,
    detuned by delta from the first motional sidebands, drives a
    state-dependent force. In the Lamb-Dicke regime and after the RWA the
    interaction-picture Hamiltonian is

        H(t) = (eta*Omega/2) (sx_1 + sx_2) (a e^{+i delta t} + a^dag e^{-i delta t})

    where eta is the Lamb-Dicke parameter and Omega the carrier Rabi rate.
    The force pushes the motional state around a loop in phase space. After
    one full loop the loop closes (motion disentangles from the qubits) and a
    geometric phase proportional to (sx_1 + sx_2)^2 is left behind, which is
    the sigma_x (x) sigma_x entangling interaction.

    Closed-loop timing: the loop closes at tau_g = 2*pi/delta.
    Maximal entanglement (a Bell state from |00>) needs geometric phase
    chi = pi/8 on the sx*sx term, which gives the analytic calibration
        Omega = delta / (2 eta).
    The script confirms this by scanning Omega numerically and locking in the
    value that maximizes Bell-state fidelity, mirroring the numerical
    beta-sweep used to fix the DRAG coefficient in transmon_drag.py.

    Ideal output state from |00>|n=0>:
        U|00> = (|00> - i|11>)/sqrt(2)   (motion returned to ground)

    Open-system channels:
        motional heating  c = sqrt(ndot) a^dag      (ndot quanta/s)
        qubit dephasing   c = sqrt(gamma_phi/2) sz  (per ion)

The point of this script:
    (a) Calibrate the gate and report the Bell-state fidelity at the closed-loop
        point, confirming Omega = delta/(2 eta) numerically.
    (b) Sweep the detuning delta around the calibrated point to show how a
        miscalibrated sideband detuning (and therefore a loop that fails to
        close) degrades fidelity.
    (c) Sweep the motional heating rate ndot to show the fidelity ceiling set
        by trap heating, the dominant hardware-side error for this gate.

Outputs:
    outputs/plots/ms_gate_calibration.png       (fidelity vs Omega)
    outputs/plots/ms_gate_error_budget.png      (fidelity vs detuning, vs heating)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. SYSTEM PARAMETERS ────────────────────────────────────────────────────
N_FOCK = 12                       # motional Fock-space truncation
eta    = 0.1                      # Lamb-Dicke parameter
nu     = 2*np.pi*1.0e6            # motional mode frequency (rad/s), 1 MHz
delta  = 2*np.pi*25.0e3           # bichromatic detuning from sideband, 25 kHz
tau_g  = 2*np.pi/delta            # closed-loop gate time (one phase-space loop)
Omega_analytic = delta/(2*eta)    # analytic maximally-entangling Rabi rate

# ── 2. OPERATORS ON qubit1 (x) qubit2 (x) motion ────────────────────────────
sx1 = qt.tensor(qt.sigmax(), qt.qeye(2), qt.qeye(N_FOCK))
sx2 = qt.tensor(qt.qeye(2), qt.sigmax(), qt.qeye(N_FOCK))
sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2), qt.qeye(N_FOCK))
sz2 = qt.tensor(qt.qeye(2), qt.sigmaz(), qt.qeye(N_FOCK))
a   = qt.tensor(qt.qeye(2), qt.qeye(2), qt.destroy(N_FOCK))
Sx  = sx1 + sx2

# motion starts in the ground state (perfect sideband cooling), qubits in |00>
psi0 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(N_FOCK, 0))

# ideal Bell target on the qubit subspace
bell = (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) +
        -1j*qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()

def coeff_p(t, args): return np.exp(1j*args['delta']*t)
def coeff_m(t, args): return np.exp(-1j*args['delta']*t)

def run_gate(Omega, delta_use, c_ops=None, tau=None):
    """Evolve the MS Hamiltonian for one closed loop; return reduced qubit rho."""
    if tau is None:
        tau = 2*np.pi/delta_use
    g = eta*Omega/2.0
    H = [[g*Sx*a, coeff_p], [g*Sx*a.dag(), coeff_m]]
    tlist = np.linspace(0.0, tau, 400)
    res = qt.mesolve(H, psi0, tlist, c_ops=c_ops or [], e_ops=[],
                     args={'delta': delta_use},
                     options={'nsteps': 20000, 'atol': 1e-9, 'rtol': 1e-7})
    rho_final = res.states[-1]
    rho_q = rho_final.ptrace([0, 1])           # trace out motion
    return rho_q

def bell_fidelity(rho_q):
    return float(np.real(qt.expect(qt.ket2dm(bell), rho_q)))

# ── 3. CALIBRATION: scan Omega, find the maximally-entangling point ──────────
print("="*64)
print(f"Closed-loop gate time tau_g = {tau_g*1e6:.2f} us  (delta = {delta/2/np.pi/1e3:.1f} kHz)")
print(f"Analytic Omega = delta/(2 eta) = 2*pi*{Omega_analytic/2/np.pi/1e3:.2f} kHz")
print("="*64)

Omega_scan = np.linspace(0.4, 1.6, 25) * Omega_analytic
fid_scan = np.array([bell_fidelity(run_gate(Om, delta)) for Om in Omega_scan])
Omega_opt = Omega_scan[np.argmax(fid_scan)]
F_opt = fid_scan.max()
print(f"Numerically optimal Omega = 2*pi*{Omega_opt/2/np.pi/1e3:.2f} kHz "
      f"({Omega_opt/Omega_analytic:.3f} x analytic)")
print(f"Bell-state fidelity at optimum (closed system) = {F_opt:.5f}")

# ── 4. DETUNING SWEEP at fixed (optimal) Omega ──────────────────────────────
# vary delta but keep gate time fixed at the calibrated tau_g -> loop mis-closes
delta_scan = delta * np.linspace(0.85, 1.15, 25)
fid_delta = np.array([bell_fidelity(run_gate(Omega_opt, d, tau=tau_g)) for d in delta_scan])

# ── 5. HEATING SWEEP at the calibrated point ────────────────────────────────
ndot_scan = np.array([0.0, 1e0, 1e1, 1e2, 3e2, 1e3, 3e3, 1e4])   # quanta/s
fid_heat = []
for nd in ndot_scan:
    c_ops = [np.sqrt(nd)*a.dag()] if nd > 0 else []
    fid_heat.append(bell_fidelity(run_gate(Omega_opt, delta, c_ops=c_ops)))
fid_heat = np.array(fid_heat)
print("-"*64)
for nd, f in zip(ndot_scan, fid_heat):
    print(f"  heating ndot = {nd:8.0f} /s   ->  Bell fidelity = {f:.5f}   (1-F = {1-f:.2e})")

# add a representative qubit-dephasing point for context
Tphi = 1.0e-3                      # 1 ms motional/qubit dephasing time
c_deph = [np.sqrt(1.0/(2*Tphi))*sz1, np.sqrt(1.0/(2*Tphi))*sz2]
F_deph = bell_fidelity(run_gate(Omega_opt, delta, c_ops=c_deph))
print(f"  qubit dephasing T_phi = 1 ms          ->  Bell fidelity = {F_deph:.5f}")
print("="*64)

# ── 6. PLOTS ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(Omega_scan/Omega_analytic, fid_scan, 'o-', color='#1f77b4')
ax.axvline(1.0, ls='--', color='grey', lw=1, label='analytic Omega = delta/(2 eta)')
ax.axvline(Omega_opt/Omega_analytic, ls=':', color='#d62728', lw=1.5, label='numeric optimum')
ax.set_xlabel('Omega / (delta / 2 eta)')
ax.set_ylabel('Bell-state fidelity')
ax.set_title('MS gate calibration: fidelity vs drive strength')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(PLOTS_DIR / 'ms_gate_calibration.png', dpi=150)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))
axL.plot(delta_scan/delta, fid_delta, 's-', color='#2ca02c')
axL.axvline(1.0, ls='--', color='grey', lw=1)
axL.set_xlabel('delta / delta_calibrated'); axL.set_ylabel('Bell-state fidelity')
axL.set_title('Detuning error (loop fails to close)'); axL.grid(alpha=0.3)

mask = ndot_scan > 0
axR.semilogx(ndot_scan[mask], 1 - fid_heat[mask], '^-', color='#9467bd')
axR.set_xlabel('motional heating rate ndot (quanta/s)')
axR.set_ylabel('Bell-state infidelity  1 - F')
axR.set_title('Heating-limited error budget'); axR.grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig(PLOTS_DIR / 'ms_gate_error_budget.png', dpi=150)

print("Saved: ms_gate_calibration.png, ms_gate_error_budget.png")
