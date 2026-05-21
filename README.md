# QuTiP Superconducting Simulation: Coherence, Driven Dynamics, and DRAG Leakage Suppression

**Author:** Paulino "Paul" Gin · BS Applied Physics + BA Mathematics, Boston College (Class of 2027)
**Stack:** QuTiP 5 · Python (NumPy / SciPy / Matplotlib / Pandas)
**Repo:** [github.com/paulggin/qutip-superconducting-simulation](https://github.com/paulggin/qutip-superconducting-simulation)
**Companion project:** [ibm-quantum-coherence-characterization](https://github.com/paulggin/ibm-quantum-coherence-characterization)

---

## Overview

This project constructs a Hamiltonian simulation of a superconducting transmon qubit in QuTiP and validates it against real `ibm_marrakesh` Q0 data from the companion IBM Quantum project. The project builds from unitary single-qubit behavior to a full 3-level transmon model with a DRAG-corrected pulse, introducing new aspects of the physics with each step:

1. Verify the install with three canonical states on the Bloch sphere.
2. Drive a clean Rabi oscillation under `qt.sesolve` and recover the input Rabi frequency from a sinusoidal fit.
3. Add amplitude damping with the Lindblad master equation, anchor T1 to real Q0 hardware data, and confirm the simulation reproduces the measured decay.
4. Add pure dephasing T_phi to recover the T2 Ramsey envelope from the same Q0 data.
5. Move to a 3-level Hilbert space, apply a Gaussian X gate, and use first-order DRAG to suppress leakage to the |2> state. Repeat the comparison with T1 and T_phi collapse operators turned on.

The deliverable is a complete framework that links established transmon physics (rotating-frame Hamiltonians, Lindblad collapse operators, and DRAG pulse shaping) to real experimental data measured on IBM Quantum hardware.

---

## Background

A superconducting transmon qubit is an oscillator with slight anharmonicity. Its three lowest levels (|0>, |1>, |2>) sit at energies roughly omega_01, 2 omega_01 + alpha, where alpha is the negative anharmonicity (here, alpha / 2pi = -0.3 GHz). For most algorithm work the |2> state is ignored and the system is treated as a qubit, but for fast pulses the |1>-|2> transition is close enough in frequency that the drive spills population into |2>: leakage out of the computational subspace.

Coherence times set the practical depth budget. **T1** is energy relaxation: the time for an excited qubit to lose energy to its environment and decay to |0>. **T2** is dephasing: the time for a superposition state to lose its definite phase. The two are tied by the relation `1/T2 = 1/(2 T1) + 1/T_phi`, where T_phi captures pure dephasing from low-frequency noise. A qubit with T2 close to 2 T1 is amplitude-damping-limited; a qubit with T2 well below 2 T1 is dephasing-limited.

QuTiP simulates this with two solvers:

- `qt.sesolve` integrates the Schrodinger equation for unitary dynamics.
- `qt.mesolve` integrates the Lindblad master equation with one collapse operator per dissipation channel.

---

## Methods

### Code architecture

Each experiment is a single Python file in `experiments/` that imports `qutip` and the standard scientific stack. The T1, T2, and transmon scripts all read the same Q0 metadata from `data/ibm_marrakesh_q0_metadata.json`, which guarantees consistent T1 / T2 / T_phi values across the three Lindblad simulations. Each script writes its plot to `plots/`.

### Bloch sphere verification (`setup_verification.py`)

Renders three states on three Bloch spheres: |0> at the north pole, |1> at the south pole, |+> = (|0> + |1>) / sqrt(2) on the +x axis. Inner products are printed to confirm `<0|0> = 1`, `<0|1> = 0`, `<+|+> = 1`, and the three Bloch vectors come out as (0,0,+1), (0,0,-1), (+1,0,0). The point of this step is to fail fast if the QuTiP install is misconfigured before any time-dependent dynamics are introduced.

### Driven Rabi oscillation (`driven_rabi.py`)

Hamiltonian: `H = (Omega_R / 2) sigma_x` with `Omega_R = 2 pi · 1.0` in dimensionless units (oscillation period equals 1). Initial state |0>. `qt.sesolve` returns `<sigma_z>(t)`, converted to `P(|1>) = (1 - <sigma_z>) / 2`. A four-parameter sinusoidal fit `A (1 - cos(omega t + phi))/2 + C` is applied with `scipy.optimize.curve_fit`. The script also re-solves with `states=True` to trace the Bloch sphere path of the precession.

### T1 Lindblad master equation (`t1_lindblad.py`)

Collapse operator: `L_1 = sqrt(1/T1) · sigma_minus`, where `sigma_minus = qt.destroy(2)`. Hamiltonian is zero (free evolution). Initial state |1>. The script runs `qt.mesolve` three times: once at the measured T1 from Q0 (315.06 us), once at the IBM published T1 (344.48 us), and once at an offset-corrected T1 fit that accounts for the readout assignment-error floor on the real data. Each simulation is interpolated onto the real delay grid and RMS residuals are computed.

### T2 Ramsey with T1 + T_phi (`t2_ramsey.py`)

Two collapse operators: amplitude damping `sqrt(1/T1) · sigma_minus` and pure dephasing `sqrt(1/(2 T_phi)) · sigma_z`. The factor of 2 inside the square root makes the qubit-subspace dephasing rate equal to 1/T_phi. T_phi is derived from `1/T_phi = 1/T2 - 1/(2 T1)` using the offset-corrected T1 = 319.70 us and the measured T2 = 46.08 us. Initial state |+>. Hamiltonian `H = pi · delta_f · sigma_z` reproduces the deliberate 0.052 MHz detuning used in the real Ramsey experiment.

### 3-level transmon with DRAG (`transmon_drag.py`)

Hilbert space: 3 levels via `qt.destroy(3)`. Free Hamiltonian in the rotating frame at omega_01: `H_0 = alpha · |2><2|` with alpha / 2pi = -0.3 GHz. Drive in the rotating-wave approximation:

```
H_d(t) = (Omega_x(t)/2)(a + a^dag) + (Omega_y(t)/2) i(a^dag - a)
```

`Omega_x(t)` is a Gaussian envelope centered at TG/2 with width sigma, truncated at the gate boundaries. The pulse area is calibrated so that the integral of `Omega_x` equals pi: this gives a pi rotation on the qubit subspace. Two variants are compared:

- **Square (no DRAG):** `Omega_y(t) = 0`.
- **DRAG:** `Omega_y(t) = -dOmega_x/dt / (2 alpha)`. The factor of 1/(2 alpha) is the leading-order cancellation coefficient that minimizes leakage to |2>, confirmed by a numerical beta sweep.

Pulse parameters: TG = 6 ns, sigma = 1.5 ns. This puts the pulse in the |alpha| · sigma ~ 1 regime where the Gaussian's spectral content at the |1>-|2> frequency is large enough to cause visible leakage. Slower pulses already leak negligibly and DRAG has nothing to correct.

Open-system version adds two collapse operators on the 3-level space: `L_1 = sqrt(1/T1) · a` (amplitude damping) and `L_phi = sqrt(2/T_phi) · a^dag a` (pure dephasing via the number operator). T1 = 319.70 us and T_phi = 49.65 us from the Q0 anchor.

---

## Results

### Driven Rabi (closed system)

The fitted Rabi frequency matches the input frequency to within numerical precision (`|Omega_fit - Omega_R| / Omega_R < 0.01%`).

![Driven Rabi oscillation](plots/driven_rabi.png)

The Bloch trajectory traces the expected great-circle path in the y-z plane: a pure X rotation precesses through |0> -> |-i y> -> |1> -> |+i y> -> |0> at the Rabi rate.

![Bloch trajectory under X drive](plots/driven_rabi_bloch_trajectory.png)

### T1 Lindblad vs real ibm_marrakesh Q0

| Quantity | Value |
| :-- | --: |
| Q0 measured T1 (offset-biased fit to sim) | **315.06 us** |
| Q0 published T1 (IBM calibration book) | 344.48 us |
| Q0 offset-corrected T1 (fit to real data with readout floor C) | **319.70 us** |
| Recovered T1 from fit to simulated curve | 315.0636 us (self-consistency: 0.0001%) |
| RMS residual (T1 = 315.06 us) | 0.0196 |
| RMS residual (T1 = 344.48 us) | 0.0185 |

The Lindblad solver and the exponential curve fitter are self-consistent to four decimal places: feeding T1 = 315.0639 us into the master equation and refitting the resulting curve returns T1 = 315.0636 us. On the real data, the publish