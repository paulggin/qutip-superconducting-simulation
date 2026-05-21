# Repository Inventory

Snapshot of the QuTiP Superconducting Simulation project.

## Install

```bash
pip install -r requirements.txt
```

## Layout

```
.
├── README.md
├── INVENTORY.md
├── .gitignore
├── requirements.txt
│
├── experiments/                              ← QuTiP simulations
│   ├── setup_verification.py                 ← QuTiP install check; renders |0>, |1>, |+> on Bloch spheres
│   ├── driven_rabi.py                        ← unitary Schrodinger evolution under H = (Omega_R/2) sigma_x
│   ├── t1_lindblad.py                        ← Lindblad amplitude damping vs real ibm_marrakesh Q0 T1 data
│   ├── t2_ramsey.py                          ← Lindblad T1 + T_phi Ramsey vs real ibm_marrakesh Q0 T2 data
│   └── transmon_drag.py                      ← 3-level transmon (|0>, |1>, |2>) with DRAG leakage suppression
│
├── data/                                     ← real-hardware reference data from project 1
│   ├── ibm_marrakesh_q0_t1.csv               ← 15 delays, P(|1>) at each, 1024 shots/point
│   ├── ibm_marrakesh_q0_t2.csv               ← Ramsey data, 0-100 us delay range
│   └── ibm_marrakesh_q0_metadata.json        ← T1/T2 fit values, EPC, gate fidelity, job IDs
│
└── plots/                                    ← final figures
    ├── bloch_sphere_verification.png         ← |0>, |1>, |+> rendered on three Bloch spheres
    ├── driven_rabi.png                       ← Rabi oscillation + fit
    ├── driven_rabi_bloch_trajectory.png      ← Bloch sphere trajectory under X drive
    ├── t1_lindblad_simulator_vs_real.png     ← regenerate by running experiments/t1_lindblad.py
    ├── t2_ramsey_simulator_vs_real.png       ← regenerate by running experiments/t2_ramsey.py
    ├── transmon_drag_comparison.png          ← unitary square vs DRAG, two panels
    └── transmon_drag_open_system.png         ← same comparison with T1 + T_phi collapse operators
```
