import sys
from pathlib import Path

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Sanity-check the install ──────────────────────────────────────────────
print("="*60)
print(f"QuTiP version: {qt.__version__}")
print(f"NumPy  version: {np.__version__}")
print(f"Python version: {sys.version.split()[0]}")
print("="*60)

# ── 2. Build the three canonical states ──────────────────────────────────────
ket_0 = qt.basis(2, 0)                     # |0>, ground state
ket_1 = qt.basis(2, 1)                     # |1>, excited state
ket_plus = (ket_0 + ket_1).unit()          # |+> = (|0> + |1>) / sqrt(2)

print()
print("|0> =")
print(ket_0)
print()
print("|+> =")
print(ket_plus)

# ── 3. Inner-product sanity checks ───────────────────────────────────────────
def inner(a, b):
    return a.overlap(b)

print()
print(f"<0|0> = {inner(ket_0, ket_0):.4f}  (expect 1+0j)")
print(f"<0|1> = {inner(ket_0, ket_1):.4f}  (expect 0+0j)")
print(f"<+|+> = {inner(ket_plus, ket_plus):.4f}  (expect 1+0j)")

# ── 4. Verify Pauli operator expectation values ──────────────────────────────
sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()

def bloch_vector(psi):
    """Return (<sx>, <sy>, <sz>) for the state psi."""
    return (
        qt.expect(sx, psi),
        qt.expect(sy, psi),
        qt.expect(sz, psi),
    )

print()
print(f"Bloch vector of |0>  = {bloch_vector(ket_0)}   (expect (0, 0, +1))")
print(f"Bloch vector of |1>  = {bloch_vector(ket_1)}   (expect (0, 0, -1))")
print(f"Bloch vector of |+>  = {bloch_vector(ket_plus)} (expect (+1, 0, 0))")

# ── 5. Render the three states on Bloch spheres ──────────────────────────────
fig = plt.figure(figsize=(15, 5))
states_labels = [
    (ket_0,    "|0>: ground"),
    (ket_1,    "|1>: excited"),
    (ket_plus, "|+>: superposition"),
]

for i, (psi, label) in enumerate(states_labels):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    b = qt.Bloch(axes=ax)
    b.add_states(psi)
    b.vector_color = ["darkblue"]
    b.point_color = ["darkblue"]
    b.frame_alpha = 0.15
    b.render()
    ax.set_title(label, fontsize=12, pad=10)

plt.suptitle("Three canonical single-qubit states", fontsize=14, y=1.02)
plt.tight_layout()

save_path = OUTPUT_DIR / "bloch_sphere_verification.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print()
print(f"Saved: {save_path}")
print()
print("If the inner products and Bloch vectors above match the expected values,")
print("QuTiP is wired up correctly. Next: driven_rabi.py.")
