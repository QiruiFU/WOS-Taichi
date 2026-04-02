"""
example_ellipse.py — Compare FD and WoS solvers on an elliptical domain.

Solves  ∇²u = 0  inside the ellipse  (x-0.5)²/0.4² + (y-0.5)²/0.3² ≤ 1
with boundary condition:
    u = 1  on the top half of the ellipse  (y ≥ 0.5)
    u = 0  on the bottom half

Run:
    python example_ellipse.py
"""

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)

from domain import EllipseDomain, SquareDomain
from fd    import FDSolver,  visualise as fd_vis
from wos   import WoSSolver, visualise as wos_vis

# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------
ellipse = EllipseDomain(cx=0.5, cy=0.5, a=0.4, b=0.3)

# ---------------------------------------------------------------------------
# FD solve
# ---------------------------------------------------------------------------
print("=" * 60)
print("Finite Difference solver — Ellipse domain")
print("=" * 60)
fd_solver = FDSolver(domain=ellipse, N=256)
fd_solver.solve(max_iters=200_000, tol=1e-6, check_every=500)
fd_vis(fd_solver, title="FD — Ellipse domain", save_path="fd_ellipse.png")

# ---------------------------------------------------------------------------
# WoS solve
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Walk-on-Spheres solver — Ellipse domain")
print("=" * 60)
GRID_RES  = 128
wos_solver = WoSSolver(domain=ellipse,
                       n_samples=GRID_RES * GRID_RES,
                       n_walks=256,
                       epsilon=1e-4,
                       max_steps=10000)
print(f"Running WoS  ({GRID_RES**2} samples, 256 walks) …")
wos_solver.solve()
wos_vis(wos_solver, title="WoS — Ellipse domain", save_path="wos_ellipse.png")

# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------
u_fd  = fd_solver.get_solution_numpy()
mask  = fd_solver.get_interior_mask_numpy() | \
        fd_solver.boundary_mask.to_numpy().astype(bool)
u_fd_masked = np.where(mask, u_fd, np.nan)

wos_vals, wos_origins = wos_solver.get_solution_numpy()
lo, hi = ellipse.bbox
grid_res = GRID_RES
wos_grid = np.full((grid_res, grid_res), np.nan)
for k in range(len(wos_vals)):
    ix = int((wos_origins[k, 0] - lo[0]) / (hi[0] - lo[0]) * grid_res)
    iy = int((wos_origins[k, 1] - lo[1]) / (hi[1] - lo[1]) * grid_res)
    ix = np.clip(ix, 0, grid_res - 1)
    iy = np.clip(iy, 0, grid_res - 1)
    wos_grid[iy, ix] = wos_vals[k]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ext = [lo[0], hi[0], lo[1], hi[1]]

im0 = axes[0].imshow(u_fd_masked.T, origin="lower", extent=ext,
                     cmap="hot", vmin=0, vmax=1)
plt.colorbar(im0, ax=axes[0])
axes[0].set_title("FD solution  (embedded grid)")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

im1 = axes[1].imshow(wos_grid, origin="lower", extent=ext,
                     cmap="hot", vmin=0, vmax=1)
plt.colorbar(im1, ax=axes[1])
axes[1].set_title("WoS solution  (Monte Carlo)")
axes[1].set_xlabel("x"); axes[1].set_ylabel("y")

plt.suptitle("Laplace equation on an elliptical domain", fontsize=13)
plt.tight_layout()
plt.savefig("comparison_ellipse.png", dpi=150)
print("\nComparison figure saved to comparison_ellipse.png")
plt.show()
