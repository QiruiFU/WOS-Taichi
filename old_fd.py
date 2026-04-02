"""
Finite Difference (FD) solver for the Laplace equation on a square domain,
implemented with Taichi. Designed to be compared against the WoS solver.

Problem:
    ∇²u(x, y) = 0  in Ω = [0,1]²
    u(x, y) = g(x, y) on ∂Ω

Method:
    Standard 5-point Gauss-Seidel / Jacobi iteration on a uniform grid.
    We use a red-black Gauss-Seidel scheme, which converges roughly 2× faster
    than plain Jacobi and maps naturally to parallel Taichi kernels.

Boundary condition (matches wos_laplace.py):
    u = 1  on the top edge
    u = 0  on left / right / bottom edges
"""

import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)

# ---------------------------------------------------------------------------
# Configuration  (keep in sync with wos_laplace.py for a fair comparison)
# ---------------------------------------------------------------------------
GRID_RES    = 256          # number of interior grid points per side (N×N grid)
MAX_ITERS   = 200_000       # maximum Gauss-Seidel sweeps
TOL         = 1e-6         # convergence tolerance (max-norm of residual)
CHECK_EVERY = 200          # check convergence every this many iterations


# ---------------------------------------------------------------------------
# Boundary condition  — must match wos_laplace.py
# ---------------------------------------------------------------------------
@ti.func
def boundary_value(i: int, j: int, N: int) -> float:
    """
    Dirichlet g at grid node (i, j) on the boundary of an (N+2)×(N+2) grid
    (ghost-cell convention: interior nodes are i,j ∈ [1, N]).

    Top row (j == N+1) → u = 1
    All other boundary nodes  → u = 0
    """
    val = 0.0
    if j == N + 1:      # top edge
        val = 1.0
    return val


# ---------------------------------------------------------------------------
# FD Solver
# ---------------------------------------------------------------------------
@ti.data_oriented
class FDSolver:
    """
    Solves ∇²u = 0 on [0,1]² with Dirichlet BCs using red-black Gauss-Seidel.

    Grid layout  (ghost-cell / halo convention):
      - Total field size: (N+2) × (N+2)
      - Interior nodes:   i, j ∈ [1 .. N]
      - Boundary nodes:   i or j ∈ {0, N+1}

    The mesh spacing is h = 1 / (N+1).
    """

    def __init__(self, N: int):
        self.N = N
        # Solution field (current and next for Jacobi; same field for G-S)
        self.u = ti.field(dtype=float, shape=(N + 2, N + 2))
        self.residual = ti.field(dtype=float, shape=())   # scalar for convergence

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    @ti.kernel
    def _init(self):
        N = self.N
        for i, j in self.u:
            on_boundary = (i == 0) or (i == N + 1) or (j == 0) or (j == N + 1)
            if on_boundary:
                self.u[i, j] = boundary_value(i, j, N)
            else:
                self.u[i, j] = 0.0

    # ------------------------------------------------------------------
    # One red-black Gauss-Seidel sweep
    # ------------------------------------------------------------------
    @ti.kernel
    def _gs_sweep(self, color: int):
        """
        Update all interior nodes of a given color (0=red, 1=black).
        Node (i,j) is red if (i+j) % 2 == 0, black otherwise.

        The update is the standard 5-point Laplacian stencil:
            u[i,j] = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]) / 4
        """
        N = self.N
        for i, j in ti.ndrange((1, N + 1), (1, N + 1)):
            if (i + j) % 2 == color:
                self.u[i, j] = 0.25 * (
                    self.u[i - 1, j] +
                    self.u[i + 1, j] +
                    self.u[i, j - 1] +
                    self.u[i, j + 1]
                )

    # ------------------------------------------------------------------
    # Residual  (max-norm of  ∇²u  at interior nodes)
    # ------------------------------------------------------------------
    @ti.kernel
    def _compute_residual(self):
        self.residual[None] = 0.0
        N   = self.N
        h = 1 / (self.N + 1)
        for i, j in ti.ndrange((1, N + 1), (1, N + 1)):
            lap = (self.u[i - 1, j] + self.u[i + 1, j] +
                   self.u[i, j - 1] + self.u[i, j + 1] -
                   4.0 * self.u[i, j]) / (h * h)
            ti.atomic_max(self.residual[None], ti.abs(lap))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def solve(self, max_iters: int = MAX_ITERS,
              tol: float = TOL, check_every: int = CHECK_EVERY):
        self._init()
        print(f"  Grid: {self.N}×{self.N} interior nodes  "
              f"(mesh spacing h = {1.0/(self.N+1):.4f})")

        for it in range(1, max_iters + 1):
            self._gs_sweep(0)   # red nodes
            self._gs_sweep(1)   # black nodes

            if it % check_every == 0:
                self._compute_residual()
                res = self.residual[None]
                print(f"  iter {it:6d}  |residual|∞ = {res:.3e}")
                if res < tol:
                    print(f"  Converged at iteration {it}.")
                    break
        else:
            print(f"  Reached max iterations ({max_iters}).")

    def get_solution_numpy(self) -> np.ndarray:
        """Return the (N+2)×(N+2) solution array (includes boundary rows/cols)."""
        return self.u.to_numpy()


# ---------------------------------------------------------------------------
# Visualisation  (standalone + comparison helper)
# ---------------------------------------------------------------------------
def visualise(solver: FDSolver):
    u_np = solver.get_solution_numpy()   # shape (N+2, N+2)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        u_np.T, origin="lower", extent=[0, 1, 0, 1],
        cmap="hot", vmin=0, vmax=1,
    )
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Finite Difference solution  ({solver.N}×{solver.N} grid)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig("fd_solution.png", dpi=150)
    plt.show()
    print("Figure saved to fd_solution.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    solver = FDSolver(N=GRID_RES)
    print(f"Running Finite Difference solver  (red-black Gauss-Seidel) …")
    solver.solve()
    visualise(solver)