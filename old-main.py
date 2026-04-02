"""
Walk on Spheres (WoS) solver for the Laplace equation on a square domain,
implemented with Taichi for GPU-parallel execution.

Problem:
    ∇²u(x) = 0  in Ω
    u(x) = g(x) on ∂Ω

Algorithm:
    For each sample point x₀ inside Ω:
    1. Compute R = dist(x, ∂Ω), the radius of the largest sphere fully inside Ω.
    2. Sample a new point x uniformly on the sphere of radius R centered at x.
    3. Repeat until x falls within the ε-shell (dist(x, ∂Ω) < ε).
    4. Assign u(x) ≈ g(x_boundary) as the boundary value at that walk's endpoint.
    The final estimate at x₀ is the average over many such walks.
"""

import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)  # fallback to ti.cpu if no GPU is available

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOMAIN_MIN  = ti.Vector([0.0, 0.0])   # lower-left corner of square domain
DOMAIN_MAX  = ti.Vector([1.0, 1.0])   # upper-right corner

GRID_RES    = 256                      # resolution for visualisation grid
N_SAMPLES   = GRID_RES * GRID_RES     # total sample points inside the domain
N_WALKS     = 5000                     # independent random walks per sample point
EPSILON     = 1e-4                    # convergence shell thickness
MAX_STEPS   = 10000                   # safety cap on walk length


# ---------------------------------------------------------------------------
# Square domain
# ---------------------------------------------------------------------------
@ti.data_oriented
class SquareDomain:
    """
    Axis-aligned square domain [lo, hi]².

    Provides:
      - dist_to_boundary(x)  → minimum distance from x to any edge
      - boundary_value(x)    → Dirichlet g(x) evaluated at (or near) the boundary
    """

    def __init__(self, lo: ti.template(), hi: ti.template()):
        self.lo = lo
        self.hi = hi

    @ti.func
    def dist_to_boundary(self, x: tm.vec2) -> float:
        """
        Return the distance from interior point x to the nearest edge.
        For a rectangle this is simply the minimum of the four edge distances.
        """
        d_left   = x[0] - self.lo[0]
        d_right  = self.hi[0] - x[0]
        d_bottom = x[1] - self.lo[1]
        d_top    = self.hi[1] - x[1]
        return ti.min(d_left, d_right, d_bottom, d_top)

    @ti.func
    def boundary_value(self, x: tm.vec2) -> float:
        """
        Dirichlet boundary condition g(x).

        Example: u = 1 on the top edge, u = 0 on all other edges.
        You can replace this with any function you like.
        """
        # Identify which edge x is closest to and return its prescribed value.
        d_left   = x[0] - self.lo[0]
        d_right  = self.hi[0] - x[0]
        d_bottom = x[1] - self.lo[1]
        d_top    = self.hi[1] - x[1]

        d_min = ti.min(d_left, d_right, d_bottom, d_top)

        val = 0.0
        if d_top == d_min:
            val = 1.0

        return val


# ---------------------------------------------------------------------------
# Sample-point struct (AoS via Taichi fields)
# ---------------------------------------------------------------------------
@ti.dataclass
class WalkState:
    pos        : tm.vec2   # current position of the walker
    value      : float     # accumulated boundary value (written at termination)
    step       : int       # steps taken in the current walk
    terminated : int       # 1 once the walker has entered the ε-shell


# ---------------------------------------------------------------------------
# WoS Solver
# ---------------------------------------------------------------------------
@ti.data_oriented
class WoSSolver:

    def __init__(self, domain: SquareDomain,
                 n_samples: int, n_walks: int,
                 epsilon: float, max_steps: int):

        self.domain    = domain
        self.n_samples = n_samples
        self.n_walks   = n_walks
        self.epsilon   = epsilon
        self.max_steps = max_steps

        # ---- persistent fields ----
        # origin of each sample (fixed throughout all walks)
        self.origins   = ti.Vector.field(2, dtype=float, shape=n_samples)

        # per-sample accumulator and walk counter
        self.accum     = ti.field(dtype=float, shape=n_samples)
        self.n_done    = ti.field(dtype=int,   shape=n_samples)

        # live walker states (one walker per sample, reused across walks)
        self.walkers   = WalkState.field(shape=n_samples)

        # final solution value at each sample
        self.solution  = ti.field(dtype=float, shape=n_samples)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    @ti.kernel
    def _init_origins(self):
        """Scatter sample origins uniformly inside the domain (rejection-free)."""
        lo = self.domain.lo
        hi = self.domain.hi
        for i in self.origins:
            # Halton-like deterministic placement on a regular grid,
            # with a small random jitter so samples are not on a lattice.
            side = ti.cast(ti.sqrt(float(self.n_samples)), int)
            ix   = i % side
            iy   = i // side
            dx   = (hi[0] - lo[0]) / float(side)
            dy   = (hi[1] - lo[1]) / float(side)
            self.origins[i] = tm.vec2(
                lo[0] + (ix + 0.5) * dx,
                lo[1] + (iy + 0.5) * dy,
            )

    @ti.kernel
    def _reset_accumulators(self):
        for i in range(self.n_samples):
            self.accum[i]  = 0.0
            self.n_done[i] = 0

    @ti.kernel
    def _reset_walkers(self):
        """Reset all walkers to their origin for a new batch of walks."""
        for i in range(self.n_samples):
            self.walkers[i].pos        = self.origins[i]
            self.walkers[i].value      = 0.0
            self.walkers[i].step       = 0
            self.walkers[i].terminated = 0

    # ------------------------------------------------------------------
    # One parallel step of the Walk-on-Spheres algorithm
    # ------------------------------------------------------------------
    @ti.func
    def _sample_on_circle(self, center: tm.vec2, radius: float) -> tm.vec2:
        """
        Sample a point uniformly on the 2-D sphere (circle) of given radius.
        Uses a simple angle parameterisation.
        """
        angle = 2.0 * tm.pi * ti.random()
        return center + radius * tm.vec2(tm.cos(angle), tm.sin(angle))

    @ti.kernel
    def _walk_step(self, walk_idx: int, step_idx: int):
        """
        Advance every non-terminated walker by one WoS step.
        Executed once per (walk, step) pair.
        """
        for i in range(self.n_samples):
            if self.walkers[i].terminated == 0:
                x = self.walkers[i].pos
                R = self.domain.dist_to_boundary(x)

                if R < self.epsilon:
                    # Walker has reached the ε-shell → record boundary value
                    bv = self.domain.boundary_value(x)
                    self.walkers[i].value      = bv
                    self.walkers[i].terminated = 1
                else:
                    # Sample uniformly on the largest inscribed sphere
                    new_pos = self._sample_on_circle(x, R)
                    self.walkers[i].pos   = new_pos
                    self.walkers[i].step += 1

                    # Safety: force termination after max_steps
                    if self.walkers[i].step >= self.max_steps:
                        bv = self.domain.boundary_value(new_pos)
                        self.walkers[i].value      = bv
                        self.walkers[i].terminated = 1

    @ti.kernel
    def _all_terminated(self) -> int:
        """Return 1 if every walker has terminated, else 0."""
        all_done = 1
        for i in range(self.n_samples):
            if self.walkers[i].terminated == 0:
                all_done = 0
        return all_done

    @ti.kernel
    def _accumulate(self):
        """After a walk finishes, add each walker's value to its sample's accumulator."""
        for i in range(self.n_samples):
            self.accum[i]  += self.walkers[i].value
            self.n_done[i] += 1

    # ------------------------------------------------------------------
    # Run one complete random walk for all sample points
    # ------------------------------------------------------------------
    def _run_single_walk(self, walk_idx: int):
        self._reset_walkers()
        # print("start walking", self.origins[2080])
        for step in range(self.max_steps):
            self._walk_step(walk_idx, step)
            # print(step, self.walkers[2080].pos, self.walkers[2080].value)
            if self._all_terminated():
                break

        self._accumulate()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    @ti.kernel
    def _compute_solution(self):
        for i in range(self.n_samples):
            n = self.n_done[i]
            if n > 0:
                self.solution[i] = self.accum[i] / float(n)
            else:
                self.solution[i] = 0.0

    def solve(self):
        """Run N_WALKS walks and compute the mean estimate at every sample point."""
        self._init_origins()
        self._reset_accumulators()

        for w in range(self.n_walks):
            if w % 100 == 0:
                print(f"  Walk {w:4d}/{self.n_walks}")
            self._run_single_walk(w)

        self._compute_solution()
        print("  Done.")

    def get_solution_numpy(self):
        return self.solution.to_numpy(), self.origins.to_numpy()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def visualise(solver: WoSSolver, grid_res: int):
    values, origins = solver.get_solution_numpy()

    lo = DOMAIN_MIN.to_numpy()
    hi = DOMAIN_MAX.to_numpy()

    # Interpolate scattered samples onto a regular grid (nearest neighbour)
    grid = np.zeros((grid_res, grid_res))
    count = np.zeros((grid_res, grid_res), dtype=int)

    for k in range(len(values)):
        ix = int((origins[k, 0] - lo[0]) / (hi[0] - lo[0]) * grid_res)
        iy = int((origins[k, 1] - lo[1]) / (hi[1] - lo[1]) * grid_res)
        ix = np.clip(ix, 0, grid_res - 1)
        iy = np.clip(iy, 0, grid_res - 1)
        grid[iy, ix] += values[k]
        count[iy, ix] += 1

    mask = count > 0
    grid[mask] /= count[mask]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].imshow(
        grid, origin="lower", extent=[lo[0], hi[0], lo[1], hi[1]],
        cmap="hot", vmin=0, vmax=1,
    )
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title("WoS solution  u(x, y)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    sc = axes[1].scatter(origins[:, 0], origins[:, 1],
                         c=values, cmap="hot", s=4, vmin=0, vmax=1)
    plt.colorbar(sc, ax=axes[1])
    axes[1].set_title("Sample-point values")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")

    plt.tight_layout()
    plt.savefig("wos_solution.png", dpi=150)
    plt.show()
    print("Figure saved to wos_solution.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    domain = SquareDomain(lo=DOMAIN_MIN, hi=DOMAIN_MAX)

    solver = WoSSolver(
        domain    = domain,
        n_samples = N_SAMPLES,
        n_walks   = N_WALKS,
        epsilon   = EPSILON,
        max_steps = MAX_STEPS,
    )

    print(f"Running Walk-on-Spheres  ({N_SAMPLES} samples, {N_WALKS} walks each) …")
    solver.solve()

    visualise(solver, GRID_RES)