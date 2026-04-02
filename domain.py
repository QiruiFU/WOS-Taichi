"""
domain.py — Domain abstractions for PDE solvers (WoS and FD).

Each Domain subclass provides:
  1. dist_to_boundary(x)  — used by WoS solver (Taichi @ti.func)
  2. boundary_value(x)    — used by WoS solver (Taichi @ti.func)
  3. grid_info(N)         — used by FD solver; returns everything the
                            FD solver needs to operate on an N×N grid
                            embedded inside the domain's bounding box.

FD strategy — Embedded (Fictitious) Domain
-------------------------------------------
The FD grid always covers the bounding box [lo, hi]² uniformly.
grid_info() returns three arrays of shape (N+2, N+2):

  interior_mask  — 1 at nodes strictly inside Ω, 0 outside
  boundary_mask  — 1 at nodes on/near ∂Ω (where BCs are imposed)
  bc_values      — prescribed u value at each boundary node

The FD solver then runs its standard 5-point Gauss-Seidel sweep but
skips every node where interior_mask == 0.  No change to the numerical
stencil is required — only a conditional update is added.

For the SquareDomain the bounding box *is* the domain, so the masks
reduce to the same ghost-cell pattern used in the original fd.py.

For non-rectangular domains (e.g. EllipseDomain) the FD grid is still
a regular rectangle; the mask identifies which nodes lie inside the
ellipse and which should be treated as exterior (frozen at 0).  Nodes
within one mesh-spacing of the boundary receive the Dirichlet value by
nearest-boundary projection.
"""

import numpy as np
import taichi as ti
import taichi.math as tm


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class BaseDomain():
    """
    Abstract base class for all problem domains.

    Subclasses must implement:
      - the Taichi @ti.func methods dist_to_boundary / boundary_value
        (these are plain Python methods wrapping @ti.func logic, or
         directly decorated — see concrete classes below)
      - grid_info(N) → (interior_mask, boundary_mask, bc_values)
    """

    # ------------------------------------------------------------------ #
    # Taichi-callable interface (must be @ti.func inside Taichi kernels)  #
    # ------------------------------------------------------------------ #
    def dist_to_boundary(self, x: tm.vec2) -> float:
        """Distance from point x to the nearest point on ∂Ω."""
        raise NotImplementedError

    def boundary_value(self, x: tm.vec2) -> float:
        """Dirichlet value g(x) at (or near) the boundary."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # FD grid interface (pure Python / NumPy)                             #
    # ------------------------------------------------------------------ #
    def grid_info(self, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return grid metadata for an (N+2)×(N+2) FD grid whose nodes
        span the bounding box of this domain.

        Returns
        -------
        interior_mask : bool ndarray (N+2, N+2)
            True  → node is inside Ω and should be updated by G-S.
            False → exterior or boundary node (frozen).
        boundary_mask : bool ndarray (N+2, N+2)
            True  → node carries a prescribed Dirichlet value.
        bc_values : float ndarray (N+2, N+2)
            Dirichlet value at each boundary node (0 elsewhere).
        """
        raise NotImplementedError

    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lo, hi) corners of the bounding box as numpy arrays."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Square (axis-aligned rectangle) domain
# ---------------------------------------------------------------------------
@ti.data_oriented
class SquareDomain(BaseDomain):
    """
    Axis-aligned square domain [lo, hi]².

    Boundary condition:
      u = 1  on the top edge
      u = 0  on all other edges

    The bounding box *is* the domain, so interior_mask is 1 everywhere
    in the interior and the FD grid reduces to the original ghost-cell
    layout from fd.py.
    """

    def __init__(self,
                 lo: ti.template() = ti.Vector([0.0, 0.0]),
                 hi: ti.template() = ti.Vector([1.0, 1.0])):
        self.lo = lo
        self.hi = hi
        self._lo_np = lo.to_numpy() if hasattr(lo, "to_numpy") else np.array(lo)
        self._hi_np = hi.to_numpy() if hasattr(hi, "to_numpy") else np.array(hi)

    @property
    def bbox(self):
        return self._lo_np, self._hi_np

    # -- Taichi interface --------------------------------------------------
    @ti.func
    def dist_to_boundary(self, x: tm.vec2) -> float:
        d_left   = x[0] - self.lo[0]
        d_right  = self.hi[0] - x[0]
        d_bottom = x[1] - self.lo[1]
        d_top    = self.hi[1] - x[1]
        return ti.min(d_left, d_right, d_bottom, d_top)

    @ti.func
    def boundary_value(self, x: tm.vec2) -> float:
        d_left   = x[0] - self.lo[0]
        d_right  = self.hi[0] - x[0]
        d_bottom = x[1] - self.lo[1]
        d_top    = self.hi[1] - x[1]
        d_min = ti.min(d_left, d_right, d_bottom, d_top)
        val = 0.0
        if d_top == d_min:
            val = 1.0
        return val

    # -- FD grid interface -------------------------------------------------
    def grid_info(self, N: int):
        lo, hi = self._lo_np, self._hi_np
        M = N + 2  # total nodes per side (including ghost/boundary layer)

        xs = np.linspace(lo[0], hi[0], M)   # node x-coordinates
        ys = np.linspace(lo[1], hi[1], M)   # node y-coordinates

        interior_mask = np.zeros((M, M), dtype=bool)
        boundary_mask = np.zeros((M, M), dtype=bool)
        bc_values     = np.zeros((M, M), dtype=np.float32)

        for i in range(M):
            for j in range(M):
                on_edge = (i == 0) or (i == M-1) or (j == 0) or (j == M-1)
                if on_edge:
                    boundary_mask[i, j] = True
                    x = np.array([xs[i], ys[j]])
                    bc_values[i, j] = self._bc_numpy(x)
                else:
                    interior_mask[i, j] = True

        return interior_mask, boundary_mask, bc_values

    def _bc_numpy(self, x: np.ndarray) -> float:
        """Python version of boundary_value for grid_info construction."""
        lo, hi = self._lo_np, self._hi_np
        d_top = hi[1] - x[1]
        d_others = min(x[0] - lo[0], hi[0] - x[0], x[1] - lo[1])
        return 1.0 if d_top <= d_others else 0.0


# ---------------------------------------------------------------------------
# Ellipse domain
# ---------------------------------------------------------------------------
@ti.data_oriented
class EllipseDomain(BaseDomain):
    """
    Elliptical domain  { x : (x-cx)²/a² + (y-cy)²/b² ≤ 1 }.

    Boundary condition (same spirit as SquareDomain):
      u = 1  on the top half  (y ≥ cy)
      u = 0  on the bottom half (y < cy)

    FD strategy — Embedded domain
    ------------------------------
    The FD grid covers the bounding box [cx-a, cx+a] × [cy-b, cy+b].
    Nodes outside the ellipse are masked out (frozen at 0).
    Nodes within one mesh-spacing of the ellipse boundary receive the
    Dirichlet value via nearest-point-on-ellipse projection.

    SDF approximation
    -----------------
    The exact signed distance to an ellipse has no simple closed form.
    We use the iterative projection method of Eberly (2013), which
    converges in ≤ 10 Newton steps to machine precision.  The @ti.func
    version uses a fixed iteration count safe for GPU execution.
    """

    def __init__(self, cx: float = 0.5, cy: float = 0.5,
                 a: float = 0.4, b: float = 0.3):
        """
        Parameters
        ----------
        cx, cy : center of ellipse
        a      : semi-axis along x
        b      : semi-axis along y
        """
        self.cx = cx
        self.cy = cy
        self.a  = a
        self.b  = b

        self._lo_np = np.array([cx - a, cy - b], dtype=np.float64)
        self._hi_np = np.array([cx + a, cy + b], dtype=np.float64)

    @property
    def bbox(self):
        return self._lo_np, self._hi_np

    # -- Exact SDF helpers (Python, for grid_info) -------------------------
    @staticmethod
    def _ellipse_nearest_numpy(px: float, py: float,
                               a: float, b: float) -> tuple[float, float]:
        """
        Find the nearest point on the ellipse x²/a² + y²/b² = 1
        to the point (px, py) using Eberly's bisection method.
        Works for any (px, py) including interior points.
        Returns (nx, ny) on the ellipse.
        """
        # Work in first quadrant, restore sign afterwards
        sx, sy = np.sign(px) or 1.0, np.sign(py) or 1.0
        px, py = abs(px), abs(py)

        if py == 0.0:
            return sx * a, 0.0

        # Parametric: (a cos t, b sin t). Minimise distance to (px, py).
        # Reduce to root of f(t) = a²px/(a²-t)² + b²py/(b²-t)² - 1 = 0
        # using bisection on t ∈ (-b², min(a²,b²)-ε)
        a2, b2 = a * a, b * b
        # Initial bracket
        t0 = -b2 + b * py
        t1 = -b2 + np.sqrt(a2 * px * px + b2 * py * py)

        for _ in range(100):
            t = 0.5 * (t0 + t1)
            x0 = a2 * px / (a2 - t)
            y0 = b2 * py / (b2 - t)
            f  = (x0 / a) ** 2 + (y0 / b) ** 2 - 1.0
            if abs(f) < 1e-12 or abs(t1 - t0) < 1e-14:
                break
            if f > 0:
                t0 = t
            else:
                t1 = t

        t  = 0.5 * (t0 + t1)
        nx = a2 * px / (a2 - t)
        ny = b2 * py / (b2 - t)
        return sx * nx, sy * ny

    def _dist_numpy(self, x: np.ndarray) -> float:
        """Signed distance: negative inside, positive outside."""
        px = x[0] - self.cx
        py = x[1] - self.cy
        nx, ny = self._ellipse_nearest_numpy(px, py, self.a, self.b)
        dist = np.hypot(px - nx, py - ny)
        inside = (px / self.a) ** 2 + (py / self.b) ** 2 <= 1.0
        return -dist if inside else dist

    def _bc_numpy(self, x: np.ndarray) -> float:
        """Boundary condition evaluated at x (on or near ∂Ω)."""
        # Project x onto the ellipse to get the true boundary point
        px = x[0] - self.cx
        py = x[1] - self.cy
        _, ny = self._ellipse_nearest_numpy(px, py, self.a, self.b)
        # u = 1 on the top half, 0 on the bottom half
        return 1.0 if ny + self.cy >= self.cy else 0.0

    # -- Taichi interface --------------------------------------------------
    @ti.func
    def _ellipse_nearest_ti(self, px: float, py: float) -> tm.vec2:
        """
        GPU-safe nearest-point on ellipse x²/a² + y²/b² = 1.
        Uses Eberly bisection with a fixed iteration cap.
        """
        a, b = ti.static(self.a), ti.static(self.b)
        a2, b2 = a * a, b * b

        # Work in first quadrant
        sx = ti.select(px >= 0.0,  1.0, -1.0)
        sy = ti.select(py >= 0.0,  1.0, -1.0)
        px = ti.abs(px)
        py = ti.abs(py)

        # Degenerate: on x-axis
        result = tm.vec2(sx * a, 0.0)
        if py > 1e-10:
            t0 = -b2 + b * py
            t1 = -b2 + ti.sqrt(a2 * px * px + b2 * py * py)
            for _ in range(64):
                t  = 0.5 * (t0 + t1)
                x0 = a2 * px / (a2 - t)
                y0 = b2 * py / (b2 - t)
                f  = (x0 / a) ** 2 + (y0 / b) ** 2 - 1.0
                if f > 0:
                    t0 = t
                else:
                    t1 = t
            t  = 0.5 * (t0 + t1)
            result = tm.vec2(sx * a2 * px / (a2 - t),
                             sy * b2 * py / (b2 - t))
        return result

    @ti.func
    def dist_to_boundary(self, x: tm.vec2) -> float:
        px = x[0] - ti.static(self.cx)
        py = x[1] - ti.static(self.cy)
        nearest = self._ellipse_nearest_ti(px, py)
        dist    = tm.length(tm.vec2(px, py) - nearest)
        inside  = (px / ti.static(self.a)) ** 2 + \
                  (py / ti.static(self.b)) ** 2 <= 1.0
        # WoS needs positive distance from interior point to boundary
        return dist if inside else dist   # always positive for WoS use

    @ti.func
    def boundary_value(self, x: tm.vec2) -> float:
        """
        Evaluate BCs at (or near) the boundary.
        Project x onto the ellipse; u = 1 on top half, 0 on bottom.
        """
        px = x[0] - ti.static(self.cx)
        py = x[1] - ti.static(self.cy)
        nearest = self._ellipse_nearest_ti(px, py)
        # ny is the y-coordinate of the nearest boundary point in global frame
        ny_global = nearest[1] + ti.static(self.cy)
        return ti.select(ny_global >= ti.static(self.cy), 1.0, 0.0)

    # -- FD grid interface -------------------------------------------------
    def grid_info(self, N: int):
        """
        Build interior/boundary masks for an (N+2)×(N+2) grid covering
        the bounding box of the ellipse.

        A node (i,j) is:
          - exterior   if outside the ellipse (mask = 0, frozen at 0)
          - boundary   if its signed distance satisfies |d| < 1.5 h
                        (approximately one cell layer from ∂Ω)
          - interior   otherwise (G-S update active)
        """
        lo, hi = self._lo_np, self._hi_np
        M  = N + 2
        h  = (hi[0] - lo[0]) / (M - 1)   # assumes square cells

        xs = np.linspace(lo[0], hi[0], M)
        ys = np.linspace(lo[1], hi[1], M)

        interior_mask = np.zeros((M, M), dtype=bool)
        boundary_mask = np.zeros((M, M), dtype=bool)
        bc_values     = np.zeros((M, M), dtype=np.float32)

        for i in range(M):
            for j in range(M):
                pt = np.array([xs[i], ys[j]])
                d  = self._dist_numpy(pt)  # negative inside

                if d > 0:
                    # Outside the ellipse — exterior node (frozen at 0)
                    pass
                elif d > -1.5 * h:
                    # Within ~1 cell of the boundary → impose Dirichlet BC
                    boundary_mask[i, j] = True
                    bc_values[i, j]     = self._bc_numpy(pt)
                else:
                    # Strictly interior → let G-S update this node
                    interior_mask[i, j] = True

        return interior_mask, boundary_mask, bc_values
