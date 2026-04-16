import numpy as np
import taichi as ti
import taichi.math as tm
import math

from .domain import BaseDomain

@ti.data_oriented
class SquareDirichlet(BaseDomain):
    def __init__(self):
        lo = ti.Vector([0.0, 0.0])
        hi = ti.Vector([1.0, 1.0])
        self.lo = lo
        self.hi = hi
        self.lo_np = lo.to_numpy()
        self.hi_np = hi.to_numpy()

    @property
    def bbox(self):
        return self.lo_np, self.hi_np

    # Finite difference
    def bc_numpy(self, x: np.ndarray):
        lo, hi = self.lo_np, self.hi_np
        d_left   = x[0] - lo[0]
        d_right  = hi[0] - x[0]
        d_bottom = x[1] - lo[1]
        d_top    = hi[1] - x[1]
        d_min = min(d_left, d_right, d_bottom, d_top)

        # bc_type, bc_value, bc_normal
        if d_top == d_min:
            return 0, 1.0, np.array([0.0,  1.0], dtype=np.float32)
        elif d_bottom == d_min:
            return 0, -1.0, np.array([0.0, -1.0], dtype=np.float32)
        elif d_left == d_min:
            return 0, -1.0, np.array([-1.0, 0.0], dtype=np.float32)
        else:
            return 0, -1.0, np.array([1.0,  0.0], dtype=np.float32)

    def source_numpy(self, x: np.ndarray) -> float:
        return 0.0

    def grid_info(self, N: int):
        lo, hi = self.lo_np, self.hi_np
        M = N + 2
        xs = np.linspace(lo[0], hi[0], M)
        ys = np.linspace(lo[1], hi[1], M)

        interior_mask = np.zeros((M, M), dtype=np.int32)
        boundary_mask = np.zeros((M, M), dtype=np.int32)
        bc_type       = np.zeros((M, M), dtype=np.int32)
        bc_values     = np.zeros((M, M), dtype=np.float32)
        bc_normals    = np.zeros((M, M, 2), dtype=np.float32)
        source_values = np.zeros((M, M), dtype=np.float32)

        for i in range(M):
            for j in range(M):
                on_edge = (i == 0) or (i == M-1) or (j == 0) or (j == M-1)
                x = np.array([xs[i], ys[j]])
                if on_edge:
                    interior_mask[i, j] = 0
                    boundary_mask[i, j] = 1
                    btype, bval, bnorm  = self.bc_numpy(x)
                    bc_type[i, j]       = btype
                    bc_values[i, j]     = bval
                    bc_normals[i, j]    = bnorm
                else:
                    boundary_mask[i, j] = 0
                    interior_mask[i, j] = 1
                source_values[i, j] = self.source_numpy(x)

        return interior_mask, boundary_mask, bc_type, bc_values, bc_normals, source_values

    # WoSt
    def dist_numpy(self, x: np.ndarray) -> float:
        d_left   = x[0] - self.lo[0]
        d_right  = self.hi[0] - x[0]
        d_bottom = x[1] - self.lo[1]
        d_top    = self.hi[1] - x[1]
        return min(d_left, d_right, d_bottom, d_top)

    @ti.func
    def dist_to_dirichlet(self, x: tm.vec2) -> float:
        d_top    = self.hi[1] - x[1]
        d_bottom = x[1] - self.lo[1]
        d_right  = self.hi[0] - x[0]
        d_left   = x[0] - self.lo[0]
        return ti.min(d_top, d_bottom, d_left, d_right)

    @ti.func
    def dist_to_silhouette(self, x: tm.vec2) -> float:
        return 1e8

    @ti.func
    def boundary_value(self, x: tm.vec2) -> float:
        d_top    = self.hi[1] - x[1]
        d_bottom = x[1] - self.lo[1]
        d_right  = self.hi[0] - x[0]
        d_left   = x[0] - self.lo[0]
        d_min = ti.min(ti.min(d_top, d_bottom), ti.min(d_left, d_right))
        val = -1.0
        if d_top == d_min:
            val = 1.0
        return val

    @ti.func
    def source(self, x: tm.vec2) -> float:
        return 0.0

    @ti.func
    def intersect_ray(self, x: tm.vec2, v: tm.vec2, R: float):
        t_min = R
        on_Neumann = 0
        n_hit = tm.vec2(0.0, 0.0)
        return t_min, on_Neumann, n_hit
