import numpy as np
import taichi as ti
import taichi.math as tm

class BaseDomain():
    # Finite Difference
    def bc_numpy(self, x: np.ndarray):
        """
            return bc_type (0 - Dirichlet, 1 - Neumann), bc_value, bc_normal
        """
        raise NotImplementedError

    def source_numpy(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def grid_info(self, N: int):
        """
            return interior_mask, boundary_mask, bc_type, bc_values, bc_normals, source_values
        """
        raise NotImplementedError

    # Walk on Star
    def dist_numpy(self, x: np.ndarray) -> float:
        """
            dist to boundary, used to delete samples outside domain
        """
        raise NotImplementedError

    def dist_to_dirichlet(self, x: tm.vec2) -> float:
        raise NotImplementedError

    def dist_to_silhouette(self, x: tm.vec2) -> float:
        raise NotImplementedError

    def boundary_value(self, x: tm.vec2) -> float:
        """
            Dirichlet boundary value
        """
        raise NotImplementedError

    def source(self, x: tm.vec2) -> float:
        raise NotImplementedError

    def intersect_ray(self, x: tm.vec2, v: tm.vec2, R: float):
        """
            return pos_intersction, on_Neumann, normal_at_hit
        """
        raise NotImplementedError

    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
