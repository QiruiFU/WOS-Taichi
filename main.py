import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)

from fd    import FDSolver,  visualise as fd_vis
from WoSt   import WoStSolver, visualise as wost_vis
from domains.Dirichlet import SquareDirichlet
from domains.Neumann import SquareNeumann, CircleNeumann
from domains.source import SquareSource, CircleSource

domain_dirichlet = SquareDirichlet()
domain_neumann_square = SquareNeumann()
domain_neumann_circle = CircleNeumann()
domain_source_square = SquareSource()
domain_source_circle = CircleSource()

# ---------------------------------------------------------------------------
# FD solve
# ---------------------------------------------------------------------------
# fd_solver = FDSolver(domain=domain_dirichlet, N=256)
# fd_solver.solve(max_iters=200_000, tol=1e-3, check_every=5000)
# fd_vis(fd_solver, title="FD — Square domain", save_path="./img/fd_dirichlet.png")

# fd_solver = FDSolver(domain=domain_neumann_square, N=256)
# fd_solver.solve(max_iters=200_000, tol=1e-3, check_every=5000)
# fd_vis(fd_solver, title="FD — Square domain", save_path="./img/fd_neumann_square.png")

# fd_solver = FDSolver(domain=domain_neumann_circle, N=256)
# fd_solver.solve(max_iters=200_000, tol=1e-3, check_every=5000)
# fd_vis(fd_solver, title="FD — Circle domain", save_path="./img/fd_neumann_circle.png")

# fd_solver = FDSolver(domain=domain_source_square, N=256)
# fd_solver.solve(max_iters=200_000, tol=1e-3, check_every=5000)
# fd_vis(fd_solver, title="FD — Square domain", save_path="./img/fd_source_square.png")

fd_solver = FDSolver(domain=domain_source_circle, N=256)
fd_solver.solve(max_iters=200_000, tol=1e-3, check_every=5000)
fd_vis(fd_solver, title="FD — Circle domain", save_path="./img/fd_source_circle.png", v_min=-1.0, v_max=1.5)

# ---------------------------------------------------------------------------
# WoSt solve
# ---------------------------------------------------------------------------
# wost_solver = WoStSolver(domain=domain_dirichlet, dx = 1 / 256,
#                        n_walks=8000, epsilon=1e-4, max_steps=10000)
# wost_solver.solve(check_every=1000)
# wost_vis(wost_solver, title="WoSt — Circle domain", save_path="./img/WoSt_dirichlet.png")

# wost_solver = WoStSolver(domain=domain_neumann_square, dx = 1 / 256,
#                        n_walks=8000, epsilon=1e-4, max_steps=10000)
# wost_solver.solve(check_every=1000)
# wost_vis(wost_solver, title="WoSt — Square domain", save_path="./img/WoSt_neumann_square.png")

# wost_solver = WoStSolver(domain=domain_neumann_circle, dx = 1 / 256,
#                        n_walks=8000, epsilon=1e-4, max_steps=10000)
# wost_solver.solve(check_every=1000)
# wost_vis(wost_solver, title="WoSt — Circle domain", save_path="./img/WoSt_neumann_circle.png")

# wost_solver = WoStSolver(domain=domain_source_square, dx = 1 / 256,
#                        n_walks=8000, epsilon=1e-4, max_steps=10000)
# wost_solver.solve(check_every=1000)
# wost_vis(wost_solver, title="WoSt — Square domain", save_path="./img/WoSt_source_square.png")

wost_solver = WoStSolver(domain=domain_source_circle, dx = 1 / 256,
                       n_walks=8000, epsilon=1e-4, max_steps=10000)
wost_solver.solve(check_every=1000)
wost_vis(wost_solver, title="WoSt — Circle domain", save_path="./img/WoSt_source_circle.png", v_min=-1.0, v_max=1.5)