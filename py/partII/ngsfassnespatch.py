#!/usr/bin/env python3

# copied from firedrake/src/firedrake/tests/regression/test_fas_snespatch.py
# pytest and pytest.fixture stuff removed

from firedrake import *

def solver_params_fcn(param):
    if param == 0:
        return {
               "mat_type": "matfree",  # noqa: E126
               "snes_type": "fas",
               "snes_fas_cycles": 1,
               "snes_fas_type": "full",
               "snes_fas_galerkin": False,
               "snes_fas_smoothup": 1,
               "snes_fas_smoothdown": 1,
               "snes_monitor": None,
               #"snes_view": None,
               "snes_max_it": 20,
               "fas_levels_snes_type": "python",
               "fas_levels_snes_python_type": "firedrake.PatchSNES",
               "fas_levels_snes_max_it": 1,
               "fas_levels_snes_convergence_test": "skip",
               "fas_levels_snes_converged_reason": None,
               "fas_levels_snes_monitor": None,
               "fas_levels_snes_linesearch_type": "basic",
               "fas_levels_snes_linesearch_damping": 4/5,
               "fas_levels_patch_snes_patch_partition_of_unity": False,
               "fas_levels_patch_snes_patch_construct_type": "star",
               "fas_levels_patch_snes_patch_construct_dim": 0,
               "fas_levels_patch_snes_patch_sub_mat_type": "seqdense",
               "fas_levels_patch_snes_patch_local_type": "additive",
               "fas_levels_patch_snes_patch_symmetrise_sweep": False,
               "fas_levels_patch_sub_snes_type": "newtonls",
               "fas_levels_patch_sub_snes_converged_reason": None,
               "fas_levels_patch_sub_snes_linesearch_type": "basic",
               "fas_levels_patch_sub_ksp_type": "preonly",
               "fas_levels_patch_sub_pc_type": "lu",
               "fas_coarse_snes_type": "newtonls",
               "fas_coarse_snes_monitor": None,
               "fas_coarse_snes_converged_reason": None,
               "fas_coarse_snes_max_it": 100,
               "fas_coarse_snes_atol": 1.0e-14,
               "fas_coarse_snes_rtol": 1.0e-14,
               "fas_coarse_snes_linesearch_type": "l2",
               "fas_coarse_ksp_type": "preonly",
               "fas_coarse_ksp_max_it": 1,
               "fas_coarse_pc_type": "python",
               "fas_coarse_pc_python_type": "firedrake.AssembledPC",
               "fas_coarse_assembled_mat_type": "aij",
               "fas_coarse_assembled_pc_type": "lu",
               "fas_coarse_assembled_pc_factor_mat_solver_type": "mumps",
               "fas_coarse_assembled_mat_mumps_icntl_14": 200,
        }
    elif param == 1:
        return {
               "mat_type": "matfree",  # noqa: E126
               "snes_type": "fas",
               "snes_fas_cycles": 1,
               "snes_fas_type": "full",
               "snes_fas_galerkin": False,
               "snes_fas_smoothup": 1,
               "snes_fas_smoothdown": 1,
               "snes_monitor": None,
               "snes_view": None,
               "snes_max_it": 20,
               "fas_levels_snes_type": "python",
               "fas_levels_snes_python_type": "firedrake.PatchSNES",
               "fas_levels_snes_max_it": 1,
               "fas_levels_snes_convergence_test": "skip",
               "fas_levels_snes_converged_reason": None,
               "fas_levels_snes_monitor": None,
               "fas_levels_snes_linesearch_type": "basic",
               "fas_levels_snes_linesearch_damping": 4/5,
               "fas_levels_patch_snes_patch_partition_of_unity": False,
               "fas_levels_patch_snes_patch_construct_type": "vanka",
               "fas_levels_patch_snes_patch_construct_dim": 0,
               "fas_levels_patch_snes_patch_vanka_dim": 0,
               "fas_levels_patch_snes_patch_sub_mat_type": "seqdense",
               "fas_levels_patch_snes_patch_local_type": "additive",
               "fas_levels_patch_snes_patch_symmetrise_sweep": False,
               "fas_levels_patch_sub_snes_type": "newtonls",
               "fas_levels_patch_sub_snes_converged_reason": None,
               "fas_levels_patch_sub_snes_linesearch_type": "basic",
               "fas_levels_patch_sub_ksp_type": "preonly",
               "fas_levels_patch_sub_pc_type": "lu",
               "fas_coarse_snes_type": "newtonls",
               "fas_coarse_snes_monitor": None,
               "fas_coarse_snes_converged_reason": None,
               "fas_coarse_snes_max_it": 100,
               "fas_coarse_snes_atol": 1.0e-14,
               "fas_coarse_snes_rtol": 1.0e-14,
               "fas_coarse_snes_linesearch_type": "l2",
               "fas_coarse_ksp_type": "preonly",
               "fas_coarse_ksp_max_it": 1,
               "fas_coarse_pc_type": "python",
               "fas_coarse_pc_python_type": "firedrake.AssembledPC",
               "fas_coarse_assembled_mat_type": "aij",
               "fas_coarse_assembled_pc_type": "lu",
               "fas_coarse_assembled_pc_factor_mat_solver_type": "mumps",
               "fas_coarse_assembled_mat_mumps_icntl_14": 200,
        }
    else:
        return {
               "mat_type": "matfree",  # noqa: E126
               "snes_type": "fas",
               "snes_fas_cycles": 1,
               "snes_fas_type": "full",
               "snes_fas_galerkin": False,
               "snes_fas_smoothup": 1,
               "snes_fas_smoothdown": 1,
               "snes_fas_full_downsweep": False,
               "snes_monitor": None,
               "snes_max_it": 20,
               "fas_levels_snes_type": "python",
               "fas_levels_snes_python_type": "firedrake.PatchSNES",
               "fas_levels_snes_max_it": 1,
               "fas_levels_snes_convergence_test": "skip",
               "fas_levels_snes_converged_reason": None,
               "fas_levels_snes_monitor": None,
               "fas_levels_snes_linesearch_type": "basic",
               "fas_levels_snes_linesearch_damping": 1.0,
               "fas_levels_patch_snes_patch_construct_type": "pardecomp",
               "fas_levels_patch_snes_patch_partition_of_unity": True,
               "fas_levels_patch_snes_patch_pardecomp_overlap": 1,
               "fas_levels_patch_snes_patch_sub_mat_type": "seqaij",
               "fas_levels_patch_snes_patch_local_type": "additive",
               "fas_levels_patch_snes_patch_symmetrise_sweep": False,
               "fas_levels_patch_sub_snes_type": "newtonls",
               "fas_levels_patch_sub_snes_monitor": None,
               "fas_levels_patch_sub_snes_converged_reason": None,
               "fas_levels_patch_sub_snes_linesearch_type": "basic",
               "fas_levels_patch_sub_ksp_type": "preonly",
               "fas_levels_patch_sub_pc_type": "lu",
               "fas_coarse_snes_type": "newtonls",
               "fas_coarse_snes_monitor": None,
               "fas_coarse_snes_converged_reason": None,
               "fas_coarse_snes_max_it": 100,
               "fas_coarse_snes_atol": 1.0e-14,
               "fas_coarse_snes_rtol": 1.0e-14,
               "fas_coarse_snes_linesearch_type": "l2",
               "fas_coarse_ksp_type": "preonly",
               "fas_coarse_ksp_max_it": 1,
               "fas_coarse_pc_type": "python",
               "fas_coarse_pc_python_type": "firedrake.AssembledPC",
               "fas_coarse_assembled_mat_type": "aij",
               "fas_coarse_assembled_pc_type": "lu",
               "fas_coarse_assembled_pc_factor_mat_solver_type": "mumps",
               "fas_coarse_assembled_mat_mumps_icntl_14": 200
        }

N = 10
nref = 1
distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
base = UnitSquareMesh(N, N, distribution_parameters=distribution_parameters)
mh = MeshHierarchy(base, nref, distribution_parameters=distribution_parameters)
mesh = mh[-1]

CG1 = FunctionSpace(mesh, "CG", 1)

u = Function(CG1)
v = TestFunction(CG1)

f = Constant(1)
F = inner(grad(u), grad(v))*dx - inner(f, v)*dx + inner(u**3 - u, v)*dx

z = zero(CG1.ufl_element().value_shape())
bcs = DirichletBC(CG1, z, "on_boundary")

nvproblem = NonlinearVariationalProblem(F, u, bcs=bcs)

for p in [0,]:
    solver_params = solver_params_fcn(p)
    solver = NonlinearVariationalSolver(nvproblem, solver_parameters=solver_params)
    solver.solve()
    assert solver.snes.reason > 0
