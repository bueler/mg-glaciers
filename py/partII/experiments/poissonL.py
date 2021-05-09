#!/usr/bin/env python3

# FIXME this code relates to a suggestion by Colin Cotter, not followed up,
# about subclassing DirichletBC to allow interior imposition of trivialized
# equations, like Dirichlet boundary conditions area handled

from firedrake import *

class InteriorBC(DirichletBC):
    #@cached_property
    def nodes(self):
        dm = self.function_space().mesh().topology_dm
        section = self.function_space().dm.getDefaultSection()
        nodes = []
        for sd in as_tuple(self.sub_domain):
            nfaces = dm.getStratumSize(FACE_SETS_LABEL, sd)
            faces = dm.getStratumIS(FACE_SETS_LABEL, sd)
            if nfaces == 0:
                continue
            for face in faces.indices:
                if dm.getLabelValue("interior_facets", face) < 0:
                    continue
                closure, _ = dm.getTransitiveClosure(face)
                for p in closure:
                    dof = section.getDof(p)
                    offset = section.getOffset(p)
                    nodes.extend((offset + d) for d in range(dof))
        return np.unique(np.asarray(nodes, dtype=IntType))

mesh = UnitSquareMesh(8, 8)

W = FunctionSpace(mesh, 'Lagrange', degree=1)
f_rhs = Constant(1.0)
u = Function(W)
v = TestFunction(W)
F = (dot(grad(u), grad(v)) - f_rhs * v) * dx

g_bdry = Constant(0.0)
bdry_ids = (1, 2, 3, 4)   # all four sides of boundary
bc = DirichletBC(W, g_bdry, bdry_ids)

# substituting this throws error; it does not have same calling signature as DirichletBC()?
# bc = InteriorBC(W, g_bdry, bdry_ids)

solve(F == 0, u, bcs = [bc], options_prefix = 's',
      solver_parameters = {'snes_type': 'ksponly',
                           'ksp_type': 'preonly',
                           'pc_type': 'cholesky'})

solnnorm = sqrt(assemble(dot(u,u) * dx))
PETSc.Sys.Print('solution norm |u|_2 = %.3e' % solnnorm)
u.rename('u')
File('soln.pvd').write(u)
