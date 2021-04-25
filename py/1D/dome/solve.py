#!/usr/bin/env python3
# (C) 2021 Ed Bueler

# TODO:
#   * add other solver packages from mccarthy/stokes/momentummodel.py

import sys
import numpy as np
from domain import bdryids
from profile import profile, profiledefaults
from firedrake import *

# Regarding element choice:  The first four are recommended.  The first three
# are Taylor-Hood.  Of the TH, P2P1 is fastest and P4P3 slowest.  The last
# three exhibit visible instability on coarser meshes.  CRdP0
mixFEchoices = ['P2P1','P3P2','P4P3','P2dP1','P2dP0','CRdP0','P1dP0']

def processopts():
    import argparse
    parser = argparse.ArgumentParser(description=
    '''Solve the Glen-Stokes momentum equations for a 2D ice sheet.  First
generate a .msh file, e.g. dome.msh, by using domain.py.  Then do:
  $ ./solve.py -mesh dome.msh
Consider adding options -s_snes_converged_reason, -s_snes_monitor,
-s_ksp_converged_reason, -s_snes_rtol, etc. to monitor and control the run.''',
    formatter_class=argparse.RawTextHelpFormatter,
    add_help=False)
    adda = parser.add_argument
    adda('-elements', metavar='X', default='P2P1', choices=mixFEchoices,
         help='mixed finite element: %s (default=P2P1)' \
              % (','.join(mixFEchoices)))
    adda('-extrude', action='store_true', default=False,
         help='benerate the profile mesh by extrusion (vs reading from file)')
    adda('-mesh', metavar='FILE.msh', default='',
         help='input file name')
    adda('-mx', type=int, metavar='MX', default=10,
         help='number of subintervals (applies with -extrude)')
    adda('-mz', type=int, metavar='MZ', default=3,
         help='number of levels (applies with -extrude)')
    adda('-o', metavar='FILE.pvd', default='',
         help='output file name')
    adda('-profile_H', type=float, metavar='X',
         default=profiledefaults['H'],
         help='dome height (applies with -extrude)')
    adda('-profile_R', type=float, metavar='X',
         default=profiledefaults['R'],
         help='dome radius (applies with -extrude)')
    #adda('-refine', type=float, default=1.0, metavar='X',
    #     help='refine resolution by this factor (default=1)')
    adda('-solvehelp', action='store_true', default=False,
         help='print help for solve.py options and stop')
    args, unknown = parser.parse_known_args()
    if len(args.mesh) > 0 and args.extrude:
        print('usage ERROR: use either -extrude or -mesh FILE but not both')
        sys.exit(1)
    if args.elements == 'CRdP0' and args.extrude:
        print('usage ERROR: unsupported element type with -extrude')
        sys.exit(2)
    if args.solvehelp:
        parser.print_help()
        sys.exit(0)
    return args

def printpar(thestr, comm=COMM_WORLD, indent=0, end='\n'):
    spaces = indent * '  '
    PETSc.Sys.Print('%s%s' % (spaces, thestr), comm=comm, end=end)

def extend(mesh, f):
    '''On an extruded mesh extend a function f(x,z), already defined on the
    base mesh, to the mesh using the 'R' constant-in-the-vertical space.'''
    Q1R = FunctionSpace(mesh, 'P', 1, vfamily='R', vdegree=0)
    fextend = Function(Q1R)
    fextend.dat.data[:] = f.dat.data_ro[:]
    return fextend

def extrudeprofile(mx, mz, L=None, xc=None, H=None, R=None):
    '''Set up base mesh of intervals on [0,L] and extrude to a profile of
    a dome as a 2D mesh in (x,z).'''
    # create extruded mesh with temporary height
    base_mesh = IntervalMesh(mx, length_or_left=0.0, right=L)
    mesh = ExtrudedMesh(base_mesh, layers=mz, layer_height=1.0/mz)
    x, z = SpatialCoordinate(mesh)
    # correct the height to equal the profile
    xbase = base_mesh.coordinates.dat.data_ro
    P1base = FunctionSpace(base_mesh,'P',1)
    zz = Function(P1base)
    _, zz.dat.data[:] = profile(mx, R=R, H=H)
    Vcoord = mesh.coordinates.function_space()
    XZ = Function(Vcoord).interpolate(as_vector([x, extend(mesh, zz) * z]))
    mesh.coordinates.assign(XZ)
    return mesh

def describe(thismesh, extruded=False, mz=None, indent=0):
    if extruded:
        printpar('extruded mesh of %d levels over base ' % mz, end='')
    if thismesh.comm.size == 1:
        printpar('mesh with %d elements and %d vertices' \
                 % (thismesh.num_cells(),thismesh.num_vertices()),
                 indent=indent)
    else:
        str = 'mesh: rank %d owns %d elements and can access %d vertices' \
              % (thismesh.comm.rank, thismesh.num_cells(),
                 thismesh.num_vertices())
        PETSc.Sys.syncPrint(str, comm=thismesh.comm)
        PETSc.Sys.syncFlush(comm=thismesh.comm)

def create_mixed_space(mesh, mixedtype):
    if   mixedtype == 'P2P1': # Taylor-Hood
        V = VectorFunctionSpace(mesh, 'CG', 2)
        W = FunctionSpace(mesh, 'CG', 1)
    elif mixedtype == 'P3P2': # Taylor-Hood
        V = VectorFunctionSpace(mesh, 'CG', 3)
        W = FunctionSpace(mesh, 'CG', 2)
    elif mixedtype == 'P4P3': # Taylor-Hood
        V = VectorFunctionSpace(mesh, 'CG', 4)
        W = FunctionSpace(mesh, 'CG', 3)
    elif mixedtype == 'P2dP1':
        V = VectorFunctionSpace(mesh, 'CG', 2)
        if args.extrude:
            W = FunctionSpace(mesh, 'DQ', 1)
        else:
            W = FunctionSpace(mesh, 'DG', 1)
    elif mixedtype == 'P2dP0':
        V = VectorFunctionSpace(mesh, 'CG', 2)
        W = FunctionSpace(mesh, 'DG', 0)
    elif mixedtype == 'CRdP0':
        V = VectorFunctionSpace(mesh, 'CR', 1)
        W = FunctionSpace(mesh, 'DG', 0)
    elif mixedtype == 'P1dP0':
        V = VectorFunctionSpace(mesh, 'CG', 1)
        W = FunctionSpace(mesh, 'DG', 0)
    else:
        print('ERROR: unknown mixed type')
        sys.exit(1)
    Z = V * W
    return V, W, Z

def solutionstats(mesh, u, p):
    '''Statistics about the solution:
        umagav  = average velocity magnitude
        umagmax = maximum velocity magnitude
        pav     = average pressure
        pmax    = maximum pressure'''
    P1 = FunctionSpace(mesh, 'CG', 1)
    one = Constant(1.0, domain=mesh)
    area = assemble(dot(one,one) * dx)
    pav = assemble(sqrt(dot(p, p)) * dx) / area
    with p.dat.vec_ro as vp:
        pmax = vp.max()[1]
    umagav = assemble(sqrt(dot(u, u)) * dx) / area
    umag = interpolate(sqrt(dot(u, u)), P1)
    with umag.dat.vec_ro as vumag:
        umagmax = vumag.max()[1]
    return umagav, umagmax, pav, pmax

def D(U):    # strain-rate tensor from velocity U
    return 0.5 * (grad(U) + grad(U).T)

secpera = 31556926.0    # seconds per year

def weakform(mesh):
    # physical constants
    g = 9.81                # m s-2
    rho = 910.0             # kg m-3
    n_glen = 3.0
    A3 = 3.1689e-24         # Pa-3 s-1;  EISMINT I value of ice softness
    B3 = A3**(-1.0/3.0)     # Pa s(1/3);  ice hardness
    eps = 0.01
    Dtyp = 2.0 / secpera    # s-1

    # body force
    f_body = Constant((0.0, - rho * g))

    # function spaces
    _, _, Z = create_mixed_space(mesh, args.elements)
    up = Function(Z)
    u,p = split(up)  # get component ufl expressions to define form

    # nonlinear weak form F(u,p;v,q)
    v,q = TestFunctions(Z)
    Du2 = 0.5 * inner(D(u), D(u)) + (eps * Dtyp)**2.0
    rr = 1.0 / n_glen - 1.0
    F = ( inner(B3 * Du2**(rr/2.0) * D(u), D(v)) \
          - p * div(v) - div(u) * q - inner(f_body, v) ) * dx

    noslip = Constant((0.0, 0.0))
    if bdryids is not None:
        # Gmsh case: u=0 on base
        bcs = [ DirichletBC(Z.sub(0), noslip, bdryids['base']) ]
    else:
        # extruded case: u=0 on base and degenerate ends
        bcs = [ DirichletBC(Z.sub(0), noslip, 'bottom'),
                DirichletBC(Z.sub(0), noslip, 1),
                DirichletBC(Z.sub(0), noslip, 2)]
    return up, F, bcs

SchurDirect = {"ksp_type": "fgmres",  # or "gmres" or "minres"
      "pc_type": "fieldsplit",
      "pc_fieldsplit_type": "schur",
      "pc_fieldsplit_schur_factorization_type": "full",  # or "diag"
      "pc_fieldsplit_schur_precondition": "a11",  # the default
      "fieldsplit_0_ksp_type": "preonly",
      "fieldsplit_0_pc_type": "lu",  # uses mumps in parallel
      "fieldsplit_1_ksp_rtol": 1.0e-3,
      "fieldsplit_1_ksp_type": "gmres",
      "fieldsplit_1_pc_type": "none"}

if __name__ == "__main__":
    args = processopts()
    if args.extrude:
        R = args.profile_R
        mesh = extrudeprofile(args.mx, args.mz, L=2.0*R, xc=R, R=R, H=args.profile_H)
        bdryids = None
    else:
        printpar('reading mesh from %s ...' % args.mesh)
        mesh = Mesh(args.mesh)
    describe(mesh, extruded=args.extrude, mz=args.mz)
    up, F, bcs = weakform(mesh)
    printpar('solving ...')
    params = SchurDirect
    params['snes_linesearch_type'] = 'bt'
    solve(F == 0, up, bcs=bcs, options_prefix='s',
          solver_parameters=params)
    u, p = up.split()
    u.rename('velocity')
    p.rename('pressure')
    umagav, umagmax, pav, pmax = solutionstats(mesh, u, p)
    printpar('flow speed: av = %10.3f m a-1,  max = %10.3f m a-1' \
             % (secpera* umagav, secpera * umagmax))
    printpar('pressure:   av = %10.3f bar,    max = %10.3f bar' \
             % (1.0e-5 * pav, 1.0e-5 * pmax))
    if len(args.o) > 0:
        printpar('writing u,p to %s ...' % args.o)
        outfile = File(args.o)
        outfile.write(u,p)
