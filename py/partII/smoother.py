'''Module for SmootherStokes class derived from SmootherObstacleProblem.'''

import numpy as np
import firedrake as fd
from basesmoother import SmootherObstacleProblem

def extend(mesh, f):
    '''On an extruded mesh extend a function f(x,z), already defined on the
    base mesh, to the mesh using the 'R' constant-in-the-vertical space.'''
    Q1R = fd.FunctionSpace(mesh, 'P', 1, vfamily='R', vdegree=0)
    fextend = fd.Function(Q1R)
    fextend.dat.data[:] = f.dat.data_ro[:]
    return fextend

def saveuflfield(mesh, field, fieldname, filename):
    Q1 = fd.FunctionSpace(mesh,'Lagrange',1)
    f = fd.Function(Q1).interpolate(field)
    f.rename(fieldname)
    print('saving %s to %s' % (fieldname,filename))
    fd.File(filename).write(f)

def saveup(u, p, filename):
    u.rename('velocity')
    p.rename('pressure')
    print('saving u,p to %s' % filename)
    fd.File(filename).write(u,p)

def D(w):
    return 0.5 * (fd.grad(w) + fd.grad(w).T)

class SmootherStokes(SmootherObstacleProblem):
    '''To evaluate the residual this Jacobi smoother solves the Stokes problem
    for the given geometry by creating an extruded mesh.'''

    def __init__(self, args, admissibleeps=1.0e-10):
        super().__init__(args, admissibleeps=admissibleeps)
        # smoother name
        self.name = 'SmootherStokes'
        # physical parameters
        self.secpera = 31556926.0        # seconds per year
        self.g = 9.81                    # m s-2
        self.rhoi = 910.0                # kg m-3
        self.nglen = 3.0
        self.A3 = 1.0e-16 / self.secpera # Pa-3 s-1;  EISMINT I ice softness
        self.B3 = self.A3**(-1.0/3.0)    # Pa s(1/3);  ice hardness
        # used in Stokes solver
        self.Hmin = 20.0                 # 20 m cliffs on ends
        self.Dtyp = 1.0 / self.secpera   # s-1
        self.sc = 1.0e-7                 # velocity scale for symmetric scaling
        # parameters for initial condition, a Bueler profile; see van der Veen
        #   (2013) section 5.3
        self.buelerL = 10.0e3       # half-width of sheet
        self.buelerH0 = 1000.0      # center thickness
        self.Gamma = 2.0 * self.A3 * (self.rhoi * self.g)**self.nglen
        self.Gamma /= (self.nglen + 2.0)

    def residual(self, mesh1d, s, ell):
        '''Compute the residual functional, namely the surface kinematical
        residual for the entire domain, for a given iterate s.  Note mesh1D is
        a MeshLevel1D instance and ell is a source term in V^j'.  This residual
        evaluation requires setting up an (x,z) Firedrake mesh, starting from
        mesh1d to make the base mesh and then using extrusion.  The extruded mesh has one short element in each ice free (according to s) column,
        and the icy columns get their height from s.
        Note returned residual array is defined on mesh1d.'''
        # generate base mesh from mesh1d
        mx = mesh1d.m + 1
        print('  base mesh: %d elements (intervals)' % mx)
        basemesh = fd.IntervalMesh(mx, length_or_left=0.0, right=mesh1d.xmax)
        # get bed elevation
        b = self.phi(mesh1d.xx())
        # mark ice free columns; put one element column there
        # FIXME PROBABLY SHOULD JUST GO WITH UNIFORM LAYER COUNTS AND SET TOTAL
        #       HEIGHT SMALL WHERE ICE FREE
        layermap = np.zeros((mx, 2), dtype=int)  # [[0,0], [0,0], ..., [0,0]]
        thkelement = ( (s[:-1] - b[:-1]) + (s[1:] - b[1:]) ) / 2.0
        icyelement = (thkelement > self.Hmin)
        #self.shownonzeros(icyelement)
        mz = self.args.mz
        layermap[:,1] = (mz - 1) * np.array(icyelement, dtype=int) + 1
        # extrude the mesh
        icycount = sum(icyelement)
        # FIXME: in parallel we must provide local, haloed layermap
        mesh = fd.ExtrudedMesh(basemesh, layers=layermap, layer_height=1.0/mz)
        # adjust height to s(x)
        P1base = fd.FunctionSpace(basemesh, 'Lagrange', 1)
        sbase = fd.Function(P1base)
        sbase.dat.data[:] = np.maximum(s, self.Hmin)
        x, z = fd.SpatialCoordinate(mesh)
        xxzz = fd.as_vector([x, extend(mesh, sbase) * z])
        coords = fd.Function(mesh.coordinates.function_space())
        mesh.coordinates.assign(coords.interpolate(xxzz))
        print('  extruded mesh: %d x %d icy and %d (%.2f m) ice-free elements' \
              % (icycount, mz, mx - icycount, self.Hmin/mz))
        # set up mixed method
        V = fd.VectorFunctionSpace(mesh, 'Lagrange', 2)
        W = fd.FunctionSpace(mesh, 'Lagrange', 1)
        n_u, n_p = V.dim(), W.dim()
        print('  sizes: n_u = %d, n_p = %d' % (n_u,n_p))
        Z = V * W
        up = fd.Function(Z)
        scu, p = fd.split(up)             # scaled velocity, unscaled pressure
        v, q = fd.TestFunctions(Z)
        # symmetrically-scaled Glen-Stokes weak form
        fbody = fd.Constant((0.0, - self.rhoi * self.g))
        sc = self.sc
        reg = self.args.eps * self.Dtyp**2
        Du2 = 0.5 * fd.inner(D(scu * sc), D(scu * sc)) + reg
        nu = 0.5 * self.B3 * Du2**((1.0 / self.nglen - 1.0)/2.0)
        F = ( sc*sc * fd.inner(2.0 * nu * D(scu), D(v)) \
              - sc * p * fd.div(v) - sc * q * fd.div(scu) \
              - sc * fd.inner(fbody, v) ) * fd.dx
        # zero Dirichlet on base (and stress-free on top and degenerate ends)
        bcs = [ fd.DirichletBC(Z.sub(0), fd.Constant((0.0, 0.0)), 'bottom')]
        # solve Stokes
        par = {'snes_linesearch_type': 'bt',
               'snes_max_it': 200,
               'snes_stol': 0.0,         # expect CONVERGED_FNORM_RELATIVE
               'ksp_type': 'preonly',
               'pc_type': 'lu',
               'pc_factor_shift_type': 'inblocks'}
        fd.solve(F == 0, up, bcs=bcs, options_prefix='s', solver_parameters=par)
        u, p = up.split()
        u *= sc
        saveup(u, p, 'soln.pvd')
        # evaluate surface kinematical residual
        res = - u[0] * z.dx(0) + u[1]   # defined on (x,z) mesh; FIXME: add a(x)
        saveuflfield(mesh, res, 'a=0 residual', 'res.pvd')
        P1 = fd.FunctionSpace(mesh, 'Lagrange', 1)
        topbc = fd.DirichletBC(P1, 1.0, 'top')
        print(topbc.nodes)
        bottombc = fd.DirichletBC(P1, 1.0, 'bottom')
        print(bottombc.nodes)
        zz = fd.Function(P1).interpolate(z)
        stop = zz.dat.data_ro[topbc.nodes]
        print(stop)
        rr = fd.Function(P1).interpolate(res)
        rtop = rr.dat.data_ro[topbc.nodes]
        print(rtop)
        #xbaseufl = fd.SpatialCoordinate(basemesh)
        #xbase = fd.Function(P1base).interpolate(xbaseufl[0])
        #print(xbase.dat.data)
        #print(fd.Function(P1).interpolate(x).dat.data)
        #print(fd.Function(P1).interpolate(z).dat.data)
        #print(fd.Function(P1).interpolate(res).dat.data)
        return

    def applyoperator(self, mesh1d, w):
        '''Apply nonlinear operator N to w to get N(w) in (V^j)'.'''
        raise NotImplementedError

    def smoothersweep(self, mesh, y, ell, phi, forward=True):
        '''Do in-place Jacobi smoothing.'''
        raise NotImplementedError

    def phi(self, x):
        '''For now we have a flat bed.'''
        return np.zeros(np.shape(x))

    def source(self, x):
        '''Continuous source term, i.e. mass balance, for Bueler profile.
        See van der Veen (2013) equations (5.49) and (5.51).  Assumes x
        is a numpy array.'''
        n = self.nglen
        invn = 1.0 / n
        r1 = 2.0 * n + 2.0                   # e.g. 8
        s1 = (1.0 - n) / n                   #     -2/3
        C = self.buelerH0**r1 * self.Gamma   # A_0 in van der Veen is Gamma here
        C /= ( 2.0 * self.buelerL * (1.0 - 1.0 / n) )**n
        xc = self.args.domainlength / 2.0
        X = (x - xc) / self.buelerL          # rescaled coord
        m = np.zeros(np.shape(x))
        # usual formula for 0 < |X| < 1
        zzz = (abs(X) > 0.0) * (abs(X) < 1.0)
        if any(zzz):
            Xin = abs(X[zzz])
            Yin = 1.0 - Xin
            m[zzz] = (C / self.buelerL) \
                     * ( Xin**invn + Yin**invn - 1.0 )**(n-1.0) \
                     * ( Xin**s1 - Yin**s1 )
        # fill singular origin with near value
        if any(X == 0.0):
            Xnear = 1.0e-8
            Ynear = 1.0 - Xnear
            m[X == 0.0] = (C / self.buelerL) \
                          * ( Xnear**invn + Ynear**invn - 1.0 )**(n-1.0) \
                          * ( Xnear**s1 - Ynear**s1 )
        # extend by ablation
        if any(abs(X) >= 1.0):
            m[abs(X) >= 1.0] = min(m)
        return m

    def initial(self, x):
        '''Default initial shape is the Bueler profile.  See van der Veen (2013)
        equation (5.50).  Assumes x is a numpy array.'''
        n = self.nglen
        p1 = n / (2.0 * n + 2.0)                  # e.g. 3/8
        q1 = 1.0 + 1.0 / n                        #      4/3
        Z = self.buelerH0 / (n - 1.0)**p1         # outer constant
        xc = self.args.domainlength / 2.0
        X = (x - xc) / self.buelerL               # rescaled coord
        Xin = abs(X[abs(X) < 1.0])                # rescaled distance from
                                                  #   center, in ice
        Yin = 1.0 - Xin
        s = np.zeros(np.shape(x))                 # correct outside ice
        s[abs(X) < 1.0] = Z * ( (n + 1.0) * Xin - 1.0 \
                                + n * Yin**q1 - n * Xin**q1 )**p1
        return s
