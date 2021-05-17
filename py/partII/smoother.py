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

def savefield(f, fieldname, filename):
    f.rename(fieldname)
    print('saving %s to %s' % (fieldname,filename))
    fd.File(filename).write(f)

def savevelocitypressure(u, p, filename):
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
        # we store the basemesh info and the bed elevation
        self.basemesh = None
        self.mx = None
        self.b = None

    def solvestokes(self, mesh, printsizes=False):
        '''Solve the Glen-Stokes problem on the input extruded mesh.
        Returns the separate velocity and pressure solutions.'''

        # set up mixed method for Stokes dynamics problem
        V = fd.VectorFunctionSpace(mesh, 'Lagrange', 2)
        W = fd.FunctionSpace(mesh, 'Lagrange', 1)
        if printsizes:
            print('            sizes: n_u = %d, n_p = %d' % (V.dim(), W.dim()))
        Z = V * W
        up = fd.Function(Z)
        scu, p = fd.split(up)       # scaled velocity, unscaled pressure
        v, q = fd.TestFunctions(Z)

        # symmetrically-scaled Glen-Stokes weak form
        fbody = fd.Constant((0.0, - self.rhoi * self.g))
        sc = self.sc
        reg = self.args.eps * self.Dtyp**2
        Du2 = 0.5 * fd.inner(D(scu * sc), D(scu * sc)) + reg
        assert self.nglen == 3.0
        nu = 0.5 * self.B3 * Du2**((1.0 / self.nglen - 1.0)/2.0)
        F = ( sc*sc * fd.inner(2.0 * nu * D(scu), D(v)) \
              - sc * p * fd.div(v) - sc * q * fd.div(scu) \
              - sc * fd.inner(fbody, v) ) * fd.dx

        # zero Dirichlet on base (and stress-free on top and cliffs)
        bcs = [ fd.DirichletBC(Z.sub(0), fd.Constant((0.0, 0.0)), 'bottom')]

        # solve Stokes
        par = {'snes_linesearch_type': 'bt',
               'snes_max_it': 200,
               'snes_stol': 0.0,       # expect CONVERGED_FNORM_RELATIVE
               'ksp_type': 'preonly',
               'pc_type': 'lu',
               'pc_factor_shift_type': 'inblocks'}
        fd.solve(F == 0, up, bcs=bcs, options_prefix='s', solver_parameters=par)

        # split, descale, and return
        u, p = up.split()
        u *= sc
        return u, p

    def residual(self, mesh1d, s, ella, icefreecolumns=True, saveupname=None):
        '''Compute the residual functional, namely the surface kinematical
        residual for the entire domain, for a given iterate s.  Note mesh1D is
        a MeshLevel1D instance and ella(.) = <a(x),.> is a source term in V^j'.
        This residual evaluation requires setting up an (x,z) Firedrake mesh,
        starting from a (stored) base mesh and then using extrusion.  The icy
        columns get their height from s, with minimum height self.Hmin.  If
        icefreecolumns=True then the extruded mesh has empty (0-element) columns
        in each location which is ice free according to s.  If saveupname is
        a string then the Stokes solution (u,p) is saved to that file.  The
        returned residual array is defined on mesh1d and in V^j'.'''

        # if needed, generate self.basemesh from mesh1d
        firstcall = (self.basemesh == None)
        if firstcall:
            self.mx = mesh1d.m + 1
            print('residual(): base mesh of %d elements (intervals)' \
                  % self.mx)
            self.basemesh = fd.IntervalMesh(self.mx, length_or_left=0.0,
                                            right=mesh1d.xmax)
            self.b = self.phi(mesh1d.xx())

        # extrude the mesh, to total height 1
        mz = self.args.mz
        if icefreecolumns:
            layermap = np.zeros((self.mx, 2), dtype=int)  # [[0,0], ..., [0,0]]
            thk = s - self.b
            thkelement = ( (thk[:-1]) + (thk[1:]) ) / 2.0
            icyelement = (thkelement > self.Hmin)
            layermap[:,1] = mz * np.array(icyelement, dtype=int)
            if firstcall:
                icycount = sum(icyelement)
                print('            extruded mesh of %d x %d icy and %d ice-free elements' \
                      % (icycount, mz, self.mx - icycount))
            # FIXME: in parallel we must provide local, haloed layermap
            mesh = fd.ExtrudedMesh(self.basemesh, layers=layermap,
                                   layer_height=1.0 / mz)
        else:
            if firstcall:
                print('            extruded mesh of %d x %d elements' \
                      % (self.mx, mz))
            mesh = fd.ExtrudedMesh(self.basemesh, mz, layer_height=1.0 / mz)

        # adjust element height in extruded mesh to proportional to s(x)
        P1base = fd.FunctionSpace(self.basemesh, 'Lagrange', 1)
        sbase = fd.Function(P1base)
        sbase.dat.data[:] = np.maximum(s, self.Hmin)
        x, z = fd.SpatialCoordinate(mesh)
        xxzz = fd.as_vector([x, extend(mesh, sbase) * z])
        coords = fd.Function(mesh.coordinates.function_space())
        mesh.coordinates.assign(coords.interpolate(xxzz))

        # solve the Glen-Stokes problem on the extruded mesh
        u, p = self.solvestokes(mesh, printsizes=firstcall)
        if saveupname is not None:
            savevelocitypressure(u, p, saveupname)

        # FIXME if icefreecolumns=True then x coordinate is invalid (and set to
        # zero) for bottombc.nodes in columns with no ice

        # identify snodes, the nodes of the Q1 mesh which are on tops of
        #     columns, i.e. at  z = s(x)
        Q1 = fd.FunctionSpace(mesh, 'Lagrange', 1)
        topbc = fd.DirichletBC(Q1, 1.0, 'top')
        if icefreecolumns:
            # when icefreecolumns == True, topbc.nodes has two issues:
            #    1. includes nodes on facets where there is no adjacent ice
            #    2. does not include 'bottom' nodes which have no icy nodes
            #       above them
            # the strategy here is to union topbc.nodes and bottombc.nodes
            # and then take last node from runs of duplicate x-coordinate
            # nodes; this assumes each column of nodes has contiguous indices
            # and increasing node order
            x, _ = fd.SpatialCoordinate(mesh)
            xx = fd.Function(Q1).interpolate(x)
            bottombc = fd.DirichletBC(Q1, 1.0, 'bottom')
            tbnodes = list(set(topbc.nodes) | set(bottombc.nodes))
            print(topbc.nodes)
            print(xx.dat.data_ro[topbc.nodes])
            print(bottombc.nodes)
            print(xx.dat.data_ro[bottombc.nodes])
            print(tbnodes)
            #print(xx.dat.data_ro[tbnodes])
            snodes = []
            xj = xx.dat.data_ro[tbnodes[0]]
            for j in range(len(tbnodes) - 1):
                xnext = xx.dat.data_ro[tbnodes[j+1]]
                #print('compare xj=%d to xj+1=%d' % (xj, xnext))
                if xnext != xj:
                    snodes.append(tbnodes[j])
                    xj = xnext
            snodes.append(tbnodes[-1])
        else:
            snodes = topbc.nodes
        #print(snodes)

        # evaluate top surface kinematical residual at snodes, i.e. on
        #   z = s(x); note n_s = <-s_x,1> is normal
        res_ufl = + u[0] * z.dx(0) - u[1]   # a=0 residual
        res = fd.Function(Q1).interpolate(res_ufl)  # ... on (x,z) mesh
        # FIXME temporary output to see (x,z) residual field even if "- ella" causes error
        if saveupname is not None:
            savefield(res, 'a=0 residual', 'a0res_' + saveupname)
        return mesh1d.ellf(res.dat.data_ro[snodes]) - ella

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
