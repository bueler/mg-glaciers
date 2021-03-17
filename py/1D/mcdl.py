'''Module implementing the multilevel constraint decomposition (MCD) method
of the Tai (2003) for the classical obstacle problem (i.e. linear interior PDE).'''

__all__ = ['mcdlvcycle', 'mcdlsolver']

def _indentprint(n, s):
    '''Print 2n spaces and then string s.'''
    for _ in range(n):
        print('  ', end='')
    print(s)

def _levelreport(indent, j, m, sweeps):
    _indentprint(indent - j, 'level %d: %d sweeps over m=%d nodes' \
                             % (j, sweeps, m))

def _coarsereport(indent, m, sweeps):
    _indentprint(indent, 'coarsest: %d sweeps over m=%d nodes' \
                         % (sweeps, m))

def _smoother(obsprob, s, mesh, v, ell, phi, forward=True, symmetric=False):
    infeas = 0
    for _ in range(s):
        infeas += obsprob.smoothersweep(mesh, v, ell, phi, forward=forward)
        if symmetric:
            infeas += obsprob.smoothersweep(mesh, v, ell, phi, forward=not forward)
    return infeas

def mcdlvcycle(args, obsprob, J, hierarchy, ell, levels=None):
    '''Apply one V-cycle of the multilevel constraint decomposition method of
    Tai (2003).  This is stated in Alg. 4.7 in Graeser & Kornhuber (2009)
    as a down-slash V(1,0) cycle.  Our implementation allows any V(down,up)
    cycle.  Input args is a dictionary with parameters.  The smoother is
    projected Gauss-Seidel or projected Jacobi according to argument obsprob.
    Note hierarchy[j] is of type MeshLevel1D.  This method generates
    all defect constraints hierarchy[j].chi.  The input linear functional ell
    is in V^J'.  The coarse solver is the same as the smoother, thus not exact.'''

    # set up
    assert args.down >= 0 and args.up >= 0 and args.coarse >= 0
    assert len(ell) == hierarchy[J].m + 2
    infeas = 0
    hierarchy[J].ell = ell

    # downward
    for k in range(J, 0, -1):
        # compute defect constraint using monotone restriction
        hierarchy[k-1].chi = hierarchy[k].mR(hierarchy[k].chi)
        # define down-obstacle
        phi = hierarchy[k].chi - hierarchy[k].cP(hierarchy[k-1].chi)
        # down smoother
        if args.mgview:
            _levelreport(levels-1, k, hierarchy[k].m, args.down)
        hierarchy[k].y = hierarchy[k].zeros()
        infeas += _smoother(obsprob, args.down, hierarchy[k], hierarchy[k].y,
                            hierarchy[k].ell, phi,
                            symmetric=args.symmetric)
        # update and canonically-restrict the residual
        hierarchy[k-1].ell = - hierarchy[k].cR(obsprob.residual(hierarchy[k],
                                                                hierarchy[k].y,
                                                                hierarchy[k].ell))

    # coarse mesh solver = smoother sweeps
    if args.mgview:
        _coarsereport(levels-1, hierarchy[0].m, args.coarse)
    hierarchy[0].y = hierarchy[0].zeros()
    infeas += _smoother(obsprob, args.coarse, hierarchy[0], hierarchy[0].y,
                        hierarchy[0].ell, hierarchy[0].chi,
                        symmetric=args.symmetric)

    # upward
    z = hierarchy[0].y
    for k in range(1, J+1):
        # accumulate corrections
        z = hierarchy[k].cP(z) + hierarchy[k].y
        if args.up > 0:
            # up smoother; up-obstacle is chi[k] not phi (see paper)
            if args.mgview:
                _levelreport(levels-1, k, hierarchy[k].m, args.up)
            infeas += _smoother(obsprob, args.up, hierarchy[k], z,
                                hierarchy[k].ell, hierarchy[k].chi,
                                symmetric=args.symmetric, forward=False)

    return z, infeas

def mcdlsolver(args, obsprob, J, hierarchy, monitor, w, ell, phi,
               iters=1, uex=None, totallevels=0):
    '''Iterate MCDL V-cycles until convergence.  The convergence criterion
    is a reduction by rtol of the "CP residual" norm, as computed by
    the irerr() method of ObstacleMonitor.'''

    assert iters >= 0
    infeascount = 0
    # do multigrid slash or V cycles
    for s in range(iters):
        # get current CP residual (inactive residual) norm
        irnorm, _ = monitor.irerr(w, ell, phi,
                                  uex=uex, indent=totallevels-1-J)
        # stop based on irtol condition
        if s == 0:
            if irnorm == 0.0:
                break
            irnorm0 = irnorm
        else:
            if irnorm <= args.irtol * irnorm0:
                break
            if irnorm > 100.0 * irnorm0:
                print('WARNING:  irnorm > 100 irnorm0')
        # Tai (2003) constraint decomposition method cycles; default=V(1,0);
        #   Alg. 4.7 in G&K (2009); see mcdl-solver and mcdl-slash in paper
        hierarchy[J].chi = phi - w                      # defect obstacle
        ell = - obsprob.residual(hierarchy[J], w, ell)  # starting source
        y, infeas = mcdlvcycle(args, obsprob, J, hierarchy, ell,
                               levels=totallevels)
        w += y
        infeascount += infeas
    return w, s, infeascount
