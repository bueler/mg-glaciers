# module for FAS cycles: V and F

import numpy as np

__all__ = ['FAS']

class FAS(object):
    '''Class for the full approximation storage (FAS) scheme.  Implements
    V-cycles and F-cycles.  At initialization:
      meshes[k]: type MeshLevel1D from meshlevel.py
      prob:      type Problem1D from problems.py
    Note meshes[kcoarse],...,meshes[kfine] are the mesh levels.  The key
    smoother and coarse-level solver component is the NGS method
    prob.ngssweep().  The coarse correction uses prob.F() for the nonlinear
    operator, meshes[k].Rfw() for full-weighting restriction of vectors,
    meshes[k].CR() for canonical restriction of linear functionals,
    and meshes[k].P() for prolongation.'''

    def __init__(self,meshes,prob,kcoarse,kfine,
                 mms=False,coarse=1,down=1,up=1,niters=2,
                 monitor=False,monitorupdate=False):
        self.meshes = meshes
        self.prob = prob
        assert kcoarse >= 0
        assert kcoarse < kfine
        self.kcoarse = kcoarse
        self.kfine = kfine
        self.mms = mms
        self.coarse = coarse
        self.down = down
        self.up = up
        self.niters = niters
        self.monitor = monitor
        self.monitorupdate = monitorupdate
        self.wu = np.zeros(self.kfine+1)

    # return L^2 norm of residual r = ell - F(w) on k level mesh
    def residualnorm(self,k,w,ell):
        mesh = self.meshes[k]
        return mesh.l2norm(ell - self.prob.F(mesh,w))

    # on monitor flag, indented-print residual norm
    def printresidualnorm(self,s,k,w,ell):
        if self.monitor:
            print('  ' * (self.kfine + 1 - k), end='')
            print('%d: residual norm %.5e' % (s,self.residualnorm(k,w,ell)))

    # on monitorupdate flag, indented-print norm of coarse-mesh update
    def printupdatenorm(self,k,du):
        if self.monitorupdate:
            print('     ' + '  ' * (self.kfine + 1 - k), end='')
            print('coarse update norm %.5e' % self.meshes[k-1].l2norm(du))

    # report work units by weighted summing wu[k]
    def wutotal(self):
        tot = 0.0
        for k in range(self.kfine+1):
            tot += self.wu[self.kfine - k] / 2.0**k
        return tot

    # compute right-hand side, a linear functional, from function g(x)
    def rhs(self,k):
        ellg = self.meshes[k].zeros()
        if self.mms:
            _, g = self.prob.mms(self.meshes[k].xx())
            g *= self.meshes[k].h
            ellg[1:-1] = g[1:-1]
        return ellg

    # solve coarsest problem by NGS sweeps; acts in-place on u
    def coarsesolve(self,u,ell):
        for q in range(self.coarse):
            self.prob.ngssweep(self.meshes[self.kcoarse],u,ell,
                               niters=self.niters)
        self.wu[self.kcoarse] += self.coarse

    # FAS V-cycle for levels k down to k=kcoarse; acts in-place on u
    def vcycle(self,k,u,ell):
        if k == self.kcoarse:
            self.coarsesolve(u,ell)
        else:
            assert k > self.kcoarse
            # smooth: NGS sweeps on fine mesh
            for q in range(self.down):
                self.prob.ngssweep(self.meshes[k],u,ell,
                                   niters=self.niters)
            self.wu[k] += self.down
            # restrict down using  ell = R' (f^h - F^h(u^h)) + F^{2h}(R u^h)
            rfine = ell - self.prob.F(self.meshes[k],u)  # residual on the fine mesh
            Ru = self.meshes[k].Rfw(u)
            coarseell = self.meshes[k].CR(rfine) + self.prob.F(self.meshes[k-1],Ru)
            # recurse
            ucoarse = Ru.copy()
            self.vcycle(k-1,ucoarse,coarseell)
            du = ucoarse - Ru
            self.printupdatenorm(k,du)
            # correct by prolongation of update:  u <- u + P(u^{2h} - R u^h)
            u += self.meshes[k].P(du)
            # smooth: NGS sweeps on fine mesh
            for q in range(self.up):
                self.prob.ngssweep(self.meshes[k],u,ell,
                                   forward=False,niters=self.niters)
            self.wu[k] += self.up

    # FAS F-cycle for levels kcoarse up to kfine; returns u
    def fcycle(self,cycles=1):
        u = self.meshes[self.kcoarse].zeros()
        ellg = self.rhs(self.kcoarse)
        self.printresidualnorm(0,self.kcoarse,u,ellg)
        self.coarsesolve(u,ellg)
        self.printresidualnorm(1,self.kcoarse,u,ellg)
        for k in range(self.kcoarse+1,self.kfine+1):
            u = self.meshes[k].P(u)  # use same prolong
            ellg = self.rhs(k)
            Z = cycles if k == self.kfine else 1
            for s in range(Z):
                self.printresidualnorm(s,k,u,ellg)
                self.vcycle(k,u,ellg)
            self.printresidualnorm(s+1,k,u,ellg)
        return u

