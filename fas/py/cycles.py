# module for FAS cycles: V and F

# meshes is of type MeshLevel1D; see meshlevel.py
# prob is of type Problem1D; see problems.py

import numpy as np

__all__ = ['FAS']

class FAS(object):

    def __init__(self,meshes,prob,kcoarse,kfine,
                 mms=False,coarse=1,down=1,up=1,niters=2,monitorupdate=False):
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
        self.monitorupdate = monitorupdate
        self.wu = np.zeros(self.kfine+1)

    # report work units by weighted summing wu[k]
    def wutotal(self):
        tot = 0
        for k in range(self.kfine+1):
            tot += self.wu[self.kfine - k] / 2**k
        return tot

    # return L^2 norm of current residual  r = ell - F(w)
    def residualnorm(self,w,ell):
        mesh = self.meshes[self.kfine]
        return mesh.l2norm(ell - self.prob.F(mesh,w))

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
            if self.monitorupdate:
                print('     ' + '  ' * (self.kfine + 1 - k), end='')
                print('coarse update norm %.5e' % self.meshes[k-1].l2norm(ucoarse - Ru))
            # prolong up
            u += self.meshes[k].prolong(ucoarse - Ru)
            # smooth: NGS sweeps on fine mesh
            for q in range(self.up):
                self.prob.ngssweep(self.meshes[k],u,ell,
                                   forward=False,niters=self.niters)
            self.wu[k] += self.up

    # FIXME TOTALLY UNTESTED NONSENSE
    def fcycle(self):
        ulist = [None] * (self.kfine+1)
        for k in range(self.kcoarse,self.kfine+1):
            ellg = self.rhs(self.meshes[k],self.prob,mms=self.mms)
            if k == self.kcoarse:
                ulist[k] = self.meshes[k].zeros()
                self.coarsesolve(ulist[self.kcoarse],ellg)
                self.wu[kcoarse] += self.coarse
            else:
                ulist[k] = self.meshes[k].prolong(ulist[k-1])
                self.vcycle(k,ulist[k],ellg)
        return ulist[self.kfine]

