#!/usr/bin/env python3

# firedrake already supports:
#   prolong coarse->fine on functions
#   restrict fine->coarse on dual (functionals)
#   inject fine->coarse on functions
# see
#   https://www.firedrakeproject.org/_modules/firedrake/mg/interface.html
#   https://www.firedrakeproject.org/_modules/firedrake/mg/kernels.html
#   https://arxiv.org/pdf/2101.05158.pdf

from firedrake import *
from firedrake.petsc import PETSc
Print = PETSc.Sys.Print

mx, my = 1, 1
mesh = UnitSquareMesh(mx, my)

hierarchy = MeshHierarchy(mesh, 1)
coarse = hierarchy[0]
fine = hierarchy[1]

def f(x,y):
    return x * exp(y)

def l2dist(W,u,v):
    diff = Function(W).interpolate(u - v)
    return sqrt(assemble(dot(diff, diff) * dx))

Wc = FunctionSpace(coarse, 'Lagrange', degree=1)
xc, yc = SpatialCoordinate(coarse)
fc = Function(Wc).interpolate(f(xc,yc))

Wf = FunctionSpace(fine, 'Lagrange', degree=1)
xf, yf = SpatialCoordinate(fine)
ff = Function(Wf).interpolate(f(xf,yf))

Pfc = Function(Wf)
prolong(fc,Pfc)
Print('|P(fc) - ff|_2 = %.3e' % (l2dist(Wf,Pfc,ff)))

RPfc = Function(Wc)
restrict(Pfc,RPfc)
Print('|R(P(fc)) - fc|_2 = %.3e  (bad: R acts on dual space)' % (l2dist(Wc,RPfc,fc)))

IPfc = Function(Wc)
inject(Pfc,IPfc)
Print('|I(P(fc)) - fc|_2 = %.3e' % (l2dist(Wc,IPfc,fc)))

if True:
    outname = 'finemesh.pvd'
    Print('saving fine mesh fields to %s ...' % outname)
    ff.rename('ff')
    Pfc.rename('P(fc)')
    File(outname).write(ff,Pfc)
    outname = 'coarsemesh.pvd'
    Print('saving coarse mesh fields to %s ...' % outname)
    fc.rename('fc')
    RPfc.rename('R(P(fc))')
    IPfc.rename('I(P(fc))')
    File(outname).write(fc,RPfc,IPfc)

