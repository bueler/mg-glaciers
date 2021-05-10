#!/usr/bin/env python3

# sent as MFE to Firedrake slack on 9 May

from firedrake import *
basemesh = UnitIntervalMesh(3)
layermap = [[0,0], [0,4], [0, 0]]
mesh = ExtrudedMesh(basemesh, layers=layermap, layer_height=0.25)
P1 = FunctionSpace(mesh, 'Lagrange', 1)
bottombc = DirichletBC(P1, 1.0, 'bottom')
print(bottombc.nodes) # SUCCEEDS
topbc = DirichletBC(P1, 1.0, 'top')
topbc.nodes  # FAILS: "AssertionError: Not expecting negative number of layers"
