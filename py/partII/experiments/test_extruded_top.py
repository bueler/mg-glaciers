#!/usr/bin/env python3

# sent as MFE to Firedrake slack on 9 May
# Lawrence responded it is unsupported ... but then on slack 13 May he patched
#   firedrake/src/firedrake/cython/extrusion_numbering.pyx
# and now it is supported

from firedrake import *

# supported extruded mesh with 1-element columns
#         6 --11
#         |    |
#         5 --10
#         |    |
#         4 -- 9
#         |    |
#    1 -- 3 -- 8 --13
#    |    |    |    |
#    0 -- 2 -- 7 --12
print('1-element columns')
basemesh = UnitIntervalMesh(3)
layermap = [[0,1], [0,4], [0, 1]]
mesh = ExtrudedMesh(basemesh, layers=layermap, layer_height=0.25)
P1 = FunctionSpace(mesh, 'Lagrange', 1)
bottombc = DirichletBC(P1, 1.0, 'bottom')
print(bottombc.nodes) # [0 2 7 12]
topbc = DirichletBC(P1, 1.0, 'top')
print(topbc.nodes) # [1 3 4 5 6 8 9 10 11 13]

# NOW supported extruded mesh with zero-element columns
#         5 --10
#         |    |
#         4 -- 9
#         |    |
#         3 -- 8
#         |    |
#         2 -- 7
#         |    |
#    0 -- 1 -- 6 -- 11   (note 1,6 are both top and bottom)
print('0-element columns')
basemesh = UnitIntervalMesh(3)
layermap = [[0,0], [0,4], [0, 0]]
mesh = ExtrudedMesh(basemesh, layers=layermap, layer_height=0.25)
P1 = FunctionSpace(mesh, 'Lagrange', 1)
bottombc = DirichletBC(P1, 1.0, 'bottom')
print(bottombc.nodes) # [0 1 6 11]
topbc = DirichletBC(P1, 1.0, 'top')
# following is now fixed:
#topbc.nodes  # FAILS: "AssertionError: Not expecting negative number of layers"
print(topbc.nodes) # [1 2 3 4 5 6 7 8 9 10]

# use Paraview point inspector for another confirmation
#f = Function(P1)
#File('foo.pvd').write(f)
