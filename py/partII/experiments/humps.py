#!/usr/bin/env python3
'''Demonstrate that extruded meshes can handle disconnected ice masses.'''

import numpy as np
from firedrake import *

dx = 1000.0
mx, mz = 12, 4                        # number of elements in extruded mesh
ice = [0,1,1,1,0,0,1,1,1,1,0,0]       # =1 where ice is present in element
profile = [0.0,   0.0, 500.0, 400.0, 0.0, 0.0,        # mx+1 node elevations
           0.0, 600.0, 700.0, 200.0, 0.0, 0.0, 0.0]

basemesh = IntervalMesh(mx, length_or_left=0.0, right=dx*mx)
layermap = np.zeros((mx,2), dtype=int)  # [[0,0], [0,0], ..., [0,0]]
layermap[:,1] = mz * np.array(ice)
mesh = ExtrudedMesh(basemesh, layers=layermap, layer_height=1.0/mz)

P1base = FunctionSpace(basemesh,'Lagrange',1)
profilebase = Function(P1base)
profilebase.dat.data[:] = np.array(profile)

def extend(mesh, f):
    '''On an extruded mesh extend a function f(x,z), already defined on the
    base mesh, to the mesh using the 'R' constant-in-the-vertical space.'''
    Q1R = FunctionSpace(mesh, 'P', 1, vfamily='R', vdegree=0)
    fextend = Function(Q1R)
    fextend.dat.data[:] = f.dat.data_ro[:]
    return fextend

x, z = SpatialCoordinate(mesh)
Vcoord = mesh.coordinates.function_space()
XZ = Function(Vcoord).interpolate(as_vector([x, extend(mesh,profilebase) * z]))
mesh.coordinates.assign(XZ)

x, z = SpatialCoordinate(mesh)
Q1 = FunctionSpace(mesh,'Lagrange',1)
f = Function(Q1).interpolate(z)
f.rename('f')
File('humps.pvd').write(f)
