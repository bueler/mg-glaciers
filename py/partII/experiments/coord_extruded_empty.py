from firedrake import *
bm = UnitIntervalMesh(4)
mesh0 = ExtrudedMesh(bm, layers=[[0,0], [0,2], [0,0], [0,0]], layer_height=1.0)
print(mesh0.coordinates.dat.data_ro)
mesh1 = ExtrudedMesh(bm, layers=[[0,1], [0,2], [0,1], [0,1]], layer_height=1.0)
print(mesh1.coordinates.dat.data_ro)
