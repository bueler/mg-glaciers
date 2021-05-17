#!/usr/bin/env python3

'''For an extruded mesh with some empty columns, loop over the base mesh cells
to get the top cells, then the top nodes from those.'''

from firedrake import *

# picture of extruded mesh, with its node indexing:
#              10--14
#               |   |
#       3---6   9--13
#       |   |   |   |
#       2---5---8--12  16--18
#       |   |   |   |   |   |
#   0---1---4---7--11--15--17   <-- base mesh of 6 intervals
#     0   1   2   3   4   5     <-- numbering of base mesh cells
bm = UnitIntervalMesh(6)   # simple mods below allow UnitSquareMesh(), etc.
mesh = ExtrudedMesh(bm, layers=[[0,0], [0,2], [0,1], [0,3], [0,0], [0,1]],
                    layer_height=0.3)

# NOTE mesh.cell_set.layers_array is a singleton  [[0,n],]  if
# "mesh = ExtrudedMesh(bm, layers=n)", so generally check mesh.variable_layers
assert mesh.variable_layers

# use mesh coordinate functions so I can check the above picture
x, z = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, 'Lagrange', 1)  # Q1
xf = Function(V).interpolate(x)
zf = Function(V).interpolate(z)

# map from *lowest* extruded-mesh cells into extruded-mesh node indices
# WARNING: here cell_node_map[j] is invalid nonsense for j=0,4 empty columns!
cell_node_map = V.cell_node_map().values
print(cell_node_map)
node_offset = V.cell_node_map().offset  # node offset as you go up a column

# to get top nodes on top cell we get the cell-wise indexing scheme
section, iset, facets = V.cell_boundary_masks
# facets is ordered with sides first, then bottom, then top
off = section.getOffset(facets[-1])    # -2 for "bottom"
dof = section.getDof(facets[-1])
indices = iset[off:off+dof]

# loop over base mesh cells, printing coordinates of top-most nodes
for cell in range(bm.cell_set.size):
    # get column of cells
    start, extent = mesh.cell_set.layers_array[cell]  # NOT mesh.layer_extents
    ncell = extent - start - 1
    if ncell == 0:
        print('base cell %d has no extrusion' % cell)
        continue
    # get all nodes for top cell
    top_cell_nodes = cell_node_map[cell, ...] + node_offset * ncell - 1
    print('top extruded cell above base cell %d has top nodes:' % cell)
    xdofs = xf.dat.data_ro[top_cell_nodes]
    zdofs = zf.dat.data_ro[top_cell_nodes]
    for j in indices:
        print('  %d at (x,z) = (%.2f,%.2f)' \
              % (top_cell_nodes[j], xdofs[j], zdofs[j]))
