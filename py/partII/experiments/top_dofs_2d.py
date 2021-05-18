#!/usr/bin/env python3

'''For an extruded mesh over a 2D triangle base mesh, with some empty columns,
loop over the base mesh cells to get the top cells and their top nodes.
The main point here is that this is a small modification of top_dofs_1d.py.
Includes check that base mesh node coordinates match up with top node
coordinates.'''

from firedrake import *

# define base mesh and get its coordinates
bm = UnitSquareMesh(2,2)
W = FunctionSpace(bm, 'Lagrange', 1)  # P1 triangles
bm_cell_node_map = W.cell_node_map().values
x, y = SpatialCoordinate(bm)
bm_xf = Function(W).interpolate(x)
bm_yf = Function(W).interpolate(y)

# print the coordinates to generate picture below
for cell in range(bm.cell_set.size):
    print('base cell %d:' % cell)
    for j in range(3):
        jj = bm_cell_node_map[cell,j]
        print('  %d at (x,y) = (%.2f,%.2f)' \
              % (jj, bm_xf.dat.data_ro[jj], bm_yf.dat.data_ro[jj]))

# base mesh cell index (left) and extruded mesh height in cells (right):
#   *---*---*    *---*---*
#   |\ 5|\ 7|    |\ 1|\ 1|
#   | \ | \ |    | \ | \ |
#   |3 \|6 \|    |3 \|0 \|
#   *---*---*    *---*---*
#   |\ 1|\ 4|    |\ 2|\ 0|
#   | \ | \ |    | \ | \ |
#   |0 \|2 \|    |0 \|1 \|
#   *---*---*    *---*---*
mesh = ExtrudedMesh(bm, layers=[[0,0], [0,2], [0,1], [0,3],
                                [0,0], [0,1], [0,0], [0,1]],
                    layer_height=0.3)

# NOTE mesh.cell_set.layers_array is a singleton  [[0,n],]  if
# "mesh = ExtrudedMesh(bm, layers=n)", so generally check mesh.variable_layers
assert mesh.variable_layers

# use mesh coordinate functions so I can check the above picture
x, y, z = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, 'Lagrange', 1)  # Q1
xf = Function(V).interpolate(x)
yf = Function(V).interpolate(y)
zf = Function(V).interpolate(z)

# map from *lowest* extruded-mesh cells into extruded-mesh node indices
# WARNING: here cell_node_map[j] is invalid nonsense for j=0,4 empty columns!
cell_node_map = V.cell_node_map().values
#print(cell_node_map)
node_offset = V.cell_node_map().offset  # node offset as you go up a column

# to get nodes on top cells, get the cell-wise indexing scheme
section, iset, facets = V.cell_boundary_masks
# facets ordered with sides first, then bottom, then top
off = section.getOffset(facets[-1])    # -2 for "bottom"
dof = section.getDof(facets[-1])
indices = iset[off:off+dof]

# loop over base mesh cells, printing coordinates of top-most nodes
for cell in range(bm.cell_set.size):
    # get column of cells
    start, extent = mesh.cell_set.layers_array[cell]  # NOT mesh.layer_extents
    ncell = extent - start - 1
    if ncell == 0:
        print('base cell %d: no extrusion' % cell)
        continue
    # get all nodes for top cell
    top_cell_nodes = cell_node_map[cell, ...] + node_offset * ncell - 1
    print('base cell %d: top extruded cell has top nodes at layer %d:' \
          % (cell, ncell))
    xdofs = xf.dat.data_ro[top_cell_nodes]
    ydofs = yf.dat.data_ro[top_cell_nodes]
    zdofs = zf.dat.data_ro[top_cell_nodes]
    for j in range(len(indices)):
        jj = indices[j]  # indices is from cell-wise indexing scheme
        # check top node coordinates match corresponding base-mesh node
        assert xdofs[jj] == bm_xf.dat.data_ro[bm_cell_node_map[cell,j]]
        assert ydofs[jj] == bm_yf.dat.data_ro[bm_cell_node_map[cell,j]]
        print('  %d at (x,y,z) = (%.2f,%.2f,%.2f)' \
              % (top_cell_nodes[jj], xdofs[jj], ydofs[jj], zdofs[jj]))
