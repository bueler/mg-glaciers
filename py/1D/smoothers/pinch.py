#!/usr/bin/env python3
# (C) 2018 Ed Bueler

# Usage:
#   $ ./pinch.py sheet.geo
#   $ gmsh -2 sheet.msh
#   ...
# which generates sheet.msh.  (Use "gmsh sheet.geo" for GUI version.)
# One may inspect the mesh by loading it in python/firedrake:
#   $ source firedrake/bin/activate
#   (firedrake) $ ipython
#   ...
#   In [1]: from firedrake import *
#   In [2]: Mesh('sheet.msh')
# Main purpose is to solve the Stokes equations on this domain.  See
# README.md and flow.py.

# computational domain dimensions in meters
Hmax = 400.0
Hpinch = 50.0
L = 3000.0       # total along-flow length (x)
Lup = 1500.0     # location of bedrock step up (x)
Ldown = 2000.0   # location of bedrock step down (x);  Ldown > Lup
Lpinch = 1000.0

# numbering of parts of boundary (same as in domain.py; used in flow)
bdryids = {'outflow' : 41,
           'top'     : 42,
           'inflow'  : 43,
           'base'    : 44}

def writegeometry(geo,bs):
    # points on boundary, with target mesh densities
    Lmid = 0.5 * (Lup + Ldown)
    geo.write('Point(1) = {%f,%f,0,lc};\n' % (L,0.0))
    geo.write('Point(2) = {%f,%f,0,lc};\n' % (L,Hpinch))
    geo.write('Point(3) = {%f,%f,0,lc};\n' % (Ldown,Hmax))
    geo.write('Point(4) = {%f,%f,0,lc};\n' % (Lpinch,Hmax))
    geo.write('Point(5) = {%f,%f,0,lc};\n' % (0.0,Hpinch))
    geo.write('Point(6) = {%f,%f,0,lc};\n' % (0.0,0.0))
    # if there is a bedrock step, *refine* in the interior corners
    if abs(bs) > 1.0:
        geo.write('Point(7) = {%f,%f,0,lc};\n' % (Lup,0.0))
        geo.write('Point(8) = {%f,%f,0,lc_corner};\n' % (Lup,bs))
        geo.write('Point(9) = {%f,%f,0,lc};\n' % (Lmid,bs))
        geo.write('Point(10) = {%f,%f,0,lc_corner};\n' % (Ldown,bs))
        geo.write('Point(11) = {%f,%f,0,lc};\n' % (Ldown,0.0))

    # lines along boundary
    geo.write('Line(12) = {1,2};\n')
    geo.write('Line(13) = {2,3};\n')
    geo.write('Line(14) = {3,4};\n')
    geo.write('Line(15) = {4,5};\n')
    geo.write('Line(16) = {5,6};\n')
    if abs(bs) > 1.0:
        geo.write('Line(17) = {6,7};\n')
        geo.write('Line(18) = {7,8};\n')
        geo.write('Line(19) = {8,9};\n')
        geo.write('Line(20) = {9,10};\n')
        geo.write('Line(21) = {10,11};\n')
        geo.write('Line(22) = {11,1};\n')
        geo.write('Line Loop(29) = {12,13,14,15,16,17,18,19,20,21,22};\n')
    else:
        geo.write('Line(17) = {6,1};\n')
        geo.write('Line Loop(29) = {12,13,14,15,16,17};\n')

    # surface allows defining a 2D mesh
    geo.write('Plane Surface(31) = {29};\n')

    # "Physical" for marking boundary conditions
    geo.write('Physical Line(%d) = {12};\n' % bdryids['outflow'])
    geo.write('Physical Line(%d) = {13,14,15};\n' % bdryids['top'])
    geo.write('Physical Line(%d) = {16};\n' % bdryids['inflow'])
    if abs(bs) > 1.0:
        geo.write('Physical Line(%d) = {17,18,19,20,21,22};\n' % bdryids['base'])
    else:
        geo.write('Physical Line(%d) = {17};\n' % bdryids['base'])

    # ensure all interior elements written ... NEEDED!
    geo.write('Physical Surface(51) = {31};\n')

# dynamically extract geometry making these definitions (tolerance=1cm):
#   bs      = height of bedrock step     = (min z-coordinate over Lup < x < Ldown)
#   bmin    = minimum base elevation     = (min z-coordinates)
#   Hout    = ice thickness at output    = (max z-coordinate at x=L)
# in parallel no process owns the whole mesh so MPI_Allreduce() is needed
def getdomaindims(mesh,tol=0.01):
    from mpi4py import MPI
    # mesh needs to be a Mesh from Firedrake
    xa = mesh.coordinates.dat.data_ro[:,0]  # .data_ro acts like VecGetArrayRead
    za = mesh.coordinates.dat.data_ro[:,1]
    loc_bs = 9.99e99
    xinmid = (xa > Lup) * (xa < Ldown)
    if any(xinmid):
        loc_bs = min(za[xinmid])
    bs = mesh.comm.allreduce(loc_bs, op=MPI.MIN)
    loc_bmin = min(za)
    bmin = mesh.comm.allreduce(loc_bmin, op=MPI.MIN)
    loc_Hout = 0.0
    if any(xa > L-tol):
        loc_Hout = max(za[xa > L-tol])
    Hout = mesh.comm.allreduce(loc_Hout, op=MPI.MAX)
    return (bs,bmin,Hout)

def processopts():
    import argparse
    parser = argparse.ArgumentParser(description=
    '''Generate .geo geometry-description file, suitable for meshing by Gmsh, for
    the outline of a glacier flow domain with bedrock steps.  Also generates
    slab-on-slope geometry with -bs 0.0.
    ''')
    parser.add_argument('-o', metavar='FILE.geo', default='glacier.geo',
                        help='output file name (ends in .geo; default=glacier.geo)')
    parser.add_argument('-bs', type=float, default=100.0, metavar='X',
                        help='height of bed step (default=100 m)')
    parser.add_argument('-hmesh', type=float, default=80.0, metavar='X',
                        help='default target mesh spacing (default=80 m)')
    parser.add_argument('-refine', type=float, default=1.0, metavar='X',
                        help='refine resolution by this factor (default=1)')
    parser.add_argument('-refine_corner', type=float, default=4.0, metavar='X',
                        help='further local refinement at interior corner by this factor (default=4)')
    parser.add_argument('-testspew', action='store_true',
                        help='write .geo contents, w/o header, to stdout', default=False)  # just for testing
    return parser.parse_args()

if __name__ == "__main__":
    from datetime import datetime
    import sys, platform, subprocess

    args = processopts()
    commandline = " ".join(sys.argv[:])  # for save in comment in generated .geo
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print('writing domain geometry to file %s ...' % args.o)
    geo = open(args.o, 'w')
    # header which records creation info
    if not args.testspew:
        geo.write('// geometry-description file created %s by %s using command\n//   %s\n\n'
                  % (now,platform.node(),commandline) )
    # set "characteristic lengths" which are used by gmsh to generate triangles
    lc = args.hmesh / args.refine
    print('setting target mesh size of %g m' % lc)
    geo.write('lc = %f;\n' % lc)
    if abs(args.bs) > 1.0:
        lc_corner = lc / args.refine_corner
        print('setting target mesh size of %g m at interior corners' % lc_corner)
    else:
        lc_corner = lc
    geo.write('lc_corner = %f;\n' % lc_corner)
    # the rest
    writegeometry(geo,args.bs)
    geo.close()
    if args.testspew:
        result = subprocess.run(['cat', args.o], stdout=subprocess.PIPE)
        print(result.stdout)
