#!/usr/bin/env python3
# (C) 2021 Ed Bueler

from profile import profile, profiledefaults

# numbering of parts of boundary; also used in solve.py
bdryids = {'top'     : 41,
           'base'    : 42}

def processopts():
    import argparse
    parser = argparse.ArgumentParser(description=
    '''Generate .geo geometry-description file, suitable for meshing by Gmsh,
for the profile of a dome:
  $ ./domain.py -o dome.geo
  $ gmsh -2 dome.geo
Then run solve.py to solve the Stokes problem.''',
      formatter_class=argparse.RawTextHelpFormatter)
    adda = parser.add_argument
    adda('-H', type=float, metavar='X',
         default=profiledefaults['H'],
         help='dome height')
    adda('-hratio', type=float, default=0.6, metavar='X',
         help='mesh spacing from formula: lc=hratio*(2R)/mx  (default=0.6)')
    adda('-mx', type=int, default=10, metavar='MX',
         help='number of subintervals (default=10)')
    adda('-o', metavar='FILE.geo', default='dome.geo',
         help='output file name (default=dome.geo)')
    adda('-R', type=float, metavar='X',
         default=profiledefaults['R'],
         help='dome radius')
    return parser.parse_args()

def writegeometry(geo,xtop,ytop):
    '''Write a .geo file which saves the profile geometry.  Boundary
    order starts with (0.0,0.0), then does the base, then does the top,
    so boundary is traversed clockwise.'''
    # points on top
    offset = 51
    ntop = len(xtop)
    geo.write('Point(%d) = {%f,%f,0,lc};\n' % (offset, 0.0, 0.0))
    for j in range(ntop-1):
        geo.write('Point(%d) = {%f,%f,0,lc};\n' \
                  % (offset + j + 1,xtop[-j-1],ytop[-j-1]))
    # line along base
    linestart = offset + ntop
    geo.write('Line(%d) = {%d,%d};\n' % (linestart, offset, offset + 1))
    # lines along top boundary
    for j in range(ntop-2):
        geo.write('Line(%d) = {%d,%d};\n' \
                  % (linestart + 1 + j, offset + j + 1, offset + j + 2))
    geo.write('Line(%d) = {%d,%d};\n' \
              % (linestart + ntop - 1, offset + ntop - 1, offset))
    # full line loop
    lineloopstart = linestart + ntop
    geo.write('Line Loop(%d) = {' % lineloopstart)
    for j in range(ntop-1):
        geo.write('%d,' % (linestart + j))
    geo.write('%d};\n' % (linestart + ntop - 1))
    # surface allows defining a 2D mesh
    surfacestart = lineloopstart + 1
    geo.write('Plane Surface(%d) = {%d};\n' % (surfacestart,lineloopstart))
    # "Physical" for marking boundary conditions
    geo.write('// boundary id = %d is top\n' % bdryids['top'])
    geo.write('Physical Line(%d) = {' % bdryids['top'])
    for j in range(ntop-2):
        geo.write('%d,' % (linestart + j + 1))
    geo.write('%d};\n' % (linestart + ntop - 1))
    geo.write('// boundary id = %d is base\n' % bdryids['base'])
    geo.write('Physical Line(%d) = {%d};\n' % (bdryids['base'],linestart))
    # ensure all interior elements are written ... NEEDED!
    geo.write('Physical Surface(%d) = {%d};\n' % \
              (surfacestart + 1, surfacestart))

if __name__ == "__main__":
    from datetime import datetime
    import sys, platform, subprocess
    import numpy as np
    # record command-line and date for creation info
    args = processopts()
    commandline = " ".join(sys.argv[:])  # for save in comment in generated .geo
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # header which records creation info
    print('writing domain geometry to file %s ...' % args.o)
    geo = open(args.o, 'w')
    geo.write('// geometry-description file created %s by %s\n' \
              % (now,platform.node()) )
    geo.write('// using command: %s\n\n' % commandline)
    # 1D arrays for profile
    x, ytop = profile(args.mx, R=args.R, H=args.H)
    # set "characteristic lengths" which are used by gmsh to generate triangles
    lc = args.hratio * (2.0 * args.R) / args.mx
    print('setting target mesh size of h=%g m for mx=%d subintervals' \
          % (lc, args.mx))
    geo.write('lc = %f;\n' % lc)
    # create the rest of the file
    writegeometry(geo, x, ytop)
    geo.close()
