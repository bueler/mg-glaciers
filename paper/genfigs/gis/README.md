# mg-glaciers/paper/genfigs/gis/

Andy Aschwanden sent me the data `gris_cross_profile.csv` from (https://nsidc.org/data/IDBMG4/versions/3)[BedMachine] processed by QGis, and a script to read it using pandas.  See `../giscross.py`.

Cite BedMachine using URL and Morlighemetal2017 in `../../review.bib`.

## old/

This version is deprecated.

These two text files `topg_j1100_69Nlat.txt` and `thk_j1100_69Nlat.txt` were generated from the PISM Jakobshavn example:

  1. `cd pism/examples/jako/`
  2. run `preprocess.sh` to download `Greenland1km.nc` (among other files) and generate `gr1km.nc`
  3. open `Greenland1km.nc` with `ncview` and dump `topg` and `thk` fields at the j=1100 cross section
  4. look at `lat` in `gr1km.nc`; j=1100 is essentially 69 degrees N latitude
  5. rename the files to the above names, and comment-out the first few lines, suitable for reading with `np.loadtxt()`

Now we can generate a Python-matplotlib figure in `../giscross.py`.
