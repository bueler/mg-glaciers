# mg-glaciers/paper/genfigs/gis/

The two text files `topg_j1100_69Nlat.txt` and `thk_j1100_69Nlat.txt` were generated from `pism/examples/jako/` by:

  1. run `preprocess.sh` to download `Greenland1km.nc` (among other files)
  2. open `Greenland1km.nc` with `ncview` and dump `topg` and `thk` fields at the j=1100 cross section; looking at `lat` in `gr1km.nc` shows this is essentially 69 degrees N latitude
  3. rename the files to the above and comment-out the first few lines

Now we can generate a Python-matplotlib figure by reading with `np.loadtxt()`.
See `giscross.py`.
