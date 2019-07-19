# Resampling - fast n-dimensional polynomial fitting

Resampling is designed to enable fitting of N-dimensional data on a
local scale using polynomials of an arbitrary order.  This was
primarily inspired by the need to resample irregular astronomical
spectroscopic data into a regularly spaced data cube.  However,
once the algorithm was complete, it was clear that this could
easily be extended to arbitrary dimensions and units.

## Licence
Resampling is licensed under a 3-clause BSD style licnse.  Please
see the LICENSE file.

## Requirements
Resampling was created using Python 3.7.  The following packages are
required:

- astropy
- Bottleneck
- numba
- numpy
- scipy
- scikit_learn

Many thanks to the developers of these great packages.  If you wish
to go through the Jupyter notebook examples, then matplotlib will
also be requied.
