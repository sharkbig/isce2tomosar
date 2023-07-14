**There's no garentee of correctness to this code**

Author: Jun-Yan Chen
MS of Dept. of Geosciences, National Taiwan University.
contact: timjunyanchen@gmail.com

# isce2tomosar
This is an simple experiment code to process the isce slc stack
to make the sar tomography

# Structure of code

paramter in tomoInversion.py
- projectFolder
- stackName
- wavelen: wavelength of X-band image 0.031 m
- H: satellite height, to estimate the slant-range distance of sar image

# python package needed
```
osgeo
h5py
scipy
numpy
matplotlib

```