#!/bin/env python
import numpy as np
import utils
import matplotlib.pyplot as plt
import os
from gpuInversion import *
import cupy



# projection information
projectFolder='/RUNDATA_16804/junyan1998/Chehualin/H4-10/crop_H4-10'
stackName='microStack.h5'
subset=[2000,2300,3800,4050] #[row_start,row_end,col_start,col_end]
subsetStack=1
overwriteStack=0
dS=3
# satelite information
wavelen=0.031
H=620e3 
win=5


##########################################

geomFolder=os.path.join(projectFolder,'geom_reference')
bperpFolder=os.path.join(projectFolder,'baselines')
slcFolder=os.path.join(projectFolder,'SLC')
# load SLC and make StackFile
if os.path.exists(stackName) and not overwriteStack:
    pass
else:
    from makeH5Stack import makeSLCStack
    print('make SLC stack')
    if subsetStack:
        makeSLCStack(slcFolder,stackName,subset)
    else:    
        makeSLCStack(slcFolder,stackName)

# load longtitude and latitdue
lon=utils.loadGeom(geomFolder,'lon.rdr',subset)
lat=utils.loadGeom(geomFolder,'lat.rdr',subset)

# load Data 
print('Loading stack dataset\n')

if subsetStack:
    real,imag=utils.loadStack(stackName)
else:
    real,imag=utils.loadStack(stackName,subset)


# load baselines 
if os.path.exists('bp.npy'):
    bperp=np.load('bp.npy')
else:
    bperp=utils.loadBperp(bperpFolder)
    np.save('bp',bperp)
    print('write to file bp.npy')

# some problem with this case :(
bperp=bperp[:-2]
real=real[:-2]
imag=imag[:-2]

# get the size of each element
nslc=real.shape[0]
lns=real.shape[1]
width=real.shape[2]


# use the differential phase instead 
masterIx=np.argsort(bperp)[int(nslc/2)]
realRef=real[masterIx,...]
imagRef=imag[masterIx,...]
_real=realRef*real+imagRef*imag
_imag=real*imagRef-imag*realRef
real=_real
imag=_imag
bperp=bperp-bperp[masterIx]
del imagRef, realRef, _real, _imag


# load incident angle and slantrange calculation
inc=utils.loadGeom(geomFolder)
incAve=np.average(inc)*3.14/180
r0=utils.calcSR(incAve,H)

# calculate baseline aperture 
deltaS=wavelen*r0/2/(np.max(bperp)-np.min(bperp))
dS=min(dS,deltaS)
# db=utils.baselineInverval(bperp)
# db=np.std(bperp)
# Srange=wavelen*r0/4/3.14/(nslc*20)**0.5/db
Srange=80
print(Srange,dS)
trial=np.arange(-Srange/2,Srange/2,dS).reshape([1,-1])
ntrial=trial.shape[1]

# generate steering matrix
print('calculate steering matrix')
bperp=bperp.reshape([-1,1])
steering=bperp*trial
steering=np.exp(steering*(1j)*4*3.14/wavelen/r0)
steeringH=np.conj(steering.T)


# normalize observation 
cpx=real+1j*imag
gaussian=gpu_moving_average_2d(cpx,3,nslc)
da=cupy.asnumpy(da_gpu(gaussian))


# start inversion
print('start inversion ... ')

# method 1: mvdr beamforming
tomography=gpuBFinversion(gaussian,steering)


# method 2: svd beamforming

# import svdBF 
# gaussian=gaussian.get().reshape(nslc,-1)
# tomography=svdBF.svdBF(gaussian,steering,1)
# tomography=tomography.reshape(lns,width,ntrial).real


# method 3: Compressive sensing method

# from CSInversion import cs_omp
# gaussian=gaussian.reshape(nslc,-1).get()
# tomography=np.empty(shape=(lns*width,ntrial),dtype=np.complex64)
# for i in range(lns*width):
#     tomography[i,:]=cs_omp(gaussian[:,i].T,steering)
#     if i% 1000==0: print(f'invert {i}/{lns*width} pixels',end='\r')
# print()
# tomography=np.abs(tomography.reshape(lns,width,ntrial))


#################################3
# verify result 

print('result output')
inten=np.log(np.average((real**2+imag**2)**0.5,axis=0))
output=trial[0,np.argmax(tomography,axis=2)]*np.sin(incAve)
output[da>0.6]=np.nan
utils.exportPointHeight(output,lon,lat)


plt.subplot(121)
plt.imshow(output,cmap='jet')
plt.colorbar(orientation= "horizontal")
plt.subplot(122)
plt.imshow(inten,cmap="gray")
plt.colorbar(orientation= "horizontal")
plt.savefig('csinverse/aveInt02')
plt.close()



for testLine in range(0,real.shape[1],10):
    plt.figure(figsize=(10,4))
    plt.pcolor(tomography[testLine,...].T,cmap='rainbow')
    plt.colorbar(orientation='horizontal')
    plt.savefig(f'csinverse/profile{testLine}')
    plt.close()


