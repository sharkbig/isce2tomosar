#!/bin/env python
import numpy as np
import utils
from utils import cdot 
import matplotlib.pyplot as plt
import os
from gpuInversion import *
import cupy as cp
from makeH5Stack import *


# projection information
projectFolder='/RUNDATA_16804/junyan1998/Chehualin/H4-10/crop_H4-10'
stackName='microStack.h5'
subset=[1800,2300,3800,4050] #[row_start,row_end,col_start,col_end]
subsetStack=1
overwriteStack=0
dS=5
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
    
    print('make SLC stack')
    if subsetStack:
        makeSLCStack(slcFolder,stackName,subset)
    else:    
        makeSLCStack(slcFolder,stackName)

print('Loading SLC stack dataset')

if subsetStack:
    real,imag=loadSLCStack(stackName)
else:
    real,imag=loadSLCStack(stackName,subset)


# load longtitude and latitdue and slant range 
print('Loading geometry dataset')
if  not os.path.exists('geom.h5') or overwriteStack:
    makeGeomStack(geomFolder,'geom.h5',subset)

lon=loadGeomStack('geom.h5','lon')
lat=loadGeomStack('geom.h5','lat')
inc=loadGeomStack('geom.h5','inc')

incAve=np.average(inc)*3.14/180
r0=utils.calcSR(incAve,H)

# load Data 


# load baselines 
print('Averaging baseline')
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



# calculate baseline aperture 
deltaS=wavelen*r0/2/(np.max(bperp)-np.min(bperp))
dS=min(dS,deltaS/2)
# db=utils.baselineInverval(bperp)
# db=np.std(bperp)
# Srange=wavelen*r0/4/3.14/(nslc*20)**0.5/db
Srange=60
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
print('apply filter')
cpx=real+1j*imag
if win==1:
    gaussian=cpx
else: 
    gaussian=gpu_moving_average_2d(cpx,win,nslc)

da=cp.asnumpy(da_gpu(gaussian))


# start inversion
print('start inversion ... ')

### method 1: mvdr beamforming (on GPU)

tomography=gpuBFinversion(gaussian,steering)
k=cp.argmax(tomography,axis=2)
s_gpu=cp.asarray(steering)

# calculate coherence
sig=cp.angle(gaussian)
sig=cp.exp(1j*sig)
L_g=np.einsum('ij,jkl->ikl',s_gpu.conj().T,sig)
L_k=np.einsum('ij,jkl->ikl',s_gpu.conj().T,s_gpu[:,k])
coh=cp.sum(L_g*L_k.conj(),axis=0)/  \
    cp.sum(L_k*L_k.conj(),axis=0)**0.5/  \
    cp.sum(L_g*L_g.conj(),axis=0)**0.5

# copy data back
coh=cp.abs(coh).get()
tomography=tomography.get().real


#### scatter distribution plot
# count,nbin=np.histogram(coh.flatten(),bins=50)
# cdf=np.cumsum(count/sum(count))
# plt.plot(nbin[1:],cdf)
# plt.savefig('coh_cumsum')


### method 2: tsvd beamforming

# import svdBF 
# gaussian=gaussian.get().reshape(nslc,-1)
# tomography=svdBF.svdBF(gaussian,steering,1e-3) # 1 for truncated threshold
# predict=np.einsum('ij,kj->ik',tomography,steering)
# residual=np.sum(gaussian.T.conj()*predict,axis=1)
# tomography=tomography.reshape(lns,width,ntrial).real
# residual=residual.reshape(lns,width)


### method 3: Compressive sensing method

# from CSInversion import cs_omp
# gaussian=gaussian.reshape(nslc,-1).get()
# tomography=np.empty(shape=(lns*width,ntrial),dtype=np.complex64)
# for i in range(lns*width):
#     tomography[i,:]=cs_omp(gaussian[:,i].T,steering)
#     if i% 1000==0: print(f'invert {i}/{lns*width} pixels',end='\r')
# print()
# tomography=np.abs(tomography.reshape(lns,width,ntrial))


#################################

# verify result 
print('result output')
inten=np.log(np.average((real**2+imag**2)**0.5,axis=0))
output=trial[0,np.argmax(tomography,axis=2)]*np.sin(incAve)

output[coh<0.6]=np.nan
coh[coh<0.6]=np.nan
utils.exportPointHeight(output,lon,lat)



plt.figure(figsize=(16,5))
plt.subplot(141)
plt.imshow(inten,cmap="gray")
plt.title('intensity')
plt.colorbar(orientation= "horizontal")
plt.subplot(142)
plt.imshow(da,cmap="jet")
plt.clim(0.4,1)
plt.title('dispersion index')
plt.colorbar(orientation= "horizontal")
plt.subplot(143)
plt.imshow(output,cmap='jet')
plt.colorbar(orientation= "horizontal")
plt.title('tomo-height estimation')
plt.subplot(144)
plt.imshow(coh,cmap="rainbow")
plt.title('phase residule (log scale)')
plt.colorbar(orientation= "horizontal")
plt.tight_layout()
plt.savefig('csinverse/aveInt02')
plt.close()

dist=np.arange(width)
d,z=np.meshgrid(dist,trial*np.sin(incAve))
for testLine in range(0,real.shape[1],10):
    plt.figure(figsize=(10,4))
    
    plt.pcolor(d,z,tomography[testLine,...].T,cmap='rainbow')
    plt.plot(output[testLine,:],'k+')
    plt.colorbar(orientation='horizontal')
    plt.savefig(f'csinverse/profile{testLine}')
    plt.close()


