#!/bin/env python
import numpy as np
import utils
import matplotlib.pyplot as plt
import os
import h5py
from scipy.linalg import lstsq,inv

# projection information
projectFolder='/RUNDATA_16804/junyan1998/Chehualin/H4-10/crop_H4-10'
stackName='microStack.h5'
subset=[1800,2400,3800,4100] #[row_start,row_end,col_start,col_end]
subsetStack=1
overwriteStack=0

# satelite information
wavelen=0.031
H=620e3  

###############

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
    real,imag=utils.loadStack(stackNmae,subset)


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
masterIx=int(nslc/2)
realRef=real[masterIx,...]
imagRef=imag[masterIx,...]
_real=realRef*real+imagRef*imag
_imag=-real*imagRef+imag*realRef
real=_real
imag=_imag
bperp=bperp-bperp[masterIx]
del imagRef, realRef, _real, _imag


# load incident angle and slantrange calculation
inc=utils.loadGeom(geomFolder)
incAve=np.average(inc)*3.14/180
r0=utils.calcSR(incAve,H)

# calculate baseline aperture 
dS=wavelen*r0/2/(np.max(bperp)-np.min(bperp))
db=np.abs(bperp[0]-bperp[1])
sortB=sorted(bperp[1:]) 
for i in range(nslc-1):
    if db< sortB[i]-sortB[i-1]:
        db=sortB[i]-sortB[i-1]
Srange=wavelen*r0/2/db

trial=np.arange(-Srange/2,Srange/2,dS).reshape([1,-1])
ntrial=trial.shape[1]

# generate steering matrix
print('calculate steering matrix')
bperp=bperp.reshape([-1,1])
steering=bperp*trial
steering=np.exp(steering*(1j)*4*3.14/wavelen/r0)
steeringH=np.conj(steering.T)


# start inversion
da=np.empty(shape=(lns,width))


print('start inversion ... ')
tomography=np.empty(shape=(lns,width,ntrial))


win=3
for j in range(lns):
    for i in range(width):
        j0=max(0,int(j-win/2))
        j1=min(lns,int(j+win/2))
        i0=max(0,int(i-win/2))
        i1=min(width,int(i+win/2))
        
        # multilook and normalization
        cpx=real[:,j0:j1,i0:i1]+1j*imag[:,j0:j1,i0:i1]
        cpx=np.average(np.average(cpx,axis=1),axis=1)
        norm=np.max(np.abs(cpx))
        cpx=cpx/norm
        cpx=cpx.reshape(-1,1)
        
        da[j,i]=utils.da(cpx)
        
        
        # calculate covariance matrix
        cpxH=np.conj(cpx.T)
        cov=np.dot(cpx,cpxH)/nslc
        
        load_factor=0.01
        cov+=np.eye(nslc)*load_factor

        try:
            invCov=inv(cov)
        except:
            tomography[j,i,:]=np.nan    
            continue
        denominator=np.dot(np.dot(steeringH,invCov),steering)
        # denominator=np.array([np.diag(denominator)]*nslc,dtype=np.complex64)
        # weight=np.dot(invCov,steering)/denominator
        # wH=np.conj(weight.T)
        # power=np.dot(np.dot(wH,cov),weight)
        tomography[j,i,:]=np.diag(1/denominator.real)


# verify result 
print('result output')
inten=np.log(np.average((real**2+real**2)**0.5,axis=0))
utils.exportPointHeight(tomography,lon,lat,trial.flatten(),0,da<0.4,'H1')

# dsm,lon,lat=utils.loadTif('dsmVerify.tif')

for testLine in range(0,real.shape[1],10):
    # plt.subplot(211)
    plt.imshow(tomography[testLine,:,::-1].T,cmap='rainbow')
    plt.colorbar(orientation='horizontal')
    # plt.subplot(212)
    # plt.plot(inten[testLine,:])

    plt.savefig(f'testImage/profile{testLine}')
    plt.close()


plt.subplot(121)
output=trial[0,np.argmax(tomography,axis=2)]*np.sin(np.radians(incAve))
output[da>0.6]=np.nan
plt.imshow(output,cmap='jet')
# plt.imshow(np.argmax(tomography,axis=2),cmap='jet')
plt.colorbar()
plt.subplot(122)
plt.imshow(inten)
plt.savefig('testImage/aveInt02')