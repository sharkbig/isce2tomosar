import numpy as np
import utils
import matplotlib.pyplot as plt
import os
from scipy.linalg import lstsq,inv

projectFolder='/RUNDATA_16804/junyan1998/Chehualin/H4-10/crop_H4-10'
stackName='test_slcStack.h5'

wavelen=0.031
H=620e3  

geomFolder=os.path.join(projectFolder,'geom_reference')
bperpFolder=os.path.join(projectFolder,'baselines')
slcFolder=os.path.join(projectFolder,'SLC')
# load SLC and make StackFile
if os.path.exists(stackName):
    pass
else:
    from makeH5Stack import makeSLCStack
    print('make SLC stack')
    makeSLCStack(slcFolder,stackName)

# load longtitude and latitdue
lon=utils.loadGeom(geomFolder,'lon.rdr')
lat=utils.loadGeom(geomFolder,'lat.rdr')
lon=lon[1800:2400,3800:4100]
lat=lat[1800:2400,3800:4100]

# load Data 
print('load data')
ds=utils.loadStack(stackName)
real=ds['slc_real'][:-2,1800:2400,3800:4100]
imag=ds['slc_imag'][:-2,1800:2400,3800:4100]


nslc=real.shape[0]
lns=real.shape[1]
width=real.shape[2]


# load baselines 
if os.path.isfile('bp.npy'):
    bperp=np.load('bp.npy')
else:
    bperp=utils.loadBperp(bperpFolder)
    np.save('bp',bperp)
    print('write to file bp.npy')
bperp=bperp[:-2] # some problem with the data writing ... 


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
incAve=np.average(inc)
r0=utils.calcSR(incAve,H)

# calculate baseline aperture
dS=wavelen*r0/2/(np.max(bperp)-np.min(bperp))
print(dS)
ntrial=50 
trial=np.linspace(-dS/2,dS/2,ntrial)

# generate steering matrix
print('calculate steering matrix')
steering=np.ones(shape=(nslc,ntrial))
steering=bperp*steering.T
steering=steering.T*trial
steering=np.exp(steering*(1j)*4*3.14/wavelen/r0/np.sin(np.radians(incAve)))
steeringH=np.conj(steering.T)


# start inversion
print('start inversion ... ')
tomography=np.empty(shape=(lns,width,ntrial))
win=3
da=np.empty(shape=(lns,width))
for j in range(lns):
    for i in range(width):
        j0=max(0,int(j-win/2))
        j1=min(lns,int(j+win/2))
        i0=max(0,int(i-win/2))
        i1=min(width,int(i+win/2))
        
        # multilook
        cpx=real[:,j0:j1,i0:i1]+1j*imag[:,j0:j1,i0:i1]
        cpx=np.average(np.average(cpx,axis=1),axis=1)
        da[j,i]=utils.da(cpx)

        # data normalization
        norm=np.max(np.abs(cpx))
        cpx=cpx.reshape(-1,1)/norm

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
        base=np.dot(np.dot(steeringH,invCov),steering)
        _weight=np.array([np.diag(base)]*nslc,dtype=np.complex64)
        weight=np.dot(invCov,steering)/_weight
        wH=np.conj(weight.T)

        power=np.dot(np.dot(wH,cov),weight)
        tomography[j,i,:]=np.diag(np.abs(power))
        # tomography[j,i,:]=np.diag(power.real)
  

# verify result 
inten=np.log(np.average((real**2+real**2)**0.5,axis=0))
utils.exportPointHeight(tomography,lon,lat,trial,0,da<0.4,'H1')
# dsm,lon,lat=utils.loadTif('dsmVerify.tif')

for testLine in range(0,real.shape[1],10):
    plt.subplot(211)
    plt.imshow(tomography[testLine,:,::-1].T,cmap='rainbow')
    plt.colorbar(orientation='horizontal')
    plt.subplot(212)
    plt.plot(inten[testLine,:])
    plt.savefig(f'testImage/profile{testLine}')
    plt.close()

plt.subplot(121)
output=trial[np.argmax(tomography,axis=2)]*np.sin(np.radians(incAve))
output[da>0.6]=np.nan
plt.imshow(output,cmap='jet')
# plt.imshow(np.argmax(tomography,axis=2),cmap='jet')
plt.colorbar()
plt.subplot(122)
plt.imshow(inten)
plt.savefig('testImage/aveInt')
