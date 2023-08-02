import h5py
import glob
from utils import loadSLC,loadGeom
import numpy as np


def makeSLCStack(slcFolder,h5Name,subset=[],suffix='.slc'):
    dates=sorted(glob.glob(slcFolder+'/20*/*'+suffix))
    print(dates)
    
    _,x,y=loadSLC(dates[0])
    nslc=len(dates)

    if len(subset)==4:
        p,q,m,n=subset
        ysize=q-p
        xsize=n-m
    elif len(subset) == 0:
        p=m=0
        ysize=q=y
        xsize=n=x
    else:
        print('wrong subset area\nLoading hole image')
        p=m=0
        ysize=q=y
        xsize=n=x
    print(ysize,xsize)
    
    # create dataset
    ds=h5py.File(h5Name,'w')
    # realpart=ds.create_dataset('slc_real',(nslc,ysize,xsize),dtype=np.float32)
    # imagpart=ds.create_dataset('slc_imag',(nslc,ysize,xsize),dtype=np.float32)
    slcLayer=ds.create_dataset('slc',(nslc*2,ysize,xsize),chunks=(1,ysize,xsize),
        dtype=np.float32)

    for i in range(nslc):
        ds,*_ = loadSLC(dates[i])
        cpx=ds.ReadAsArray()
        # realpart[i,:,:]=cpx.real[p:q,m:n]
        # imagpart[i,:,:]=cpx.imag[p:q,m:n]
        slcLayer[i,:,:]=cpx.real[p:q,m:n]
        slcLayer[i+nslc,:,:]=cpx.imag[p:q,m:n]
        ds=None 
        print('slc process:',dates[i])
    
def makeGeomStack(geomFolder,h5Name='geom.h5',subset=[],suffix='.rdr'):
    lon=loadGeom(geomFolder,'lon.rdr',subset)
    lat=loadGeom(geomFolder,'lat.rdr',subset)
    inc=loadGeom(geomFolder,'los.rdr',subset,band=1)
    
    ysize,xsize=lon.shape

    ds=h5py.File(h5Name,'w')
    lonLayer=ds.create_dataset('lon',(ysize,xsize),chunks=(ysize,xsize))
    latLayer=ds.create_dataset('lat',(ysize,xsize),chunks=(ysize,xsize))
    incLayer=ds.create_dataset('inc',(ysize,xsize),chunks=(ysize,xsize))

    lonLayer[...]=lon
    latLayer[...]=lat  
    incLayer[...]=inc
    ds.close()

def loadSLCStack(stack,subset=[],verbose=False,key='slc'):
    ds=h5py.File(stack)    
    if verbose:
        print(ds.keys())
    data=ds[key]
    nslc=int(data.shape[0]/2)
    ny=data.shape[1]
    nx=data.shape[2]
    if len(subset) == 0:
        p,q,m,n=0,ny,0,nx
    else:
        p,q,m,n=subset
    real=data[0:nslc,p:q,m:n]
    imag=data[nslc:2*nslc,p:q,m:n]

    return real,imag

def loadGeomStack(stack,key,subset=[],verbose=False):
    ds=h5py.File(stack)    
    if verbose:
        print(ds.keys())
    data=ds[key]
    ny,nx =data.shape

    if len(subset) == 0:
        p,q,m,n=0,ny,0,nx
    else:
        p,q,m,n=subset
    data=data[p:q,m:n]
    ds.close()    

    return data