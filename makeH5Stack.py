import h5py
import glob
from utils import loadSLC
import numpy as np 
def makeSLCStack(slcFolder,h5Name,subset=[]):
    dates=sorted(glob.glob(slcFolder+'/20*/*.slc'))
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
    
