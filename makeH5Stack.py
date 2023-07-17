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
        y=q-p
        x=n-m
    elif len(subset) == 0:
        p=m=0;q=y;n=x
    else:
        print('wrong subset area\nLoading hole image')
        p=m=0;q=y;n=x
    # create dataset
    ds=h5py.File(h5Name,'w')
    realpart=ds.create_dataset('slc_real',(nslc,y,x),dtype=np.float32)
    imagpart=ds.create_dataset('slc_imag',(nslc,y,x),dtype=np.float32)
    
    for i in range(nslc):
        ds,*_ = loadSLC(dates[i])
        cpx=ds.ReadAsArray()
        realpart[i,p:q,m:n]=cpx.real
        imagpart[i,p:q,m:n]=cpx.imag
        ds=None 
        print('slc process:',dates[i])
    
