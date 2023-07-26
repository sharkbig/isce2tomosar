from osgeo import gdal
import h5py 
import os
import glob 
import numpy as np

def loadSLC(file):
    dat=gdal.Open(file)
    band=dat.GetRasterBand(1)
    return dat,dat.RasterXSize, dat.RasterYSize


def loadTif(file):
    dat=gdal.Open(file)
    band=dat.GetRasterBand(1)
    gt=dat.GetGeoTransform()
    lon=np.linspace(gt[0],gt[0]+gt[1]*dat.RasterXSize,dat.RasterXSize)
    lat=np.linspace(gt[3],gt[3]+gt[5]*dat.RasterYSize,dat.RasterYSize)
    return dat,lon,lat

def loadStack(stack,subset=[],verbose=False,key='slc'):
    ds=h5py.File(stack)    
    if verbose:
        print(ds.keys())
    data=ds['slc']
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

def subsetDataset(data,subset,axis=0):
    p,q,m,n=subset
    if data.ndim==2:
        return data[p:q,m:n]
    if axis==0:
        return data[:,p:q,m:n]
    elif axis==-1:
        return data[p:q,m:n,:]
    else:
        raise ValueError("axis must be 0 or -1")
    

def loadGeom(geomFolder,product='los.rdr',subset=[]):
    losPath=os.path.join(geomFolder,product)
    los=gdal.Open(losPath)
    band=los.GetRasterBand(1)

    if len(subset) == 4:
        p,q,m,n=subset
    else:
        p,q,m,n=0,los.RasterYSize,0,los.RasterXSize

    return band.ReadAsArray()[p:q,m:n]

def loadBperp(bperpFolder):
    dateList=sorted(glob.glob(bperpFolder+'/*/20??????'))
    bp=np.empty(shape=len(dateList))
    ii= 0
    for i in dateList:
        dst=gdal.Open(i)
        band=dst.GetRasterBand(1)
        bp[ii]=np.average(band.ReadAsArray())
        ii+=1
        dst=None
    return bp


def calcSR(inc,H=600e3):
    if inc>3.14:
        print('Warning: the unit of incident angle should be in radians')
    avgInc=3.14-inc
    Re=6371e3
    b=-2*Re*np.cos(avgInc)
    c=-(2*Re*H+H**2)
    SR=(-b+(b**2-4*c)**0.5)/2
    return SR

def da(slcStack):
    amp=np.abs(slcStack)
    std=np.std(amp,axis=0)
    ave=np.average(amp,axis=0)
    da=std/ave
    return da

def exportPointHeight(tomo,lon,lat,trial,powerThreshold,dmask,outName='maxOutput'):
    tmask=np.max(tomo,axis=2)>powerThreshold
    mask=dmask*tmask
    h=trial[np.argmax(tomo,axis=2)]
    h=h[mask]
    lon=lon[mask]
    lat=lat[mask]
    np.savetxt(outName,np.c_[lon,lat,h])
    

def baselineInverval(bperp):
    db=-np.inf
    sortB=sorted(bperp) 
    for i in range(len(bperp)-1):
        db=max(db,sortB[i+1]-sortB[i])
    return db

