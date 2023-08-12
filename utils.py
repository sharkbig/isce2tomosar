from osgeo import gdal
import h5py 
import os
import glob 
import numpy as np
from datetime import datetime

def loadSLC(file):
    dat=gdal.Open(file)
    band=dat.GetRasterBand(1)
    return dat,dat.RasterXSize, dat.RasterYSize


def loadTif(file):
    dat=gdal.Open(file)
    band=dat.GetRasterBand(1)
    gt=dat.GetGeoTransform()
    lon=np.linspace(gt[0],gt[0]+gt[1]*dat.RasterXSize,dat.RasterXSize)
    lat=np.linspace(gt[3],gt[3]+gt[5]*dat.RasterXSize,dat.RasterYSize)
    return dat,lon,lat



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
    

def loadGeom(geomFolder,product='los.rdr',subset=[],band=1):
    losPath=os.path.join(geomFolder,product)
    los=gdal.Open(losPath)
    band=los.GetRasterBand(band)

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


def exportPointHeight(tomo,lon,lat,outName='maxOutput'):
    nanmask=~np.isnan(tomo)
    h=tomo[nanmask]
    lon=lon[nanmask]
    lat=lat[nanmask]
    np.savetxt(outName,np.c_[lon,lat,h])
    

def baselineInverval(bperp):
    db=-np.inf
    sortB=sorted(bperp) 
    for i in range(len(bperp)-1):
        db=max(db,sortB[i+1]-sortB[i])
    return db

def temporalBaseline(slcFolder,refIx=0):
    date=sorted(os.listdir(slcFolder))
    refdate=datetime.strptime(date[refIx],'%Y%m%d')
    tbl=np.zeros(shape=len(date),dtype=int)
    ix=0
    for i in date:
        tbl[ix]=(datetime.strptime(i,'%Y%m%d')-refdate).days
        ix+=1
    return tbl
