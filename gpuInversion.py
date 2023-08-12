import cupy as cp
import numpy as np
from cupyx.scipy import signal
import scipy
def da_gpu(slcStack):
    amp=cp.abs(slcStack)
    std=cp.std(amp,axis=0)
    ave=cp.average(amp,axis=0)
    da=std/ave
    return da


def da(slcStack,axis=0):
    amp=np.abs(slcStack)
    std=np.std(amp,axis=axis)
    ave=np.average(amp,axis=axis)
    da=std/ave
    return da

def gpu_moving_average_2d(arr_cpu, window_size):
    """
    input: cpu arr
    output: gpu arr
    """
    arr_gpu=cp.asarray(arr_cpu)
    patch=arr_cpu.shape[0]
    # window = scipy.signal.gaussian(window_size,1).reshape(-1,1)
    # window = window*window.T
    # window_ave = cp.asanyarray(window,dtype=cp.complex64)


    window_ave = cp.ones((window_size, window_size), dtype=cp.complex64) / (window_size**2)
    # Perform 2D convolution to calculate the moving average
    
    result_gpu=cp.empty(shape=arr_gpu.shape,dtype=cp.complex64)
    for i in range(patch):
        # result_gpu[i,...]=signal.convolve2d(arr_gpu[i,...], window, mode='same')
        result_gpu[i,...]=signal.convolve2d(arr_gpu[i,...], window_ave, mode='same')
    return result_gpu



def gpuBFinversion(cpxGpu,steering):
    """
    input: gpu array 
    output: cpu array
    """
    steeringGpu=cp.asarray(steering)
    patch,ydim,xdim=cpxGpu.shape

    level=steering.shape[1]
    # cpxGpu=cp.asarray(cpxCpu)
    cpxGpu=cpxGpu.reshape(patch,ydim*xdim)

    # start calculation    
    norm=cp.max(cp.abs(cpxGpu),axis=0)
    cpxGpu=cpxGpu/norm
    cpxGpu=cpxGpu.T
    cpxGpu=cpxGpu[:,:,cp.newaxis]
    
    

    cov=cp.einsum('ijk,ilk->ijl',cpxGpu,cp.conj(cpxGpu))
    cov+=cp.eye(patch)*0.01

    covI=cp.linalg.inv(cov)
    power=cp.einsum('jl,ijk->ilk',cp.conj(steeringGpu),covI)
    power=cp.einsum('ijk,kl->ijl',power,steeringGpu)
    power=1/cp.diagonal(power,0,1,2)
    power=power.reshape(ydim,xdim,level)
    return cp.abs(power)



def BFinversion(cpxArray,steering):
    patch,ydim,xdim=cpxArray.shape
    level=steering.shape[1]
    cpx=cpxArray.reshape(patch,ydim*xdim)

    # start calculation
    norm=np.max(np.abs(cpx),axis=0)
    cpx=cpx/norm
    cpx=cpx.T
    
    cov=np.einsum('ij,il->ijl',cpx,np.conj(cpx))/patch
    load=0.01*np.eye(patch)
    cov+=load
    covI=np.linalg.inv(cov)
    
    power=np.einsum('jl,ijk->ilk',np.conj(steering),covI)
    power=np.einsum('ijk,kl->ijl',power,steering)
    power=1/np.diagonal(power,0,1,2)
    power=power.reshape(ydim,xdim,level)
    return np.abs(power)

def gpuCalcCoh(tomography,steering,gaussian):
    k=cp.argmax(tomography,axis=2)
    s_gpu=cp.asarray(steering)
    sig=cp.exp(1j*cp.angle(gaussian))
    L_g=cp.einsum('ij,jkl->ikl',s_gpu.conj().T,sig)
    L_k=cp.einsum('ij,jkl->ikl',s_gpu.conj().T,s_gpu[:,k])
    coh=cp.sum(L_g*L_k.conj(),axis=0)/cp.sum(L_k*L_k.conj(),axis=0)**0.5/cp.sum(L_g*L_g.conj(),axis=0)**0.5    
    return cp.abs(coh)


def CalcCoh(tomography,steering,gaussian):
    k=np.argmax(tomography,axis=2)
    sig=np.exp(1j*np.angle(gaussian))

    L_g=np.einsum('ij,jkl->ikl',steering.conj().T,sig)
    L_k=np.einsum('ij,jkl->ikl',steering.conj().T,steering[:,k])
    coh=np.sum(L_g*L_k.conj(),axis=0)/np.sum(L_k*L_k.conj(),axis=0)**0.5/np.sum(L_g*L_g.conj(),axis=0)**0.5    
    return np.abs(coh)

#### original code 
# for j in range(lns):
#     for i in range(width):
#         j0=max(0,int(j-win/2))
#         j1=min(lns,int(j+win/2))
#         i0=max(0,int(i-win/2))
#         i1=min(width,int(i+win/2))
        
#         # multilook and normalization
#         cpx=real[:,j0:j1,i0:i1]+1j*imag[:,j0:j1,i0:i1]
#         cpx=np.average(np.average(cpx,axis=1),axis=1)
#         norm=np.max(np.abs(cpx))
#         cpx=cpx/norm
#         cpx=cpx.reshape(-1,1)
        
#         da[j,i]=utils.da(cpx)
        
        
#         # calculate covariance matrix
#         cpxH=np.conj(cpx.T)
#         cov=np.dot(cpx,cpxH)/nslc
        
#         load_factor=0.025
#         cov=cov+np.eye(nslc)*load_factor
#         invCov=inv(cov)
#         denominator=np.dot(np.dot(steeringH,invCov),steering)
#         # denominator=np.array([np.diag(denominator)]*nslc,dtype=np.complex64)
#         # weight=np.dot(invCov,steering)/denominator
#         # wH=np.conj(weight.T)
#         # power=np.dot(np.dot(wH,cov),weight)
#         tomography[j,i,:]=np.log10(np.diag(1/denominator.real))
