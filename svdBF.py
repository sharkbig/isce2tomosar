import scipy as scp
import numpy as np

# this is the truncated svd with
# wienner filter for singular value

def svdBF(signal,T,rcond=2,noise=0):
    """
    input: 
    signal[m,n]: 
    """
    
    norm=np.max(np.abs(signal),axis=0)
    signal=signal/norm
    
    #### apply point-wise noise estimation 
    # nsig=signal.shape[1]
    # m,n=T.shape
    u,s,vh=scp.linalg.svd(T)
    # ne=np.count_nonzero(s>rcond)
    # noise=np.abs(np.dot(np.conj(u.T),signal)[ne:])
    # noise=(np.sum(noise**2,axis=0)*m/(m-ne))**0.5
    # print(noise)
    # tomography=np.empty(shape=(nsig,n))
    # for i in range(nsig):
    #     s_modified=np.where(s>rcond,s/(s**2+noise[i]**2),0)    
    #     invt=np.dot(np.conj(vh.T)[:,:m],np.dot(np.diag(s_modified),np.conj(u.T)))
    #     tomography[i,:]=np.dot(invt,signal[:,i]).T

    # print(s)
    invt=np.linalg.pinv(T,rcond=rcond)
    tomography=np.dot(invt,signal).T

    return tomography.real
