import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

def cs_omp(signal,T_mat):
    nslc=signal.shape[0]
    niter=int(nslc/4)
    level=T_mat.shape[1]

    hat_x=np.zeros(shape=level,dtype=np.complex64)
    
    #initial residual
    T=T_mat.copy()
    r=signal.copy()
    aug_t=np.zeros(shape=(nslc,niter),dtype=np.complex64)

    # data nromalization is important
    norm=np.max(np.abs(signal)) 
    signal=signal/norm

    posArr=[]
    for i in range(niter):
        T_h=np.conj(T.T)
        pos=np.argmax(np.dot(T_h,r))
        posArr.append(pos)
        aug_t[:,i]=T[:,pos]

        # remove observation of  that position!
        T[:,pos]=0

        local_T=aug_t[:,:i+1]
        # aug_x=np.dot(np.linalg.pinv(local_T),signal)
        aug_x=scp.linalg.lstsq(local_T,signal)[0]
        r=signal-np.dot(local_T,aug_x)
        if np.max(np.abs(r))<1e-6: break
    hat_x[posArr]=aug_x
    return hat_x
        

if __name__ == "__main__":
    # np.random.seed(20)
    nsig=100
    nobs=200
    nsp=int(nsig/10)
    sparseSig=np.zeros(nsig,dtype=np.complex64)
    sparseLoc=np.random.rand(nsp)*nsig
    sparseSig[sparseLoc.astype(int)]=np.exp(1j*np.random.rand(nsp))*np.random.rand(nsp)

    T=np.exp(1j*(np.linspace(-10,10,nobs*nsig)/10*3.14)).reshape(nobs,nsig)+np.random.rand(nobs,nsig)


    # T=np.abs(T)
    # sparseSig=np.abs(sparseSig)

    obs=np.dot(T,sparseSig)
    reconst_sig=cs_omp(obs,T)
    plt.subplot(211)
    plt.plot(np.abs(sparseSig),label='orig')
    plt.subplot(212)
    plt.plot(np.abs(reconst_sig),label='omp')
    plt.savefig('testCS')
