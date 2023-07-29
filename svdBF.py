import scipy as scp
import numpy as np

# svd it's not able solve the super-resolution solution.
# And it will lead to the under-determined for "pinv"

def svdBF(signal,T,rcond=2,calcSig=False):

    norm=np.max(np.abs(signal),axis=0)
    signal=signal/norm
    if calcSig:
        u,s,vh=scp.linalg.svd(T)
        print(s)
    invt=scp.linalg.pinv(T,rcond=rcond)
    tomography=np.einsum('ij,jk->ik',invt,signal).T

    return tomography.real