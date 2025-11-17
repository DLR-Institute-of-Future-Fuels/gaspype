import numpy as np
from .typing import FloatArray


def null_space(A: FloatArray) -> FloatArray:
    """
    Compute an orthonormal basis for the null space of A using NumPy SVD.

    Args:
        A: Input matrix of shape (m, n)
    
    Return:
        Null space vectors as columns, shape (n, n - rank)
    """
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s, initial=0.) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q