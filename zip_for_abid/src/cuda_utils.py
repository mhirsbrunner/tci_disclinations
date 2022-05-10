import numpy as np
import numpy.linalg as nlg
from numba import cuda, complex64,


# Matrix Operator Utilities
@cuda.jit(
    'complex64[:,:](complex64, complex64[:,:], complex64[:,:])', device=True)
def mul_scalar_mat(a, B, C):
    """
        Multiply all elements in B by a scalar a and store into C
    """
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            C[i, j] = a * B[i, j]
    return C


@cuda.jit("complex64(complex64)", device=True)
def conjugate(num):
    '''
        Takes complex conjugate of a scalar and returns it's value
    '''
    return num.real - num.imag * 1j


@cuda.jit('complex64[:,:](complex64[:,:], complex64[:,:])', device=True)
def hconj_mat(A, B):  # Hermitian conjugate
    '''
        Takes complex conjugate of a matrix A and stores into B
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[j, i] = conjugate(A[i, j])
    return B


# Surface Green Function Code
@cuda.jit
def surface_green_function(energy, h00, h01, surf_pert, gs):
    it_max = 20
    tol = 1e-12

    temp = cuda.local.array(h00.shape, dtype=complex64)
    cuda.local.
    energy_mat = mul_scalar_mat(energy, energy_mat, energy_mat)

    eps_s = cuda.to_device(h00)
    eps = cuda.(h00)

    beta = cuda.device_array_like(h01)
    alpha = h01.conj().T


    it = 0
    alpha_norm = 1
    beta_norm = 1

    while alpha_norm > tol or beta_norm > tol:
        g0_alpha = nlg.solve(energy - eps, alpha)
        g0_beta = nlg.solve(energy - eps, beta)

        eps_s = eps_s + alpha @ g0_beta
        eps = eps + alpha @ g0_beta + beta @ g0_alpha

        alpha = alpha @ g0_alpha
        beta = beta @ g0_beta

        alpha_norm = nlg.norm(alpha)
        beta_norm = nlg.norm(beta)

        it += 1

        if it > it_max:
            print(f'Max iterations reached. alpha_norm: {alpha_norm}, beta_norm: {beta_norm}')
            break

    gs = nlg.inv(energy - eps_s - surf_pert)

    if return_bulk:
        gb = nlg.inv(energy - eps)
        return gs, gb
    else:
        return gs
