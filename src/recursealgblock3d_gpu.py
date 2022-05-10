# -*- coding: utf-8 -*-
"""
(c) Copyright: Computational Nanoelectronics Lab, University of Florida
Created on Thu Jan 28 23:19:54 2021
recrusive algorithm gpu block calculation module
@author: yangning
"""

import numpy as np
from numba import cuda, complex64, float32, int32

@cuda.jit('complex64[:,:](complex64[:,:], complex64[:,:], complex64[:,:])', device=True)
def inverse_matrix(mat, B, A):
    EPS = 1.0e-19
    n = mat.shape[0]
    # enlarge original matrix A = [ mat , I ]
    for pivot in range(n):
        for j in range(n):
            A[pivot,j] = mat[pivot,j]
            if (pivot != j):
                A[pivot,j+n] = 0+0j
            else:
                A[pivot,j+n] = 1+0j    
    #A = np.insert(mat, n, values=np.eye(n), axis=1) # enlarge original matrix A = [ mat , I ]
    for pivot in range(n):
        # find max value then swap rows
        if(pivot < n - 1):
            maxrow = pivot
            maxval = abs(A[pivot, pivot])
            for row in range(pivot + 1, n):  
                val = abs(A[row, pivot])
                if(val > maxval):
                    maxval = val
                    maxrow = row
            if(maxrow != pivot):
                nn = A.shape[1]
                for x in range(nn):
                    tmp = A[pivot, x]
                    A[pivot, x] = A[maxrow, x]
                    A[maxrow, x] = tmp           
 
        coef = 1.0 / A[pivot, pivot]
        if abs(coef) > EPS:
            for col in range(pivot, 2 * n):
                A[pivot, col] = coef * A[pivot, col] 
 
        for row in range(n):
            if row == pivot:
                continue
            coef = 1.0 * A[row, pivot]
            if abs(coef) > EPS:
                for col in range(pivot, 2 * n):
                    A[row, col] -= coef * A[pivot, col]
 
    # return right part of A
    for pivot in range(n):
        for j in range(n):
            B[pivot,j] = A[pivot, n+j]
    return B

@cuda.jit(
    'complex64[:,:](complex64[:,:], complex64[:,:], complex64[:,:])', device=True)
def mul_mat(A, B, C):
    '''
        Matrix multiply two 4x4 matrix (A.dot(B)) and store it in C, returing reference to C
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            tmp = 0
            for k in range(B.shape[0]):
                tmp = A[i, k] * B[k, j] + tmp  
            C[i, j] = tmp
    return C

@cuda.jit(
    'complex64(complex64, complex64, complex64)', device=True)
def add_num(a, b, c):
    '''
        Matrix multiply two 2x2 matrix (A.dot(B)) and store it in C, returing reference to C
    '''
    c=a+b
    return c


@cuda.jit(
    'complex64[:,:](complex64[:,:], complex64[:,:], complex64[:,:])', device=True)
def sub_mat(A, B, C):
    '''
        Subtract 2x2 matrix B from A and store in C
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i, j] = A[i, j] - B[i, j]
    return C

@cuda.jit(
    'void(complex64[:,:], complex64[:,:], complex64[:,:])', device=True)
def add_mat(A, B, C):
    '''
        Add 2x2 matrix B from A and store in C
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i, j] = A[i, j] + B[i, j]


@cuda.jit('void(complex64[:,:], complex64[:,:])', device=True)
def set_mat(B, A):
    '''
        Copies elements from A into B
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i, j] = A[i, j]


@cuda.jit('void(complex64[:,:])', device=True)
def neg_mat(A):
    '''
        Negates all the elements in a matrix
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = -A[i, j]


@cuda.jit(
    'complex64[:,:](complex64[:,:], complex64[:,:], complex64[:,:])', device=True)
def pw_mul(A, B, C):
    '''
        Piecewise multiply A and B and store into C
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i, j] = A[i, j] * B[i, j]
    return C

@cuda.jit(
    'complex64[:,:](complex64, complex64[:,:], complex64[:,:])', device=True)
def add_scalar_mat(a, B, C):
    '''
        Add all elements in B by a scalar a and store into C
    '''
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            C[i, j] = a + B[i, j]
    return C


@cuda.jit(
    'complex64[:,:](complex64, complex64[:,:], complex64[:,:])', device=True)
def mul_scalar_mat(a, B, C):
    '''
        Multiply all elements in B by a scalar a and store into C
    '''
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            C[i, j] = a * B[i, j]
    return C

@cuda.jit('float32[:](complex64[:,:],float32[:])', device=True)
def abs_err(A, error):  # error needs to be an array to output a value
    '''
        absolute value sum for error assessment
    '''
    tmp=0.0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            num=A[i,j]  # complex64 value
            tmp += (num.real*num.real+num.imag*num.imag)**0.5
            
    error[0]=tmp
    return error

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

@cuda.jit
def recursealgblock3d(AD, ALD, AUD, Sigin, Sigout, gaL, grL, ginL, gipL, Grl, Grd, Gru, 
                 Gnd, Gnu, Gnl, Al_cr, Ad_cr, Au_cr, Gal, Gad, Gau):
    ################################################################
    # Title:        recursealgblock3d.m
    # function [Grl,Grd,Gru,Gnl,Gnd,Gnu,Gpl,Gpd,Gpu] = recursealg3d(Np,Al,Ad,Au,Sigin,Sigout)
    # recursive algorithm to solve for the diagonal elements only of
    # the Non-equilibrium Green's function
    # HANDLES MATRICES BY 3 DIAGONALS
    # Grl,Grd,Gru = retarded Green's function
    # Gnl,Gnd,Gnu = electron Green's function
    # Gpl,Gpd,Gpu = hole Green's function
    # Np = size of the matrices
    # Al,Ad,Au = matrix of coefficients
    # Sigin = matrix of in-scattering self-energies (diagonal)
    # Sigout = matrix of out-scattering self-energies (diagonal)
    
    Ne=Grl.shape[1]
    Np=1+Grl.shape[0]
    
    i = cuda.grid(1)
   
    if i > Ne:
        return

    for q in range(Np): # Hermitian conjugate
        hconj_mat(AUD[q, i], Al_cr[q, i])
        hconj_mat(AD[q, i], Ad_cr[q, i])
        hconj_mat(ALD[q, i], Au_cr[q, i])

    # step 1 initialization
    # replace 4 with integer N (8 for 2N) to fit your input Hamiltonian N-dimensional submatrix below
    tmpinv = cuda.local.array((4, 8), dtype=complex64)
    tmp = cuda.local.array((4, 4), dtype=complex64)
    tmp1 = cuda.local.array((4, 4), dtype=complex64)
    sla2 = cuda.local.array((4, 4), dtype=complex64)
    prom = cuda.local.array((4, 4), dtype=complex64)
    tmp3 = cuda.local.array((4, 4), dtype=complex64)
    tmp4 = cuda.local.array((4, 4), dtype=complex64)
    
    inverse_matrix(AD[0, i], grL[0, i], tmpinv)
    
    # obtain the left-connected function
    for q in range(1, Np):
        mul_mat(ALD[q-1, i], grL[q-1, i], tmp)
        mul_mat(tmp, AUD[q-1, i], tmp1)
        sub_mat(AD[q, i], tmp1, tmp)
        inverse_matrix(tmp, grL[q, i], tmpinv)

    for q in range(Np):
        hconj_mat(grL[q, i], gaL[q, i])

    set_mat(Grd[Np-1, i], grL[Np-1, i])   # step2
    for q in range(Np-2, -1, -1):   # obtain off diagonal and diagonal of Gr
        mul_mat(Grd[q+1, i], ALD[q, i], tmp)
        mul_mat(tmp, grL[q, i], Grl[q, i])
        mul_scalar_mat(-1, Grl[q, i], Grl[q, i])

        mul_mat(grL[q, i], AUD[q, i], tmp)
        mul_mat(tmp, Grd[q+1, i], Gru[q, i])
        mul_scalar_mat(-1, Gru[q, i], Gru[q, i])

        mul_mat(grL[q, i], AUD[q, i], tmp)
        mul_mat(tmp, Grl[q, i], tmp1)
        sub_mat(grL[q, i], tmp1, Grd[q, i])

    for q in range(Np):   # calculate Ga
        # advanced Green's function
        hconj_mat(Grd[q, i], Gad[q, i])
        if q < Np-1:
            hconj_mat(Gru[q, i], Gal[q, i])
            hconj_mat(Grl[q, i], Gau[q, i])

    mul_mat(grL[0, i],Sigin[0, i], tmp)    # step3
    mul_mat(tmp, gaL[0, i], ginL[0, i])

    for q in range(1, Np):
        mul_mat(ALD[q-1, i], ginL[q-1, i], tmp)
        mul_mat(tmp, Au_cr[q-1, i], sla2)
        #sla2 = ALD[q-1,i]*ginL[q-1, i]*Au_cr[q-1, i]
        add_mat(Sigin[q, i], sla2, prom)

        #prom = Sigin[q, i] + sla2
        mul_mat(grL[q, i], prom, tmp)
        mul_mat(tmp, gaL[q, i], ginL[q, i]) # left-connected in-scattering

    set_mat(Gnd[Np-1, i], ginL[Np-1, i])   # step 4

    for q in range(Np-2, -1, -1):
        mul_mat(Grd[q+1, i], ALD[q, i], tmp)
        mul_mat(tmp, ginL[q, i], tmp1)
        mul_scalar_mat(-1, tmp1, tmp1)

        mul_mat(Gnd[q+1, i], Al_cr[q, i], tmp)
        mul_mat(tmp, gaL[q, i], tmp3)
        sub_mat(tmp1, tmp3, Gnl[q, i])

        mul_mat(grL[q, i], AUD[q, i], tmp)
        mul_mat(tmp, Gnd[q+1, i], tmp1)
        mul_mat(tmp1, Al_cr[q, i], tmp)
        mul_mat(tmp, gaL[q, i], tmp1)  # grL[q]@Au[q]@Gnd[q+1]@Al[q]@grL[q].T.conj()

        mul_mat(ginL[q, i], Au_cr[q, i], tmp)
        mul_mat(tmp, Gal[q, i], tmp3) # ginL[q]@Au[q]@Gru[q].conj().T

        mul_mat(Gru[q, i], ALD[q, i], tmp)
        mul_mat(tmp, ginL[q, i], tmp4)  # Gru[q]@Al[q]@ginL[q]

        add_mat(ginL[q, i], tmp1, tmp)
        add_mat(tmp3, tmp4, tmp1)
        sub_mat(tmp, tmp1, Gnd[q, i])

        hconj_mat(Gnl[q, i], Gnu[q, i])
    
    return
