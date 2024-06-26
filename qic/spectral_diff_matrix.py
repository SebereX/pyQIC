#!/usr/bin/env python3

"""
This module contains a subroutine for making spectral differentiation matrices.
"""

import numpy as np
from scipy.linalg import toeplitz

def spectral_diff_matrix(n, xmin=0, xmax=2*np.pi):
    """
    Return the spectral differentiation matrix for n grid points
    on the periodic domain [xmin, xmax). This routine is based on the
    matlab code in the DMSuite package by S.C. Reddy and J.A.C. Weideman, available at
    http://www.mathworks.com/matlabcentral/fileexchange/29
    or here:
    http://dip.sun.ac.za/~weideman/research/differ.html  
    """

    h = 2 * np.pi / n
    kk = np.arange(1, n)
    n1 = int(np.floor((n - 1) / 2))
    n2 = int(np.ceil((n - 1) / 2))
    if np.mod(n, 2) == 0:
        topc = 1 / np.tan(np.arange(1, n2 + 1) * h / 2)
        temp = np.concatenate((topc, -np.flip(topc[0:n1])))
    else:
        topc = 1 / np.sin(np.arange(1, n2 + 1) * h / 2)
        temp = np.concatenate((topc, np.flip(topc[0:n1])))

    col1 = np.concatenate(([0], 0.5 * ((-1) ** kk) * temp))
    row1 = -col1
    D = 2 * np.pi / (xmax - xmin) * toeplitz(col1, r=row1)
    return D

def spectral_diff_matrix_extended(n, xmin=0, xmax=2*np.pi):
    """
    Return the spectral differentiation matrix for n grid points
    on the half periodic domain [xmin, xmax). This routine is based on the
    matlab code in the DMSuite package by S.C. Reddy and J.A.C. Weideman, available at
    http://www.mathworks.com/matlabcentral/fileexchange/29
    or here:
    http://dip.sun.ac.za/~weideman/research/differ.html  
    """

    h = 2*np.pi / n
    kk = np.arange(1, n)
    n1 = int(np.floor((n - 1) / 2))
    n2 = int(np.ceil((n - 1) / 2))
    if np.mod(n, 2) == 0:
        topc = 1 / np.sin(np.arange(1, n2 + 1) * h / 2)
        temp = np.concatenate((topc, np.flip(topc[0:n1])))
    else:
        topc = 1 / np.tan(np.arange(1, n2 + 1) * h / 2)
        temp = np.concatenate((topc, -np.flip(topc[0:n1])))

    col1 = np.concatenate(([0], 0.5 * ((-1) ** kk) * temp))
    row1 = -col1
    D = 2 * np.pi / (xmax - xmin) * toeplitz(col1, r=row1)
    return D

def finite_difference_matrix(N, order=2):
    """
    Creates a finite difference differentiation matrix for periodic functions.

    Parameters:
     - N (int): Number of grid points.
     - order (int): Order of accuracy of the finite difference scheme (2, 4, or 6).

    Returns:
     - numpy.ndarray: Differentiation matrix of size (N, N).
    """
    # Grid spacing
    h = 2 * np.pi/N

    D = np.zeros((N, N))
    
    if order == 2:
        # Second-order central finite difference
        for i in range(N):
            D[i, (i-1) % N] = -1 / (2 * h)
            D[i, (i+1) % N] = 1 / (2 * h)
    
    elif order == 4:
        # Fourth-order central finite difference
        for i in range(N):
            D[i, (i-2) % N] = 1 / (12 * h)
            D[i, (i-1) % N] = -8 / (12 * h)
            D[i, (i+1) % N] = 8 / (12 * h)
            D[i, (i+2) % N] = -1 / (12 * h)
    
    elif order == 6:
        # Sixth-order central finite difference
        for i in range(N):
            D[i, (i-3) % N] = -1 / (60 * h)
            D[i, (i-2) % N] = 3 / (20 * h)
            D[i, (i-1) % N] = -3 / (4 * h)
            D[i, (i+1) % N] = 3 / (4 * h)
            D[i, (i+2) % N] = -3 / (20 * h)
            D[i, (i+3) % N] = 1 / (60 * h)
    
    else:
        raise ValueError("Unsupported order. Choose 2, 4, or 6.")
    
    return D