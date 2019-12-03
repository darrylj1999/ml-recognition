#!/usr/bin/env python3
import os
import sys
import imageio
import numpy as np
import matplotlib.pyplot as plot
from numpy.linalg import norm, eig
from compression import uncompress
def pca( X ):
    # Get eigenvalues and eigenvectors of covariance matrix, X^T X / N
    covariance = np.matmul( X.transpose(), X ) / X.shape[0]
    l, v = eig( covariance )
    # Sort eigenvalues and eigenvectors by magnitude of eigenvalue
    real_l = np.abs(l)
    # Transpose so rows become eigenvectors
    v = v.transpose()
    real_l, l, v = zip( *sorted( zip(real_l, l, v), reverse=True ) )
    return l, v
# Decompose high dimensional vector into linear combination of k eigenvectors
def deconstruct( sample, k, eigenvectors ):
    coefficients = np.array([])
    for k_idx in range( k ):
        coefficients = np.append( coefficients, np.dot( eigenvectors[k_idx], sample ) )
    return coefficients
# Retrieve high dimensional vector using linear combination of eigenvectors
def reconstruct( coefficients, eigenvectors ):
    reconstruction = coefficients[0]*eigenvectors[0]
    for k_idx in range( 1, len(coefficients) ):
        reconstruction = reconstruction + coefficients[k_idx]*eigenvectors[k_idx]
    return reconstruction