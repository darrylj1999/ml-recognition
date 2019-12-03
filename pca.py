#!/usr/bin/env python3
import os
import sys
import imageio
import numpy as np
import matplotlib.pyplot as plot
from numpy.linalg import norm, eig
from compression import uncompress
# Standardize (transform to standard normal distribution)
""" mu = np.mean( vector )
sigma = np.std( vector )
return ( vector - mu ) / sigma """
# Normalize (unit vector l2 norm)
""" return vector / norm(vector) """
def normalize( vector ):
    # Scale (min = 0., max = 1.)
    return vector / 255.
# Decompose high dimensional vector into linear combination of k eigenvectors
def deconstruct( sample, k, eigenvectors ):
    coefficients = np.array([])
    for k_idx in range( num_components ):
        coefficients = np.append( coefficients, np.dot( v[k_idx], sample ) )
    return coefficients
# Retrieve high dimensional vector using linear combination of eigenvectors
def reconstruct( coefficients, eigenvectors ):
    reconstruction = coefficients[0]*v[0]
    for k_idx in range( 1, len(coefficients) ):
        reconstruction = reconstruction + coefficients[k_idx]*v[k_idx]
    return reconstruction
def pca( X ):
    # Get eigenvalues and eigenvectors of covariance matrix, X^T X / N
    covariance = np.matmul( samples.transpose(), samples ) / len(samples)
    l, v = eig( covariance )
    # Sort eigenvalues and eigenvectors by magnitude of eigenvalue
    real_l = np.abs(l)
    # Transpose so rows become eigenvectors
    v = v.transpose()
    real_l, l, v = zip( *sorted( zip(real_l, l, v), reverse=True ) )
    return l, v
if __name__ == "__main__":
    dataset = sys.argv[1:]
    # Make matrix of samples, X
    samples = []
    for samplefilename in dataset:
        sample = imageio.mimread(samplefilename)
        normalized_sample = normalize( sample[0][0] )
        samples.append( normalized_sample )
    samples = np.matrix( samples )
    # Mean for each normalized variable
    mean_sample = np.mean( samples, axis=0 )
    l, v = pca( samples )
    real_l = np.abs(l)
    # Find number of components which sum to give 99% of total energy
    total_energy = np.repeat( np.sum( real_l ), samples.shape[1] )
    threshold_energy = 0.99 * total_energy
    # Add one to transform from [0, N) to (0, N]
    k = np.argmin( np.square( np.cumsum(real_l) - threshold_energy ) ) + 1
    # Plot Cumulative Energy
    plot.figure(1)
    plot.semilogx( range(1, len(real_l)+1), real_l, 'r--', label='Eigenvalue $| \\lambda_k |$' )
    plot.semilogx( range(1, len(real_l)+1), np.cumsum( real_l ), 'r-', label='Cumulative Energy $e = \\sum_{{i=0}}^{{k-1}} | \\lambda_i |$' )
    plot.semilogx( range(1, len(real_l)+1), threshold_energy, 'b-' , label='$0.99E$' )
    plot.grid(True)
    plot.title( 'Components required to capture 99% of energy = '+str(k) )
    plot.xlabel('Number of Components ($k$)')
    plot.legend()
    plot.tight_layout()
    plot.savefig('a_iii.pdf')
    # Plot top 10 eigenfaces
    for idx in range( 10 ):
        representation = uncompress(np.abs(v[idx]))
        imageio.imwrite( 'vector'+str(idx+1)+'.gif', np.matrix(representation) * 255. )
    # Reconstruct random sample using `num_components`
    num_components = 50
    idx = np.random.randint( 0, samples.shape[0] )
    sample = np.asarray(samples[idx]).flatten()
    # Deconstruct
    coefficients = deconstruct( sample, num_components, v )
    # Reconstruct
    reconstruction = reconstruct( coefficients, v )
    reconstruction = uncompress( np.abs(reconstruction) )
    imageio.imwrite( str(idx) + '_' + str(num_components) + '_' + os.path.split(dataset[idx])[1], np.matrix(reconstruction) * 255. )