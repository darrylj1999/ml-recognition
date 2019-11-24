#!/usr/bin/env python3
import sys
import imageio
import numpy as np
import matplotlib.pyplot as plot
from numpy.linalg import norm, eig
from compression import uncompress
dataset = sys.argv[1:]
# Standardize (transform to standard normal distribution)
""" mu = np.mean( vector )
sigma = np.std( vector )
return ( vector - mu ) / sigma """
# Normalize (unit vector l2 norm)
""" return vector / norm(vector) """
def normalize( vector ):
    # Scale (min = 0., max = 1.)
    return vector / 255.
# Make matrix of samples, X
samples = []
for samplefilename in dataset:
    sample = imageio.mimread(samplefilename)
    normalized_sample = normalize( sample[0][0] )
    samples.append( normalized_sample )
samples = np.matrix( samples )
# Get eigenvalues and eigenvectors of covariance matrix, X^T X / N
covariance = np.matmul( samples.transpose(), samples ) / len(samples)
l, v = eig( covariance )
# Sort eigenvalues and eigenvectors by magnitude of eigenvalue
real_l = np.abs(l)
# Transpose so rows become eigenvectors
v = v.transpose()
real_l, l, v = zip( *sorted( zip(real_l, l, v), reverse=True ) )
# Find number of components which sum to give 99% of total energy
total_energy = np.repeat( np.sum( real_l ), covariance.shape[1] )
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
for i in range( 10 ):
    representation = uncompress(np.abs(np.asarray(v[i])))
    imageio.imwrite( 'vector'+str(i+1)+'.gif', np.matrix(representation) * 255. )