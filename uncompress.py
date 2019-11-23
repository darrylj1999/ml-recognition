#!/usr/bin/env python3
# Transform (1x900) to (30,30) GIF
import os
import sys
import imageio
import numpy as np
infilename = sys.argv[1]
outdirectory = './'
_, outfilename = os.path.split( infilename )
# Extract first frame from GIF (only one frame)
gif = imageio.mimread(infilename)[0]
# Extract columns and stack horizontally
dimensions = (30, 30)
columns = np.array([ gif[0,(i*dimensions[1]):(i*dimensions[1] + dimensions[0])] for i in range( dimensions[1] ) ])
columns = columns.transpose()
# Save output to file
imageio.imwrite( outdirectory + outfilename, columns )
print( infilename, '-->', outdirectory+outfilename )