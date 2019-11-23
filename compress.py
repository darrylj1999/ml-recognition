#!/usr/bin/env python3
# Compress and store as (1x900) GIF
import os
import sys
import imageio
import cv2 as cv
import numpy as np
infilename = sys.argv[1]
outdirectory = './dataset/compressed/'
_, outfilename = os.path.split( infilename )
# Extract first frame from GIF (only one frame)
gif = imageio.mimread(infilename)[0]
# Shrink into 30x30 GIF
dimensions = (30, 30)
resized_gif = cv.resize( gif, dimensions, interpolation=cv.INTER_AREA )
# Stack columns one after the other
stacked_gif = np.matrix( np.concatenate( tuple( resized_gif[:,idx] for idx in range(dimensions[1]) ) ) )
# Save output to file
imageio.mimwrite( outdirectory + outfilename, stacked_gif )
print( infilename, '-->', outdirectory+outfilename )