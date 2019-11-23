#!/usr/bin/env python3
# Compress and store as (1x900) GIF
import os
import sys
import imageio
import cv2 as cv
import numpy as np
from compression import compress
infilename = sys.argv[1]
outdirectory = './dataset/compressed/'
_, outfilename = os.path.split( infilename )
# Extract first frame from GIF (only one frame)
gif = imageio.mimread(infilename)[0]
stacked_gif = compress( gif )
# Save output to file
imageio.mimwrite( outdirectory + outfilename, stacked_gif )
print( infilename, '-->', outdirectory+outfilename )