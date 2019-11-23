#!/usr/bin/env python3
# Transform (1x900) to (30,30) GIF
import os
import sys
import imageio
import numpy as np
from compression import uncompress
infilename = sys.argv[1]
outdirectory = './'
_, outfilename = os.path.split( infilename )
# Extract first frame from GIF (only one frame)
gif = imageio.mimread(infilename)[0]
# Extract columns and stack horizontally
columns = uncompress(gif)
# Save output to file
imageio.imwrite( outdirectory + outfilename, columns )
print( infilename, '-->', outdirectory+outfilename )