import cv2 as cv
import numpy as np
dimensions = (30, 30)
def compress(gif):
    # Shrink GIF
    resized_gif = cv.resize( gif, dimensions, interpolation=cv.INTER_AREA )
    # Stack columns one after the other
    stacked_gif = np.matrix( np.concatenate( tuple( resized_gif[:,idx] for idx in range(dimensions[1]) ) ) )
    return stacked_gif
def uncompress(gif):
    # Extract columns and stack horizontally
    columns = np.array([ gif[0,(i*dimensions[1]):(i*dimensions[1] + dimensions[0])] for i in range( dimensions[1] ) ])
    columns = columns.transpose()
    return columns