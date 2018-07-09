# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 08:01:34 2018

@author: kaisapais
"""

import numpy as np
import math
from skimage import morphology


def differCells(centerimage,radius=8,mindist=1):
    """ Create target mask where each cell is presented with a circle.
    If circles touch each other, radius is reduced.
    Inputs: 
        centerimage: numpy array of zeros and ones with single point marking one cell
        radius: maximum radius of a circle
        mindist: minimum distance (in pixels) between circles
    Outputs:
        ci: target mask. """
    ci = np.copy(centerimage)
    y,x = np.where(ci)
    yr,xr = np.where(morphology.disk(radius,int))
    yr = yr-radius
    xr = xr-radius
    for i in range(len(y)):
        # Calculate distance to other cell points
        diffy = (y[i]-y)**2
        diffx = (x[i]-x)**2
        eucdiff = np.sqrt(diffy+diffx)
        eucdiff = np.delete(eucdiff,i)

        if (min(eucdiff) <= radius*2+mindist):
            # Too close, calculate new radius
            newrad = int(math.floor(min(eucdiff)/2))-mindist
            if newrad < 1:
                # single pixel only
                indy = np.array(y[i])
                indx = np.array(x[i])

            else:
                yr_,xr_ = np.where(morphology.disk(newrad,int))
                
                yr_ = yr_ - newrad
                xr_ = xr_ - newrad

                indy = y[i]+yr_
                indx = x[i]+xr_
            
            
        else:
            # Acceptable distance
            indy = y[i]+yr
            indx = x[i]+xr
            
        # Catch coordinates exceeding borders
        indy[indy>=ci.shape[0]] = y[i]
        indx[indx>=ci.shape[1]] = x[i]
        indy[indy<0] = y[i]
        indx[indx<0] = x[i]
            
        ci[indy,indx] = 1
    
    return ci
    