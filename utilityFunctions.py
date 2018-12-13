# -*- coding: utf-8 -*-
"""
Created on Mon May  7 09:45:22 2018

@author: kaisapais

Utility functions
"""

import json
import os
from skimage.io import imread
from skimage import morphology
from skimage import measure
import numpy as np
from math import ceil,floor,sqrt
from scipy.ndimage import filters
import math
from skimage.feature import peak_local_max
from scipy.signal import convolve2d


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
import tensorflow as tf
import gc


def set_seed(seed=1):
    np.random.seed(seed)


def loadPaths():
    """Loads data paths from config file"""
    
    with open('config.json','r') as f:
        config = json.load(f)
        
    return config['trainpath'],config['testpath']


def loadShape():
    """Loads static image shape from config file
    (no need to read image for shape)"""

    with open('config.json','r') as f:
        config = json.load(f)
        
    return [int(config['imshapey']),int(config['imshapex'])]

def checkDir(dirname):
    """Check if directory exists, create if does not"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def readStack(layers=list(range(1,26)), cell='PC-3', day=3, imnum=2,imtype='train',normalim=False,tofloat=True):
    """Read image stack. Return list of images.
    Defaults output whole stack of training images"""
    
    if imtype == 'train':
        #Train path
        path,p = loadPaths()
        imformat = '.png'
        
    elif imtype == 'test':
        #Test path
        p,path = loadPaths()
        imformat = '.bmp'
        
    else:
        print('Invalid image type:',imtype,'. Valid types are "train" and "test".')
        return 0
    
    imstack = []
    
    for l,layer in enumerate(layers):
        #Read each layer and append
        im = imread(''.join([path, '/', cell, '/day', str(day), '/image_', str(imnum), '_', str(layer), imformat]))
        if normalim:
            im = normalize(im.astype('float32'),0,1)
        elif tofloat:
            im = im.astype('float32')
            im = im/255
        imstack.append(im)
        
    return imstack
    
def visualize(ims,yshape=None,xshape=None,figtitle=' ',subfigtitle=None):
    """visualize list of images, or single image.
    yshape and xshape: amount of images in y- and x-axis.
    If not defined, best shapes are calculated.
    Subfigure title list has to have the same length as image list"""
    
    if isinstance(ims,(list,tuple)):
        pass
    else:#single image
        plt.figure()
        plt.suptitle(figtitle)
        plt.imshow(ims)
        plt.axis('off')
        return
    
    imamount = len(ims)
    
    if imamount == 1:
        plt.figure();
        plt.suptitle(figtitle)
        plt.imshow(ims[0])
        plt.axis('off')
        return
    
    if not yshape:
        if not xshape:
            #no shape given
            yshape = int(floor(sqrt(imamount)))
            xshape = int(ceil(imamount/yshape))
        else:
            #xshape given
            yshape = int(ceil(len(ims)/xshape))
    elif not xshape:
        #yshape given
        xshape = int(ceil(len(ims)/yshape))
        
    if xshape*yshape < len(ims):
        xshape = int(ceil(len(ims)/yshape))
    
    fig = plt.figure()
    fig.suptitle(figtitle)
    gs1 = gridspec.GridSpec(yshape,xshape)
    gs1.update(wspace=0.025, hspace=0.05)
    
    ax1 = plt.subplot(gs1[0])
    ax1.imshow(ims[0])
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    ax1.set_adjustable('box-forced')
    ax1.axis('off')
    
    if subfigtitle and len(subfigtitle)==len(ims):
        ax1.set_title(subfigtitle[0])
        
    for i in range(1,len(ims)):
        ax2 = plt.subplot(gs1[i], sharex=ax1,sharey=ax1)
        ax2.imshow(ims[i])
        ax2.set_adjustable('box-forced')
        ax2.axis('off')
        if subfigtitle and len(subfigtitle)==len(ims):
            ax2.set_title(subfigtitle[i])


def blockCoordinates(sy=1196,sx=1596,bls=[128,128],ol=0):
    """ Calculate block coordinates for making blocks.
    bls defines size of blocks, ol is the overlap."""
    
    if ol*2 >= bls[0] or ol*2 >= bls[1]:
        print('Overlap cannot be bigger than blocksize')
        return 0,0
    bly = list(range(0,sy-(bls[0]),(bls[0]-ol)))
    blx = list(range(0,sx-(bls[0]),bls[1]-ol))
    if bly[-1] < (sy-bls[0]):
        bly.append(sy-(bls[0]))
    if blx[-1] < (sx-bls[1]):
        blx.append(sx-(bls[1]))
    
    return bly,blx
    
def makeBlocks(ims,blocksize=[128,128],overlap=0,netmode='2d'):
    """ Create correct sized blocks from list of images.
    netmode defines output dimensions: 2d for 4-dimensionial, 3d for 5-dimensional (network input dimensions)"""
    
    """TODO: implement resizing"""
    
    sy,sx = ims[0].shape    
    
    if overlap/2 != round(overlap/2):
        print('Overlap must be even number!')
        return 0
    
    bly,blx = blockCoordinates(sy,sx,blocksize,overlap)
    
    if netmode == '3d':
        blocks = np.zeros((len(bly)*len(blx), blocksize[0], blocksize[1], len(ims), 1))
    elif netmode == '2d':
        blocks = np.zeros((len(bly)*len(blx), blocksize[0], blocksize[1], len(ims)))
    else:
        print('Unrecognized mode for network. Creating blocks for 2d network.')
        blocks = np.zeros((len(bly)*len(blx), len(ims), blocksize[0], blocksize[1]))
        
    runner = 0
    for a,y in enumerate(bly):
        for b,x in enumerate(blx):
            for l in range(len(ims)):
                if netmode == '3d':
                    blocks[runner,:,:,l,0] = ims[l][y:y+blocksize[0], x:x+blocksize[1]]
                elif netmode == '2d':
                    blocks[runner,:,:,l] = ims[l][y:y+blocksize[0], x:x+blocksize[1]]
            runner += 1
    
    #Create dictionary for buildFromBlocks function            
    stats = {'ycoords':bly, 'xcoords':blx, 'blocksize':blocksize, 'overlap':overlap, 'ysize':sy, 'xsize':sx, 'resize':1}
    
    return blocks,stats
    
    
def buildFromBlocks(blocks,stats,origsize=None):
    """ Build image from blocks with given statistics dictionary.
    Give origsize if output has different size than size in config file."""
    
    bls = stats['blocksize']
    overlap = stats['overlap']
    resize = stats['resize']
    ycoords = stats['ycoords']
    xcoords = stats['xcoords']
    
    if origsize:
        ys = origsize[0]
        xs = origsize[1]
    else:
        [ys,xs] = loadShape()

    
    if stats['resize'] != 1:
        ycoords = [round(resize*x) for x in ycoords]
        xcoords = [round(resize*x) for x in xcoords]
        bls = [round(resize*x) for x in bls]
        overlap = round(resize*overlap)

            
    ydiff = ycoords[1]-ycoords[0]+int(overlap)
    bldiff = blocks.shape[-3]

    if ydiff != bldiff:
        olim = int((ydiff-bldiff)/2)
        olb = 0
    else:
        olim = int(overlap/2)
        olb = olim
#    print(resize,olim,olb)
    
    if olb == 0:
        olb2 = None
    else:
        olb2 = -olb
    ims = []
     
    for i in range(blocks.shape[-1]):
        im = np.zeros((int(resize*stats['ysize']), int(resize*stats['xsize'])))
        runner = 0
        for a,y in enumerate(ycoords):
            for b,x in enumerate(xcoords):
                if y == 0 and x==0:
                    im[y:y+(bls[0]-olim),x:x+(bls[1]-olim)] = blocks[runner,...,0:olb2,0:olb2,i]
                elif y+bls[0]==ys and x==0:
                    im[y+olim:y+(bls[0]),x:x+(bls[1]-olim)] = blocks[runner,...,olb:,0:olb2,i]
                elif x+bls[1]==xs and y==0:
                    im[y:y+(bls[0]-olim),x+olim:x+(bls[1])] = blocks[runner,...,0:olb2,olb:,i]
                elif y == 0:                    
                    im[y:y+(bls[0]-olim),x+olim:x+(bls[1]-olim)] = blocks[runner,...,0:olb2,olb:olb2,i]
                elif x == 0:                    
                    im[y+olim:y+(bls[0]-olim),x:x+(bls[1]-olim)] = blocks[runner,...,olb:olb2,0:olb2,i]
                elif y+bls[0]==ys and x+bls[1]==xs:
                    im[y+olim:y+(bls[0]),x+olim:x+(bls[1])] = blocks[runner,...,olb:,olb:,i]
                elif y+bls[0]==ys:
                    im[y+olim:y+(bls[0]),x+olim:x+(bls[1]-olim)] = blocks[runner,...,olb:,olb:olb2,i]
                elif x+bls[1]==xs:
                    im[y+olim:y+(bls[0]-olim),x+olim:x+(bls[1])] = blocks[runner,...,olb:olb2,olb:,i]
                else:
                    im[y+olim:y+(bls[0]-olim),x+olim:x+(bls[1]-olim)] = blocks[runner,...,olb:olb2,olb:olb2,i]
                runner += 1
        ims.append(im)
    
    return ims


def loadGroundTruth(day=3,imnum=2):
    """ Read ground truth file for training data.
    Ground truth file has corner points for each cell."""
    trp,tep = loadPaths()
    
    gtfile = ''.join([trp,'/PC-3/day',str(day),'/image_',str(imnum),'.txt'])
    
        
    firstline = True

    indsy = []
    indsx = []
    with open(gtfile,'r') as f:
        for line in f:
            if firstline:
                # skip header
                firstline = False
            else:
                thisstr = line
                strlist = thisstr.split(' ')
                thisclass = strlist[4][0:-1]
                if myDict(thisclass): #if cell...
                    ints = []
                    for j in range(4): #...get corner coordinates
                        ints.append(int(strlist[j]))
                    indsy.append(ints[1] + round((ints[3]-ints[1])/2))
                    indsx.append(ints[0] + round((ints[2]-ints[0])/2))
        
    
    return [indsy,indsx]

def loadTestGroundTruth(day=3,imnum=1,celltype='PC-3'):
    """ Read ground truth file for test data.
    Cell center points are given in ground truth file."""
    trp,tep = loadPaths()
    gtpath = tep[0:-4]
    gtfile = ''.join([gtpath,'ground_truth/test/',celltype,'/day', str(day), '/image_', str(imnum), '.txt'])
    #print(gtfile)

    inds = []
    try:
        with open(gtfile,'r') as f:
            for line in f:
                thisstr = line
                strlist = thisstr.split(',')
                ints = []
                for j in range(2): 
                    ints.append(int(strlist[j]))
                inds.append([ints[1],ints[0]])
    except FileNotFoundError:
        print('Ground truth file not found. Returning empty list.')
        return inds
    
    return inds

        
def trainingTarget(gt,sy=None,sx=None,radius=8,inverse=False,gaussian=False):
    """ Create target mask for training """
    if not sy:
        sy,sx = loadShape()
    
    mask = np.zeros((sy,sx))
    mask[gt] = 1
    if gaussian:
        mask = filters.gaussian_filter(mask,5)
        mask = normalize(mask)
    else:
        mask = differCells(mask,radius)
    
    if inverse:
        mask = mask - 1
        mask[np.where(mask<0)] = 1
    
    return mask

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
    
    
    
        
def augmentData(blocks,randlist=0):
    """ for each sample, select randomly augmentation method from:
    rotation (2,3), flip updown (0), flip rightleft (1). 
    Input randlist for creating same transformations as 
    previously (e.g. to match x_train and y_train)"""

    howMany = blocks.shape[0]
    newBlocks = np.zeros(blocks.shape)
    
    if randlist is 0:
        randlist = []
        for i in range(howMany):
            randlist.append(np.random.randint(0,5))
    

    if len(blocks.shape)==5:
        for i in range(howMany):
            block = blocks[i,:,:,:,0]
            
            
            method = randlist[i]
            
            if method == 0:
                newblock = np.flip(block,axis=0)
                
            elif method == 1:
                newblock = np.flip(block,axis=1)
                
            elif method == 2:
                newblock = np.rot90(block,k=1,axes=(0,1))
                
            elif method == 3:
                newblock = np.rot90(block,k=3,axes=(0,1))
                
            else:
                newblock=block
                
            newBlocks[i,:,:,:,0] = newblock
    elif len(blocks.shape)==4:
        for i in range(howMany):
            block = blocks[i,:,:,:]
        
            method = randlist[i]
            
            if method == 0:
                newblock = np.flip(block,axis=0)
                
            elif method == 1:
                newblock = np.flip(block,axis=1)
                
            elif method == 2:
                newblock = np.rot90(block,k=1,axes=(0,1))
                
            elif method == 3:
                newblock = np.rot90(block,k=3,axes=(0,1))
            else:
                newblock=block
                
            newBlocks[i,:,:,:] = newblock

    return newBlocks, randlist

def heteroData(blocks):
    """ For each sample, select randomly method from:
     do nothing (0), subtract (1), multiply (2), add noise (3) """


    if len(blocks.shape)>2:
        howMany = blocks.shape[0]
    else:
        howMany = 1
    newBlocks = np.zeros(blocks.shape)
    bl_len = len(blocks.shape)
    for i in list(range(howMany)):
        if bl_len == 5:
            block = blocks[i,0,:,:,:]
            layers = block.shape[-1]
        elif bl_len == 4:
            block = blocks[i,:,:,:]
            layers = block.shape[-1]
        elif bl_len == 3:
            block = blocks[i,:,:]
            layers = 1
        else: #one image
            block = blocks
            layers = 1
            
        for l in range(layers):
            if bl_len > 3:
                bl = block[...,l]
            else:
                bl = block
                
            method = np.random.randint(0,5)
            
            if method == 0:
                add = np.random.randint(1,10)
                add = add/100
                newblock = bl + add
                
            if method == 1:
                subtract = np.random.randint(1,10)
                subtract = subtract/100
                newblock = bl - subtract
                
            elif method == 2:
                noise = np.random.normal(0,0.01,bl.shape)
                newblock = bl + noise            
                
            elif method == 3:
                noise = np.random.normal(0,0.03,bl.shape)
                newblock = bl + noise
                
            else:
                newblock = bl
        
        
            if bl_len == 5:    
                newBlocks[i,0,:,:,l] = newblock
            elif bl_len == 4:
                newBlocks[i,:,:,l] = newblock
            elif bl_len == 3:
                newBlocks[i,:,:] = newblock
            else:
                newBlocks = newblock
    return newBlocks    


def getPointIm(im):
    """Create single points from each separated area in binary image input."""
    
    pointim = np.zeros(im.shape)
    cell_labels = measure.label(im, background=0)
    
    for l in range(1,np.max(cell_labels)):
        [y,x] = np.where(cell_labels==l)
        cy = np.min(y) + np.round((np.max(y) - np.min(y))/2)
        cx = np.min(x) + np.round((np.max(x) - np.min(x))/2)
        pointim[int(cy),int(cx)] = 1

    return pointim


def findPeaks_plm(predim,th=0.2,otsu=False,mindist=5):
    """Get cell centers from confidence map via detection of local peaks"""
    
    if otsu:
        #overrides given threshold
        th = threshold_otsu(predim)
    
    peaks = peak_local_max(predim,min_distance=mindist,threshold_abs=th,indices=False).astype(int)
    
    labeled = measure.label(peaks>0.5,background=0)
    res = np.zeros(predim.shape)
    
    if len(np.unique(labeled)) == 1:
        return res
    
    for i in range(1,len(np.unique(labeled))):
        y,x = np.where(labeled==i)
        if len(y) > 1:
            y=int(round(np.mean(y)))
            x=int(round(np.mean(x)))
        res[y,x] = 1
    
    return res


def drawCellIm(pointim,im,radius=4):
    """Overlay points from pointim to image."""
    
    im = normalize(im)
    imrgb = np.dstack((im,im,im))
    bs = morphology.disk(radius)
    dilim = morphology.binary_dilation(pointim,bs).astype('float32')
    #print(imrgb.shape,np.max(dilim),np.max(imrgb),pointim.dtype,np.max(pointim))
    imrgb[np.where(dilim==1)] = [1,0,0]
    
    return imrgb


def gtImage(gt,mult=[1,1]):
    [sy,sx] = loadShape()
    im = np.zeros((sy*mult[0],sx*mult[1]))
    for i,g in enumerate(gt):
        im[g[0],g[1]] = 1
    return im


def densityMap(gt,mult=[1,1],radius=50):
    """Draw density map"""
    
    im = gtImage(gt,mult)
    imc = filters.gaussian_filter(im,radius)
    imc = imc*(100/np.max(imc))
    
    return imc


def normalize(im,imin=0,imax=1):
    im = im.astype('float32')
    if np.max(im)-np.min(im) == 0:
        return im
    im = imin + ((im-np.min(im))*(imax-imin))/(np.max(im)-np.min(im))
    return im    

        
def myDict(c):
    return {
        'cell' : 1,
        'background' : 0,
    }[c]

    
def killAll(bn):
    """Delete the model to free memory.
    Prevents memory problems resulting from tensorflow and keras bugs.
    Does not work with Theano backend.
    Input: network class."""
    
    del bn.model
    K.clear_session()
    gc.collect()
    tf.reset_default_graph()
        
    
    
