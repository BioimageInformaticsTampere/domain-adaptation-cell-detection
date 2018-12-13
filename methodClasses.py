# -*- coding: utf-8 -*-
"""
Created on Mon May  7 09:45:37 2018

@author: kaisapais

Method classes
"""

import utilityFunctions
import result_counting

import numpy as np
from sklearn.model_selection import train_test_split
import time
from scipy import io
from scipy.ndimage import filters,interpolation
from skimage import morphology

import random

from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Conv3D, MaxPooling3D, Reshape, AveragePooling3D, ZeroPadding2D, Lambda
from keras.optimizers import *
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import initializers

import warnings

warnings.filterwarnings("ignore")

    
class UNet_2D():
    
    def __init__(self,layer=[13]):
        if isinstance(layer,(list)):
            self.layer = layer
        else:
            self.layer= [layer]
        np.random.seed(1)
        random.seed(1)
        utilityFunctions.set_seed(1)

        self.blocksize = [128,128]
        self.inputdim = len(self.layer)
        self.netmode = '2d'
        self.optimizer = 'adam'
        self.loss = 'binary_crossentropy'
        self.pos_weight = 0.2
        #self.loss = self.weightedCrossEntropy()
        self.initializeModel()
        
        wn = 'unet2d'
        for i,l in enumerate(self.layer):
            wn = wn+'_'+str(l)
            
        self.weightname = wn
        self.weightstr = 'weights/'+self.weightname+'.h5'  
        self.traindays = list(range(1,7))

        self.subtract = False        
        self.randomsize = False
        
    def initializeModel(self):
         
        k_init = initializers.glorot_uniform(seed=1)

        inputs= Input((self.blocksize[0],self.blocksize[1],self.inputdim))
        
        conv11 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv11')(inputs)
        conv12 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv12')(conv11)
        pool1 = MaxPooling2D(pool_size=(2, 2),name='maxpool1')(conv12)
		
        conv21 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv21')(pool1)
        conv22 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv22')(conv21)
        pool2 = MaxPooling2D(pool_size=(2, 2),name='maxpool2')(conv22)
        		
        conv31 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv31')(pool2)
        conv32 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv32')(conv31)
        pool3 = MaxPooling2D(pool_size=(2, 2), name='maxpool3')(conv32)
        		
        conv41 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv41')(pool3)
        conv42 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv42')(conv41)
        drop4 = Dropout(0.25)(conv42)
        pool4 = MaxPooling2D(pool_size=(2, 2), name='maxpool4')(conv42)#drop4)
        
        conv51 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv51')(pool4)
        conv52 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv52')(conv51)
        drop5 = Dropout(0.5)(conv52)
        
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='upsampling6')(UpSampling2D(size = (2,2))(drop5))
        merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3,name='merge6')
        conv61 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv61')(merge6)
        conv62 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv62')(conv61)
        
        up7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='upsampling7')(UpSampling2D(size = (2,2))(conv62))
        merge7 = merge([conv32,up7], mode = 'concat', concat_axis = 3,name='merge7')
        conv71 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv71')(merge7)
        conv72 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv72')(conv71)
        
        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='upsampling8')(UpSampling2D(size = (2,2))(conv72))
        merge8 = merge([conv22,up8], mode = 'concat', concat_axis = 3,name='merge8')
        conv81 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv81')(merge8)
        conv82 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv82')(conv81)
        
        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='upsampling9')(UpSampling2D(size = (2,2))(conv82))
        merge9 = merge([conv12,up9], mode = 'concat', concat_axis = 3,name='merge9')
        conv91 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv91')(merge9)
        conv92 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv92')(conv91)
        		  
        conv10 = Conv2D(1, 3, padding='same', activation = 'sigmoid', name='conv10')(conv92)
        
        self.model = Model(input = inputs, output = conv10)
        
        self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = ['accuracy'])



    def loadTrainData(self):
        """Load train data"""
        days = self.traindays

        bly,blx = utilityFunctions.blockCoordinates(bls=self.blocksize)
        bll = len(bly)*len(blx)
        
        if self.randomsize:
            #double the size
            bll += bll

        x_train = np.zeros((bll*len(days)*2, self.blocksize[0], self.blocksize[1],self.inputdim))
        y_train = np.zeros((bll*len(days)*2, self.blocksize[0], self.blocksize[1],1))
        #print(x_train.shape)
        imnums = [1,2]
        runner = 0
        lastind = 0
        gtamount = 0
        for d,day in enumerate(days):
            for i,imnum in enumerate(imnums):

                layer = self.layer    
                    
                ims = utilityFunctions.readStack(layer, day=day, imnum=imnum, tofloat=True)
                if self.subtract:
                    ims = [ims[1]-ims[0]]
                else:
                    ims = [x-0.5 for x in ims]
                gt = utilityFunctions.loadGroundTruth(day=day,imnum=imnum)
                gtamount += len(gt[0])
                print(len(gt[0]),gtamount)

                if self.randomsize:
                    #augment training data by resizing
                    mask = utilityFunctions.trainingTarget(gt,radius=8)
                    bly,blx = utilityFunctions.blockCoordinates(bls=self.blocksize)
                    bll = len(bly)*len(blx)
                    xt,stats = utilityFunctions.makeBlocks(ims, self.blocksize)
                    yt,stats = utilityFunctions.makeBlocks([mask], self.blocksize)
                    x_train[lastind:lastind+bll,...] = xt
                    y_train[lastind:lastind+bll,...] = yt
                    lastind += bll
                    
                    z = 0.75 
                    #z=np.random.choice([0.7,0.8,0.9,1,1.1,1.2]) #if randomized
                    ims = [interpolation.zoom(x,z) for x in ims]
                    
                    [sy,sx] = ims[0].shape

                    gt=[[int(round(z*x)) for x in gt[0]],[int(round(z*x)) for x in gt[1]]]
                    
                    mask = utilityFunctions.trainingTarget(gt,sy=sy,sx=sx,radius=6)
                    bly,blx = utilityFunctions.blockCoordinates(sy=sy, sx=sx, bls=self.blocksize)
                    bll = len(bly)*len(blx)
                    xt,stats = utilityFunctions.makeBlocks(ims, self.blocksize)
                    yt,stats = utilityFunctions.makeBlocks([mask], self.blocksize)
                    x_train[lastind:lastind+bll,...] = xt
                    y_train[lastind:lastind+bll,...] = yt
                    lastind += bll


                else:
                    #basic training data
                    mask = utilityFunctions.trainingTarget(gt,radius=8)
                    xt,stats = utilityFunctions.makeBlocks(ims, self.blocksize)
        
                    x_train[runner*bll:runner*bll+bll,...] = xt
                    yt,stats = utilityFunctions.makeBlocks([mask], self.blocksize)
        
                    y_train[runner*bll:runner*bll+bll,...] = yt
                runner += 1
                
        print(x_train.shape)
        if self.randomsize:
            x_train = x_train[0:lastind,...]
            y_train = y_train[0:lastind,...]
            
        print(x_train.shape)
        return x_train,y_train

        

    def finalizeTrainData(self,x_train,y_train):
        """Apply transformations to data for better generalization. Divide to train and validation sets."""

        x_train = utilityFunctions.heteroData(x_train)
        
        x_train,rl = utilityFunctions.augmentData(x_train)
        y_train,rl = utilityFunctions.augmentData(y_train,rl)
        x_train,rl = utilityFunctions.augmentData(x_train)
        y_train,rl = utilityFunctions.augmentData(y_train,rl)        
        
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train)        
        
        return x_train, x_test, y_train, y_test
                
    def train(self,epochs=5,batchsize=5,loadweights=False):
        """Train function."""
        
        x_train, y_train = self.loadTrainData()

        x_train, x_val, y_train, y_val = self.finalizeTrainData(x_train, y_train)
        
        if loadweights:
            self.model.load_weights(self.weightstr,by_name=True)
            
        cblist = [ModelCheckpoint(self.weightstr,monitor='val_loss',save_weights_only=True,save_best_only=True)]
                
        print('Training started ',time.strftime('%d.%m. at %H:%M'))

        hist=self.model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batchsize, 
                        shuffle=True,
                        validation_data=(x_val, y_val),
                        verbose=1, #2 when piping, 1 in terminal
                        callbacks=cblist)
        
                
        print('Training ended ',time.strftime('%d.%m. at %H:%M'))
        utilityFunctions.checkDir('training')
        np.save('training/'+self.weightname+'_loss_nop.npy',hist.history['loss'])
        np.save('training/'+self.weightname+'_val_loss_nop.npy',hist.history['val_loss'])
        #print(hist.history['loss'])
        return hist.history['val_loss'],hist.history['loss']        

       
        
    def switchOptimizer(self,optimizer,lr=0.1,mom=0.8,nest=True):
        
        if optimizer=='sgd':
            self.optimizer = SGD(lr=lr, momentum=mom, nesterov=nest)
            
        else:
            self.optimizer = optimizer
            
    def getResults(self,celltype='BT-474',day=1,imnum=2,visualize=False,th=0.5,layers=None,lw=True,da=False):
        if layers:
            pass
        else:
            layers=self.layer
            
        ims = utilityFunctions.readStack(layers=layers, cell=celltype, day=day, imnum=imnum, imtype='test', tofloat=True)
        if self.subtract:
            ims = [ims[1]-ims[0]]
        else:
            
            ims = [x-0.5 for x in ims]
            
        if self.blocksize[0] < 130 or self.blocksize[1] < 130:
            bltmp = self.blocksize
            self.blocksize=[512,512]
            self.initializeModel()
            
        else:
            bltmp = False

        blocks, stats = utilityFunctions.makeBlocks(ims, self.blocksize, netmode=self.netmode,overlap=64)
        
        if lw:
            self.model.load_weights(self.weightstr)

        bls = list(blocks.shape)
        bls[0] = 1
        if self.netmode == '2d':
            fornet = np.zeros((bls[0],self.blocksize[0],self.blocksize[1],self.inputdim))
        else:
            fornet = np.zeros((bls[0],self.blocksize[0],self.blocksize[1],len(self.layer),1))
        
        predbl = np.zeros((blocks.shape[0], self.blocksize[0], self.blocksize[1],1))
        for i in range(blocks.shape[0]):
            fornet[0,...] = blocks[i,...]
            ptmp = self.model.predict(fornet)
            if self.model.output_shape[1] < self.blocksize[0]:
                ptmp_ = np.zeros((1,self.blocksize[0], self.blocksize[1],1))
                ptmp_[0,...,0] = interpolation.zoom(ptmp[0,...,0],4)
                ptmp = ptmp_
            predbl[i,...] = ptmp
#        pred = self.model.predict(blocks)
        predim = utilityFunctions.buildFromBlocks(predbl,stats)
        
        if bltmp:
            self.blocksize = bltmp
            self.initializeModel()
#            self.model.load_weights(self.weightstr)
        
        if visualize:
            # Visualize results. If ground truth exists, count scores.
            
            focused = utilityFunctions.readStack(layers=[13], cell=celltype, day=day, imnum=imnum, imtype='test', tofloat=True)
            gt = utilityFunctions.loadTestGroundTruth(day=day,imnum=imnum, celltype=celltype)
                
            pointim = utilityFunctions.getPointIm(predim[0]>th)
            #pointim = utilityFunctions.findPeaks_plm(predim[0])
            cellrgb = utilityFunctions.drawCellIm(pointim,focused[0])
            if len(gt)==0:
                ims.extend(predim)
                ims.append(cellrgb)
                titlestr = ''.join([celltype,', day ',str(day),', number ',str(imnum),'. Layer ', str(self.layer[0]), '.'])
                utilityFunctions.visualize(ims,figtitle=titlestr)
            else:
                
                tp,fp,fn,tpgt = result_counting.getCoordinates2(pointim,gt)
                scoresim = result_counting.scoresImage(focused,tp,fp,fn)
                f1,prec,rec = result_counting.countScores(len(tp),len(fp),len(fn))
                f1 = round(1000*f1)/1000
                prec = round(1000*prec)/1000
                rec = round(1000*rec)/1000
                ims.extend(predim)
                ims.append(cellrgb)
                ims.append(scoresim)
                titlestr = ''.join([celltype,', day ',str(day),', number ',str(imnum),'. Layer ', str(self.layer[0]), '. F1: ', str(f1), ', precision: ', str(prec), ', recall: ', str(rec)])
                scorestr = ''.join(['TP (red,',str(len(tp)),'), FP (green,',str(len(fp)), ') FN (blue,', str(len(fn)),')'])
                subtitlestr = ['Input','Prediction','Found cells',scorestr]
                utilityFunctions.visualize(ims, figtitle=titlestr, subfigtitle=subtitlestr)
                return predim[0],scoresim

#        if da:
#            #pointim = utilityFunctions.findPeaks_plm(predim[0])#which one...
#            pointim = utilityFunctions.getPointIm(predim[0]>0.5)
#            return predim[0]#,pointim
        return predim[0]
    
    def adaptDomain(self,celltype='22Rv1',day=3,imnums=list(range(5,8)),epochs=10,batchsize=5,radius=4,th=0.5):
        """Adapt to new domain (cell line)"""
        self.blocksize = [512,512]
        self.initializeModel()
        self.model.load_weights(self.weightstr)
        pis = []        
        

        for i,imnum in enumerate(imnums):
            p = self.getResults(celltype=celltype,day=day,imnum=imnum,visualize=False,da=True,lw=False)
            pi = utilityFunctions.findPeaks_plm(p,th)
            points = np.where(pi)
            points = [list(points[0]),list(points[1])]
            pi = utilityFunctions.trainingTarget(points,radius=radius)
#            pi = morphology.binary_dilation(pi,morphology.disk(radius))
            pis.append(pi)

        self.blocksize = [128,128]
        self.initializeModel()
        self.model.load_weights(self.weightstr)


        bly,blx = utilityFunctions.blockCoordinates(bls=self.blocksize)
        bll = len(bly)*len(blx)
        x_train = np.zeros((len(imnums)*bll,self.blocksize[0],self.blocksize[1],self.inputdim))
        y_train = np.zeros((len(imnums)*bll,self.blocksize[0],self.blocksize[1],1))
        lastind = 0

        for i,imnum in enumerate(imnums):
        
            ims = utilityFunctions.readStack(layers=self.layer,cell=celltype,day=day,imnum=imnum,imtype='test',tofloat=True)
            ims = [x-0.5 for x in ims]
            xt,stats = utilityFunctions.makeBlocks(ims, self.blocksize)
            yt,stats = utilityFunctions.makeBlocks([pis[i]], self.blocksize)
            
            x_train[lastind:lastind+bll,...] = xt
            y_train[lastind:lastind+bll,...] = yt
            lastind += bll


        xt,yt = self.loadTrainData()
        
        sourcedata = np.random.choice(list(range(yt.shape[0])),replace=False,size=y_train.shape[0])
        
        xt = xt[sourcedata,...]
        yt = yt[sourcedata,...]

        x_train = np.concatenate((x_train,xt),axis=0)
        y_train = np.concatenate((y_train,yt),axis=0)
        x_train, x_val, y_train, y_val = self.finalizeTrainData(x_train, y_train)
        
# freeze some layers?            
#        for i in range(len(self.model.layers)): 
#            lname = self.model.layers[i].name
#            if lname[0:4] == 'conv' and int(lname[4]) < 4:
#                self.model.layers[i].trainable = False
        utilityFunctions.checkDir('weights')
        utilityFunctions.checkDir('intermed_weights')
           
        self.weightstr = 'weights/'+self.weightname+'_da_'+celltype+'.h5'
        fp2 = 'intermed_weights/'+self.weightname+'_{epoch:02d}-{val_loss:.2f}.h5'
        
        cblist = [ModelCheckpoint(self.weightstr,monitor='val_loss',save_weights_only=True,save_best_only=True),ModelCheckpoint(fp2,save_weights_only=True,save_best_only=False)]
                
        print('Training started ',time.strftime('%d.%m. at %H:%M'))
        

        
        hist=self.model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batchsize, 
                        shuffle=True,
                        validation_data=(x_val, y_val),
                        verbose=1, #2 when piping, 1 in terminal
                        callbacks=cblist)
        
        print('Training ended ',time.strftime('%d.%m. at %H:%M'))
        utilityFunctions.checkDir('training')
        np.save('training/'+self.weightname+'_loss_nop.npy',hist.history['loss'])
        np.save('training/'+self.weightname+'_val_loss_nop.npy',hist.history['val_loss'])    
    

class UNet_small(UNet_2D):
    
    def __init__(self,layer=[13]):
        if isinstance(layer,(list)):
            self.layer = layer
        else:
            self.layer= [layer]
        np.random.seed(1)
        random.seed(1)
        utilityFunctions.set_seed(1)

        self.blocksize = [128,128]
        self.inputdim = len(self.layer)
        self.netmode = '2d'
        self.optimizer = 'adam'
        self.loss = 'binary_crossentropy'
        self.pos_weight = 0.2
        self.initializeModel()
        wn = 'unet_sm'
        for i,l in enumerate(self.layer):
            wn = wn+'_'+str(l)
            
        self.weightname = wn


        self.weightstr = 'weights/'+self.weightname+'.h5'  
        self.traindays = list(range(1,7))
        self.subtract = False        
        
        self.randomsize = False
        
    def initializeModel(self):
         
        k_init = initializers.glorot_uniform(seed=1)
        
        inputs= Input((self.blocksize[0],self.blocksize[1],self.inputdim))
        
        conv11 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv11')(inputs)
        conv12 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv12')(conv11)
        pool1 = MaxPooling2D(pool_size=(2, 2),name='maxpool1')(conv12)
		
        conv21 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv21')(pool1)
        conv22 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv22')(conv21)
        pool2 = MaxPooling2D(pool_size=(2, 2),name='maxpool2')(conv22)
        		
        conv31 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv31')(pool2)
        conv32 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv32')(conv31)
        pool3 = MaxPooling2D(pool_size=(2, 2), name='maxpool3')(conv32)
        		
        conv41 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv41')(pool3)
        conv42 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv42')(conv41)
        drop4 = Dropout(0.25)(conv42)
                
        up7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='upsampling7')(UpSampling2D(size = (2,2))(drop4))
        merge7 = merge([conv32,up7], mode = 'concat', concat_axis = 3,name='merge7')
        conv71 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv71')(merge7)
        conv72 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv72')(conv71)
        
        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='upsampling8')(UpSampling2D(size = (2,2))(conv72))
        merge8 = merge([conv22,up8], mode = 'concat', concat_axis = 3,name='merge8')
        conv81 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv81')(merge8)
        conv82 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv82')(conv81)
        
        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='upsampling9')(UpSampling2D(size = (2,2))(conv82))
        merge9 = merge([conv12,up9], mode = 'concat', concat_axis = 3,name='merge9')
        conv91 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv91')(merge9)
        conv92 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv92')(conv91)
        		  
        conv10 = Conv2D(1, 3, padding='same', activation = 'sigmoid', name='conv10')(conv92)
        
        self.model = Model(input = inputs, output = conv10)
        
        self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = ['accuracy'])





                
            
class UNet_smaller(UNet_2D):
    
    def __init__(self,layer=[13]):
        if isinstance(layer,(list)):
            self.layer = layer
        else:
            self.layer= [layer]
        np.random.seed(1)
        random.seed(1)
        utilityFunctions.set_seed(1)

        self.blocksize = [128,128]
        self.inputdim = len(self.layer)
        self.netmode = '2d'
        self.optimizer = 'adam'
        self.loss = 'binary_crossentropy'
        self.pos_weight = 0.2
        #self.loss = self.weightedCrossEntropy()
        self.initializeModel()
        wn = 'unet_smer'
        for i,l in enumerate(self.layer):
            wn = wn+'_'+str(l)
            
        self.weightname = wn


        self.weightstr = 'weights/'+self.weightname+'.h5'  
        self.traindays = list(range(1,7))
        self.subtract = False        
        
        self.randomsize = False
        
    def initializeModel(self):
         
        k_init = initializers.glorot_uniform()#(seed=1)

        inputs= Input((self.blocksize[0],self.blocksize[1],self.inputdim))
        
        conv11 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv11')(inputs)
        conv12 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv12')(conv11)
        pool1 = MaxPooling2D(pool_size=(2, 2),name='maxpool1')(conv12)
		
        conv21 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv21')(pool1)
        conv22 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv22')(conv21)
        pool2 = MaxPooling2D(pool_size=(2, 2),name='maxpool2')(conv22)
        		
        conv31 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv31')(pool2)
        conv32 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv32')(conv31)
        
        up4 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='upsampling4')(UpSampling2D(size = (2,2))(conv32))
        merge4 = merge([conv22,up4], mode = 'concat', concat_axis = 3,name='merge4')
        conv41 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv41')(merge4)
        conv42 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv42')(conv41)
        
        up5 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='upsampling5')(UpSampling2D(size = (2,2))(conv42))
        merge5 = merge([conv12,up5], mode = 'concat', concat_axis = 3,name='merge5')
        conv51 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv51')(merge5)
        conv52 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv52')(conv51)
        		  
        conv6 = Conv2D(1, 3, padding='same', activation = 'sigmoid', name='conv6')(conv52)
        
        self.model = Model(input = inputs, output = conv6)
        
        self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = ['accuracy'])

            
        
    
class UNet_smallest(UNet_2D):
    
    def __init__(self,layer=[13]):
        if isinstance(layer,(list)):
            self.layer = layer
        else:
            self.layer= [layer]
        np.random.seed(1)
        random.seed(1)
        utilityFunctions.set_seed(1)

        self.blocksize = [128,128]
        self.inputdim = len(self.layer)
        self.netmode = '2d'
        self.optimizer = 'adam'
        self.loss = 'binary_crossentropy'
        self.pos_weight = 0.2
        self.initializeModel()
        wn = 'unet_smest'
        for i,l in enumerate(self.layer):
            wn = wn+'_'+str(l)
            
        self.weightname = wn


        self.weightstr = 'weights/'+self.weightname+'.h5'  
        self.traindays = list(range(1,7))
        self.subtract = False        
        
        self.randomsize = False
        
    def initializeModel(self):
         
        k_init = initializers.glorot_uniform()#(seed=1)

        inputs= Input((self.blocksize[0],self.blocksize[1],self.inputdim))
        
        conv11 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv11')(inputs)
        conv12 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv12')(conv11)
        pool1 = MaxPooling2D(pool_size=(2, 2),name='maxpool1')(conv12)
		
        conv21 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv21')(pool1)
        conv22 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv22')(conv21)
        
        up5 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='upsampling5')(UpSampling2D(size = (2,2))(conv22))
        merge5 = merge([conv12,up5], mode = 'concat', concat_axis = 3,name='merge5')
        conv51 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv51')(merge5)
        conv52 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init, name='conv52')(conv51)
        		  
        conv6 = Conv2D(1, 3, padding='same', activation = 'sigmoid', name='conv6')(conv52)
        
        self.model = Model(input = inputs, output = conv6)
        
        self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = ['accuracy'])

        

        
