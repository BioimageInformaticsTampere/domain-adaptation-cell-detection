# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 08:55:09 2018

@author: kaisapais

This is the script for whole domain adaptation pipeline (initial training and domain adaptation for all cell lines).

"""
from __future__ import print_function

import result_counting
import methodClasses
import utilityFunctions

import numpy as np
from skimage import measure

from scipy import io
from scipy.ndimage import filters

from skimage import morphology
from skimage.feature import peak_local_max

import matplotlib.pyplot as plt


#define what to do
training = False
adapting = False
testing = True



cells = ['PC-3', 'LNCaP', 'BT-474', '22Rv1']
days = [3, 3, 1, 3] #annotated days
imnums = [1,2,3,4]


#length defines the amount of epochs set
blss = [128,192,112,160,96,176] #[128,176,160,192,96,112,144,128]

epoch = 10

np.random.seed(1)



def train(bn,lw=False):
    """Train with PC-3"""
    traindays = list(range(1,7))
    epoch = 10
    epochs = 0
    lr = 0.1
    for b,bls in enumerate(blss):
        print(bls,bn.weightname)
        
        
        bn.randomsize = True
        bn.traindays = traindays
        bn.blocksize = [bls,bls]
        bn.switchOptimizer('sgd',lr)
        
        bn.initializeModel()

        h=bn.train(epochs=epoch,loadweights=lw)
        print(h)
        epochs += epoch
        if epochs < 65:#65
            lr = lr*0.5
        print('Trained %s for %i epochs'%(bn.weightname,epochs))
        wstr = 'weights/%s_%i.h5'%(bn.weightname,epochs)
        bn.model.load_weights(bn.weightstr)
        bn.model.save_weights(wstr)
        utilityFunctions.killAll(bn)
        
        lw = True
        traindays = list(range(1,7))
        row = ['Trained for %i epochs'%epochs]
        result_counting.writeRowToFile(row,bn.weightname+'.csv')
        result_counting.writeRowToFile(['no postprocessing, th 0.5'],bn.weightname+'.csv')

        test(bn,postproc=False,epoch=epochs)
        result_counting.writeRowToFile(['postprocessing + cell specific th'],bn.weightname+'.csv')
        test(bn,postproc=True)
        
        utilityFunctions.killAll(bn)

    print('Model %s trained for %i epochs'%(bn.weightname,epochs))


def adapt(bn,cell,days):
    """Adapt to new domain"""

    epoch = 1#0
    adaptimnums = list(range(5,16))#exclude annotated
    if cell == '22Rv1':
        radius = 6
        th = 0.2
    elif cell == 'BT-474':
        radius = 6
        th = 0.2
    elif cell == 'LNCaP':
        radius = 6
        th = 0.5
    else:
        
        radius = 6
        th = 0.5


    epochs = 0
    lr = 0.1
    for b,bls in enumerate(blss):
        day = np.random.choice(days)
        imnum = list(np.random.choice(adaptimnums,size=4,replace=False))
        
        bn.blocksize = [bls,bls]
        bn.switchOptimizer('sgd',lr)        
        bn.initializeModel()

        bn.adaptDomain(celltype=cell,day=day,epochs=epoch,imnums=imnum,radius=radius,th=th)
        
        if epochs < 65:
            lr = lr*0.5
            
        epochs += epoch
        print('Adapted %s for %i epochs with %s'%(bn.weightname,epochs,cell))
        wstr = 'weights/%s_%s_%i.h5'%(bn.weightname,'adapt',epochs)
        bn.model.load_weights(bn.weightstr)
        bn.model.save_weights(wstr)
        utilityFunctions.killAll(bn)
        
        row = ['Adapted for %i epochs with %s'%(epochs,cell)]
        result_counting.writeRowToFile(row,bn.weightname+'.csv')
        result_counting.writeRowToFile(['no postprocessing, th 0.5'],bn.weightname+'.csv')
        test(bn,postproc=False,epoch=epochs+60,adaptcell=cell)
        result_counting.writeRowToFile(['postprocessing + cell specific th'],bn.weightname+'.csv')
        test(bn,postproc=True)
        
        utilityFunctions.killAll(bn)

    print('Model %s adapted for %i epochs'%(bn.weightname,epochs))



def test(bn,fls=None,postproc=False,epoch=False,adaptcell=False,netname='sm',wstr=False):
    """Count scores. Run after each set of 10 epochs"""
    
    if fls:
        pass
    else:
        fls = [bn.layer]

    bn.blocksize = [512,512]
    bn.initializeModel()

        
    if not wstr:
        print('basic weightstr')
        bn.model.load_weights(bn.weightstr)
    else:
        print('given weightstr')
        bn.model.load_weights(wstr)
        
    totalf1 = []
    totalprec = []
    totalrec = []
    th = 0.5
    utilityFunctions.checkDir('coordinate_npys')
    utilityFunctions.checkDir('result_npys')
    utilityFunctions.checkDir('resultcsvs')
    for i,cell in enumerate(cells):

        for f_,fl_ in enumerate(fls):
       
            day = days[i]
            
            f1s = []
            precs = []
            recs = []            
            row=[]

            for j,imnum in enumerate(imnums):
                print(imnum,day,cell)
                gt = utilityFunctions.loadTestGroundTruth(day=day,imnum=imnum, celltype=cell)
                pred = bn.getResults(celltype=cell,day=day,imnum=imnum,visualize=False,th=th,layers=fl_,lw=False)

                if postproc:
                    if cell=='PC-3':
                        th=0.5
                    elif cell=='LNCaP':
                        th=0.3
                    else:
                        th=0.2
    
                    pointim = utilityFunctions.findPeaks_plm(pred,th=th)

                else:
                    th=0.5
                    pointim = utilityFunctions.getPointIm(pred>th)
                    
                #plt.figure();plt.imshow(pointim)
                #plt.figure();plt.imshow(pred)
                tp,fp,fn,tpgt = result_counting.getCoordinates2(pointim, gt)
                npyfolder = 'coordinate_npys/'
                utilityFunctions.checkDir(npyfolder)
                #save detections and prediction
                sn1 = npyfolder+netname+'_tp_'+cell+'_im'+str(imnum)+'.npy'
                sn2 = npyfolder+netname+'_fp_'+cell+'_im'+str(imnum)+'.npy'
                sn3 = npyfolder+netname+'_fn_'+cell+'_im'+str(imnum)+'.npy'
                sn4 = 'result_npys/'+netname+'_'+cell+'_im'+str(imnum)+'.npy'
                np.save(sn1,tp)
                np.save(sn2,fp)
                np.save(sn3,fn)
                np.save(sn4,pred)

                f1,prec,rec = result_counting.countScores(len(tp),len(fp),len(fn))
                
                print(cell,imnum,f1,prec,rec)
                f1s.append(f1)
                precs.append(prec)
                recs.append(rec)
                
            #print(cell,np.mean(f1s),fl,[l1,l2])
            f1_ = str((np.round(np.mean(f1s)*10000))/10000)
            prec_ = str((np.round(np.mean(precs)*10000))/10000)
            rec_ = str((np.round(np.mean(recs)*10000))/10000)
            row = [cell,str(fl_),f1_,prec_,rec_]
            result_counting.writeRowToFile(row,bn.weightname+'.csv')
            
            print(cell,fl_,np.mean(f1s),np.mean(precs),np.mean(recs)) 
            totalf1.append(np.mean(f1s))
            totalprec.append(np.mean(precs))
            totalrec.append(np.mean(recs))
            
            utilityFunctions.checkDir('plot_csvs')
            
            if epoch:
                if adaptcell:
                    if adaptcell == cell:
                        result_counting.writeRowToFile([epoch,np.mean(f1s)],'plot_csvs/f1_'+cell+'.csv',folder=True)
                        result_counting.writeRowToFile([epoch,np.mean(precs)],'plot_csvs/prec_'+cell+'.csv',folder=True)
                        result_counting.writeRowToFile([epoch,np.mean(recs)],'plot_csvs/rec_'+cell+'.csv',folder=True)
                    
                    
                else:
                    result_counting.writeRowToFile([epoch,np.mean(f1s)],'plot_csvs/f1_'+cell+'.csv',folder=True)
                    result_counting.writeRowToFile([epoch,np.mean(precs)],'plot_csvs/prec_'+cell+'.csv',folder=True)
                    result_counting.writeRowToFile([epoch,np.mean(recs)],'plot_csvs/rec_'+cell+'.csv',folder=True)
            
    row = [np.mean(totalf1),np.mean(totalprec),np.mean(totalrec)]
    result_counting.writeRowToFile(row,bn.weightname+'.csv')

    print(bn.weightname,np.mean(totalf1))

fls = [[13,14,15]]#,[11,12,13,14,15]],[12,16]


mycells = ['22Rv1','BT-474','LNCaP','PC-3']
mydays = [[3,4],[2],[3,4],[3,4]]

if training:
    bn = methodClasses.UNet_small([13,14,15])
    train(bn)

if adapting:
    for i,mycell in enumerate(mycells):

        bn = methodClasses.UNet_small([13,14,15])
        adapt(bn,mycell,mydays[i])

if testing:
    #Testing only
    bn = methodClasses.UNet_small([13,14,15])
    #imnums = [1]
    bn.weightstr = 'weights/unet_sm_13_14_15_da_22Rv1.h5'
    test(bn)

