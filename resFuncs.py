# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 11:23:04 2018

@author: kaisapais
"""

import utilityFunctions
import methodClasses
import result_counting


import numpy as np
import matplotlib.pyplot as plt


cells = ['PC-3','LNCaP','BT-474','22Rv1']
days = [3,3,1,3]
imnums = [1,2,3,4]

bn = methodClasses.UNet_small([13,14,15])#[11,12,13,14,15]
bn.blocksize=[512,512]
bn.initializeModel()
#bn.model.load_weights(bn.weightstr)
#bn.model.load_weights('weights/unet_sm_13_14_15.h5')

def countDensityScores(areas=3):
    for c,cell in enumerate(cells):
        day = days[c]
        
        f1s = list(np.zeros((areas,)))
        precs = list(np.zeros((areas,)))
        recs = list(np.zeros((areas,)))
        
        layers = bn.layer
        x_vec = list(range(0,areas))
        th = 0.5
        markers = ['o','x','.','v']
        plt.figure();
        for i,imnum in enumerate(imnums):
            gt = utilityFunctions.loadTestGroundTruth(celltype=cell, imnum=imnum,day=day)
            print(len(gt))
            pred = bn.getResults(celltype=cell,day=day,imnum=imnum,th=th,layers=layers)
            pointim = utilityFunctions.findPeaks_plm(pred,th=th)
            dmap = utilityFunctions.densityMap(gt)
            #dscores = 
            print(cell,imnum)
            f1,prec,rec = result_counting.densityScores(pointim,dmap,gt,areas)
            plt.plot(x_vec,f1,label=str(imnum)+' F1',marker=markers[i],color='r')
            plt.plot(x_vec,prec,label=str(imnum)+' precision',marker=markers[i],color='b')
            plt.plot(x_vec,rec,label=str(imnum)+' recall',marker=markers[i],color='m')
            f1s = [x+y for x,y in zip(f1s,f1)]
            precs = [x+y for x,y in zip(precs,prec)]
            recs = [x+y for x,y in zip(recs,rec)]
        plt.legend(loc='best')
        plt.title(cell)
        plt.show()
        plt.savefig(''.join(['density_scores/',cell,'_',str(areas),'_sep.png']),bbox_inches='tight')
        
        
        f1s = [x/len(imnums) for x in f1s]
        precs = [x/len(imnums) for x in precs]
        recs = [x/len(imnums) for x in recs]
        
        plt.figure();
        plt.plot(x_vec,f1s,'ro-',label='F-score')
        plt.plot(x_vec,precs,'bx-',label='Precision')
        plt.plot(x_vec,recs,'m.-',label='Recall')
        leg = plt.legend(loc='best')
        plt.title(cell)
        plt.show()
        plt.savefig(''.join(['density_scores/',cell,'_',str(areas),'.png']),bbox_inches='tight')


def countWholeDensityScores(areas=3):
    for c,cell in enumerate(cells):
        day = days[c]
        th = 0.5
        layers = bn.layer

        x_vec = list(range(0,areas))
        gts = []
        pointims = []
        for i,imnum in enumerate(imnums):
            gt = utilityFunctions.loadTestGroundTruth(celltype=cell, imnum=imnum,day=day)
            gts.append(gt)
            print(len(gt))
            pred = bn.getResults(celltype=cell,day=day,imnum=imnum,th=th,layers=layers)
            pointim = utilityFunctions.findPeaks_plm(pred,th=th)
            pointims.append(pointim)
        
        pi1 = np.concatenate((pointims[0],pointims[1]),axis=1)
        pi2 = np.concatenate((pointims[2],pointims[3]),axis=1)
        pointim = np.concatenate((pi1,pi2),axis=1)
        print(pi1.shape,pointim.shape)
        
        
        
        [sy,sx] = utilityFunctions.loadShape()

        for i in range(len(gts[1])):
            gts[1][i][1] += sx
        for i in range(len(gts[2])):
            gts[2][i][1] += 2*sx
        for i in range(len(gts[3])):
            gts[3][i][1] += 3*sx
        gt = gts[0]+gts[1]+gts[2]+gts[3]
        
        dmap = utilityFunctions.densityMap(gt,mult=[1,4])
        #utilityFunctions.visualize([dmap])
        #dscores = 
        print(cell,imnum)
        f1,prec,rec,cellperc = result_counting.densityScores(pointim,dmap,gt,areas,mult=[1,4])
        
        plt.figure();
        plt.plot(x_vec,f1,'ro-',label='F-score')
        plt.plot(x_vec,prec,'bx-',label='Precision')
        plt.plot(x_vec,rec,'m.-',label='Recall')
        plt.plot(x_vec,cellperc,'gv',label='Cell percentage')
        leg = plt.legend(loc='best')
        plt.title(cell)
        plt.show()
        plt.savefig(''.join(['density_scores/whole_1x4_best_',cell,'_',str(areas),'.png']),bbox_inches='tight')
        
        
def countTotalDensityScores(areas=3,fname=''):
    allims = []
    allgts = []
    for c,cell in enumerate(cells):
        day = days[c]
        layers = bn.layer
        th = 0.5

        x_vec = list(range(0,areas))
        gts = []
        pointims = []
        for i,imnum in enumerate(imnums):
            gt = utilityFunctions.loadTestGroundTruth(celltype=cell, imnum=imnum,day=day)
            gts.append(gt)
            print(len(gt))
            pred = bn.getResults(celltype=cell,day=day,imnum=imnum,th=th,layers=layers,lw=False)
            #pointim = utilityFunctions.findPeaks(pred,th=th)
            pointim = utilityFunctions.getPointIm(pred>th)
            pointims.append(pointim)
        
        pi1 = np.concatenate((pointims[0],pointims[1],pointims[2],pointims[3]),axis=1)
        allims.append(pi1)
        print(pi1.shape)
        
        
        
        [sy,sx] = utilityFunctions.loadShape()

        for i in range(len(gts[1])):
            gts[1][i][1] += sx
        for i in range(len(gts[2])):
            gts[2][i][1] += 2*sx
        for i in range(len(gts[3])):
            gts[3][i][1] += 3*sx
        allgts.append(gts[0]+gts[1]+gts[2]+gts[3])
        
    pointim = np.concatenate((allims[0],allims[1],allims[2],allims[3]),axis=0)
    
    for i in range(len(allgts[1])):
        allgts[1][i][0] += sy
    for i in range(len(allgts[2])):
        allgts[2][i][0] += 2*sy
    for i in range(len(allgts[3])):
        allgts[3][i][0] += 3*sy

    gt = allgts[0] + allgts[1] + allgts[2] + allgts[3]
    dmap = utilityFunctions.densityMap(gt,mult=[4,4],radius=40)
    utilityFunctions.visualize([dmap])
    #dscores = 
    #print(cell,imnum)
    f1,prec,rec,cellperc = result_counting.densityScores(pointim,dmap,gt,areas,mult=[4,4])
    
    for i in range(len(x_vec)):
        result_counting.writeRowToFile([x_vec[i],f1[i]],'plot_csvs/density_f1_'+fname+'.csv',folder=True)
        result_counting.writeRowToFile([x_vec[i],prec[i]],'plot_csvs/density_prec_'+fname+'.csv',folder=True)
        result_counting.writeRowToFile([x_vec[i],rec[i]],'plot_csvs/density_rec_'+fname+'.csv',folder=True)
        result_counting.writeRowToFile([x_vec[i],cellperc[i]],'plot_csvs/density_cellperc.csv',folder=True)
    
    
    
    
    plt.figure();
    plt.plot(x_vec,f1,'ro-',label='F-score')
    plt.plot(x_vec,prec,'bx-',label='Precision')
    plt.plot(x_vec,rec,'m.-',label='Recall')
    plt.plot(x_vec,cellperc,'gv',label='Cell percentage')
    leg = plt.legend(loc='best')
    plt.title('All')
    plt.show()
    plt.savefig(''.join(['density_scores/all_best_',str(areas),'.png']),bbox_inches='tight')


def scoreAndSave(net=None):
    if net:
        pass
    else:
        net=bn
        
    

bn.model.load_weights('weights/unet_sm_13_14_15.h5')    
countTotalDensityScores(areas=5)
#bn.model.load_weights('weights/unet_sm_13_14_15_da_LNCaP.h5')    
#countTotalDensityScores(areas=5,fname='daLNCaP')
#bn.model.load_weights('weights/unet_sm_13_14_15_da_BT-474.h5')    
#countTotalDensityScores(areas=5,fname='daBT-474')
#bn.model.load_weights('weights/unet_sm_13_14_15_da_22Rv1.h5')    
#countTotalDensityScores(areas=5,fname='da22Rv1')
