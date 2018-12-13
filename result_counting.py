# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:22:53 2018

@author: kaisapais
"""
import utilityFunctions

import numpy as np
import csv
from operator import add,sub
from scipy import misc
from PIL import Image
from skimage import morphology

import matplotlib.pyplot as plt


cells = ['PC-3','LNCaP','BT-474','22Rv1']
testdays = [3,3,1,3]
trainp, testp = utilityFunctions.loadPaths()

def testGroundtruth(celltype='BT-474',day=1,imnum=3):
    fname = testp+'/ground_truth/test/' + celltype + '/day'+str(day)+'/image_'+str(imnum)+'.txt'
    
    l = []
    with open(fname,'r') as csvfile:
        freader = csv.reader(csvfile, delimiter=',')
        #l = []
        firstLine = True
        for row in freader:
            if firstLine:
                firstLine = False
                continue
            r = []

            r.append(int(float(row[1])))#y
            r.append(int(float(row[0])))#x
       
                
            l.append(r)
            #r = []
            #r.append(int(row[2]))#y
            #r.append(int(row[1]))#x
                    
            #l.append(r)
            
    return l

def writeResultFile(scores,fname='test001.csv',description='test results'):
    fname = 'resultcsvs/'+fname
    with open(fname,'w') as csvfile:
        csvwriter = csv.writer(csvfile,delimiter=',',lineterminator='\n')
        runner = 0
        csvwriter.writerow([description])
        for i,cell in enumerate(cells):
            for imnum in range(1,5):
                f1 = round(1000*scores[runner][0])/1000
                prec = round(1000*scores[runner][1])/1000
                rec = round(1000*scores[runner][2])/1000
                csvwriter.writerow([cell,imnum,f1,prec,rec])
                runner += 1    
                
def writeRowToFile(scores,fname='test001.csv',folder=False):
    if not folder:
        fname = 'resultcsvs/'+fname
    with open(fname,'a') as csvfile:
        csvwriter = csv.writer(csvfile,delimiter=',',lineterminator='\n')
        #for i,val in enumerate(scores):
        #    row.append(str(val))
            
        csvwriter.writerow(scores)


def writeAllFile(scores,fname='test001.csv'):
    fname = 'csvfiles/'+fname
    print(scores)
    with open(fname,'w') as csvfile:
        csvwriter = csv.writer(csvfile,delimiter=' ',lineterminator='\n')
        for i in list(range(4)):
            csvwriter.writerow([round(10000*scores[i])/10000])


                
def countValvals(im,gt,fromGT=True):
    """Count scores for image where single point marks found cell"""
    foundy,foundx = np.where(im)
    gtamount = len(gt)
    foundamount = len(foundy)
    mindist = 20
    tp = 0
    if fromGT:
        for i,gtcoo in enumerate(gt):
            [y,x] = gtcoo
            dy = [(y-a)**2 for a in foundy]
            dx = [(x-a)**2 for a in foundx]
            dists = np.sqrt(list(map(add,dy,dx)))
            if len(dists)>0:
                if np.min(dists) < mindist:
                    minind = np.argmin(dists)
                    foundy = np.delete(foundy,minind)
                    foundx = np.delete(foundx,minind)
                    tp += 1
                
    else:
        gt_ = list(gt)
    
        for i in range(len(foundy)):
            y = foundy[i]
            x = foundx[i]
            d = [list(map(sub,[y,x],b)) for b in gt_]
            dists = np.sqrt([a[0]**2 + a[1]**2 for a in d])
            if np.min(dists) < mindist:
                minind = np.argmin(dists)
                del gt_[minind]
                tp += 1
    
    fp = foundamount - tp
    fn = gtamount - tp
    return tp,fp,fn


def getCoordinates(im,gt,fromGT=True):
    """Count scores for image where single point marks found cell"""
    foundy,foundx = np.where(im)
    gtamount = len(gt)
    foundamount = len(foundy)
    mindist = 20
    tp = []
    fp = []
    fn = []
    tp_ = 0
    if fromGT:
        for i,gtcoo in enumerate(gt):
            [y,x] = gtcoo
            dy = [(y-a)**2 for a in foundy]
            dx = [(x-a)**2 for a in foundx]
            dists = np.sqrt(list(map(add,dy,dx)))
            if np.min(dists) < mindist:
                minind = np.argmin(dists)
                tp.append([foundy[minind],foundx[minind]])
                foundy = np.delete(foundy,minind)
                foundx = np.delete(foundx,minind)
                tp_ += 1
            else:
                fn.append([y,x])
                
    else:
        gt_ = list(gt)
    
        for i in range(len(foundy)):
            y = foundy[i]
            x = foundx[i]
            d = [list(map(sub,[y,x],b)) for b in gt_]
            dists = np.sqrt([a[0]**2 + a[1]**2 for a in d])
            if np.min(dists) < mindist:
                minind = np.argmin(dists)
                del gt_[minind]
                tp += 1
    
    for i in range(len(foundy)):
        fp.append([foundy[i],foundx[i]])
    fp_ = foundamount - tp_
    fn_ = gtamount - tp_
    print('TP:',tp_,'FP:',fp_,'FN:',fn_)
    return tp,fp,fn


def getCoordinates2(im,gt):
    """Count scores for image where single point marks found cell"""
    foundy,foundx = np.where(im)
    gtamount = len(gt)
    
    foundamount = len(foundy)
    if foundamount==0:
        return [],[],gt,[]
    distsmatrix = np.zeros((gtamount,foundamount),'float32')
    mindist = 20
    tp = []
    tpgt = []
    fp = []
    fn = []
    tp_ = 0

    # Create matrix with all euclidean distances
    for i,gtcoo in enumerate(gt):
        [y,x] = gtcoo
        dy = [(y-a)**2 for a in foundy]
        dx = [(x-a)**2 for a in foundx]
        dists = np.sqrt(list(map(add,dy,dx)))
        distsmatrix[i,:] = dists
        
    # Find clear hits from matrix
    foundinds = []
    gtinds = []
    while np.min(distsmatrix)<mindist:
        tpinds = []
        for i in range(gtamount):
            values = distsmatrix[i,:]
            minind = np.argmin(values)
            minval = values[minind]
            values2 = distsmatrix[:,minind]
    
            minind2 = np.argmin(values2)
            if minind2 == i and minval < mindist:
                distsmatrix[i,:] = 2000
                distsmatrix[:,minind] = 2000
                tp_ += 1
                tp.append([foundy[minind],foundx[minind]])
                tpgt.append(gt[i])
                foundinds.append(minind)
                gtinds.append(i)
    
    
    # Find indices not used yet
    gtnotused = set(list(range(gtamount))) - set(gtinds)
    foundnotused = set(list(range(foundamount))) - set(foundinds)
    for i,val in enumerate(foundnotused):
        fp.append([foundy[val],foundx[val]])
        
    for i,val in enumerate(gtnotused):
        fn.append([gt[val][0],gt[val][1]])
        
    fp_ = foundamount - tp_
    fn_ = gtamount - tp_
#    print('TP:',tp_,'FP:',fp_,'FN:',fn_)            
        
    return tp,fp,fn,tpgt
#        if np.min(dists) < mindist:
#            minind = np.argmin(dists)
#            tp.append([foundy[minind],foundx[minind]])
#            foundy = np.delete(foundy,minind)
#            foundx = np.delete(foundx,minind)
#            tp_ += 1
#        else:
#            fn.append([y,x])
#                
#
#    
#    for i in range(len(foundy)):
#        fp.append([foundy[i],foundx[i]])
#    fp_ = foundamount - tp_
#    fn_ = gtamount - tp_
#    print('TP:',tp_,'FP:',fp_,'FN:',fn_)
#    return tp,fp,fn

def scoresImage(im,tp,fp,fn):
    im = utilityFunctions.normalize(im[0])
    imrgb = np.dstack((im,im,im))
    tpim = np.zeros(im.shape,'float32')
    fpim = np.copy(tpim)
    fnim = np.copy(tpim)
    for i,p in enumerate(tp):
        tpim[p[0],p[1]] = 1
    bs = morphology.disk(4)
    diltp = morphology.binary_dilation(tpim,bs).astype('float32')
    for i,p in enumerate(fp):
        fpim[p[0],p[1]] = 1
    dilfp = morphology.binary_dilation(fpim,bs).astype('float32')
    for i,p in enumerate(fn):
        fnim[p[0],p[1]] = 1
    dilfn = morphology.binary_dilation(fnim,bs).astype('float32')
    imrgb[np.where(diltp==1)] = [1,0,0]
    imrgb[np.where(dilfp==1)] = [0,1,0]
    imrgb[np.where(dilfn==1)] = [0,0,1]

    #plt.figure();plt.imshow(imrgb)
    return imrgb
            
def countScores(tp,fp,fn):    
    tpfp = tp+fp
    if tpfp == 0:
        prec = 0
    else:
        prec = tp/tpfp
    #print(i,prec)
    tpfn = tp+fn
    if tpfn == 0:
        rec = 0
    else:
        rec = tp/tpfn
    
    #print(i,rec)
    if (prec+rec) == 0:
        f1 = 0 #class not present
    else:
        f1 = 2*(prec*rec)/(prec+rec)
                
    return f1, prec,rec
                
def allResults(netobject,scoresfile='test001.csv',description='test scores'):
    """Count scores for all test images and write to file"""
    scorelist = []
    for i,cell in enumerate(cells):
        for imnum in range(1,5):
            predim,peakim = netobject.getResults(cell,testdays[i],imnum)
            gt = testGroundtruth(cell,testdays[i],imnum)
            tp,fp,fn = countValvals(peakim,gt)
            f1,prec,rec = countScores(tp,fp,fn)
            scorelist.append([f1,prec,rec])
            print(cell,f1)
            #print(cell,imnum,f1,prec,rec)
            
    writeResultFile(scorelist,scoresfile,description)
    
def drawAllResults(netobject,scoresfile='net13'):
    """Count scores for all test images and write to file"""
    trainp,testp=dataFuncs.loadPaths()
    days = [3,3,1,3]
    f1s = []
    precs = []
    recs = []
    for i,cell in enumerate(cells):
        f1_ = []
        prec_ = []
        rec_ = []
        for imnum in range(1,5):
            predim,peakim = netobject.getResults(cell,testdays[i],imnum)
            np.save('small_npys/'+cell+'_'+str(imnum)+'_'+scoresfile+'.npy',predim)
            focusedim = misc.imread(testp+'/'+cell+'/day'+str(days[i])+'/image_'+str(imnum)+'.bmp')
            cellim = drawCells(np.dstack((focusedim,focusedim,focusedim)),peakim)
            Image.fromarray(cellim).save('resIms_3d/'+cell+'_'+str(imnum)+'_'+scoresfile+'.png')
            gt = testGroundtruth(cell,testdays[i],imnum)
            tp,fp,fn = countValvals(peakim,gt)
            f1,prec,rec = countScores(tp,fp,fn)
            f1_.append(f1)
            prec_.append(prec)
            rec_.append(rec)
            #print(cell,f1)
        f1s.append(np.mean(f1_))
        precs.append(np.mean(prec_))
        recs.append(np.mean(rec_))
            #print(cell,imnum,f1,prec,rec)
    writeAllFile(f1s,scoresfile+'_f1.csv')   
    writeAllFile(precs,scoresfile+'_prec.csv')    
    writeAllFile(recs,scoresfile+'_rec.csv')

    



def drawCells(orig,res):
    

    #TMP
    trueMask = morphology.binary_dilation(res,selem=morphology.disk(7,int)).astype(int)# - morphology.binary_dilation(res,selem=morphology.disk(3,int)).astype(int)
    
    y,x=np.where(trueMask)
    orig[y,x,0] = 38#21
    orig[y,x,1] = 81#70
    orig[y,x,2] = 7#78
    trueMask = morphology.binary_dilation(res,selem=morphology.disk(4,int)).astype(int)# - morphology.binary_dilation(res,selem=morphology.disk(3,int)).astype(int)
    
    y,x=np.where(trueMask)
    orig[y,x,0] = 212
    orig[y,x,1] = 202
    orig[y,x,2] = 136
    
        
    return orig
                

def densityScores(pointim,densim,gt,areas=3,mult=[1,1],fname='density_info.csv'):
    gtim = utilityFunctions.gtImage(gt,mult=mult)
    lims = list(range(0,100,round(100/areas)))
    if len(lims)==areas:
        lims.append(100)
    
    masks = []
    for i,val in enumerate(lims):
        if i == 0:
            continue
        elif i>0 and i<(len(lims)-1):
            di = (densim<=val).astype(int) - (densim<lims[i-1]).astype(int)
            #print(i,val)
            
        else:
            di = (densim>=lims[i-1]).astype(int)
            #print('last',i,val)
        masks.append(di)
    f1s = []
    precs = []
    recs = []
    cellperc = []
    
    for i,mask in enumerate(masks):
        pi = pointim*mask
        gim = gtim*mask
        newgt = np.where(gim)
        print(len(newgt[0]),len(np.where(mask)[0]))
        writeRowToFile(['area '+str(i)],fname)
        writeRowToFile([str(len(newgt[0]))+' cells',str(len(np.where(mask)[0]))+' pixels'],fname)
        
        newgt = [x.tolist() for x in newgt]
        newgt = [[x,y] for x,y in zip(newgt[0],newgt[1])]
        #print(len(newgt))
        tp,fp,fn,tpgt = getCoordinates2(pi,newgt)
        f1,prec,rec = countScores(len(tp),len(fp),len(fn))
        print(i,f1,prec,rec)
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)
        cellperc.append(len(tp)+len(fn))
        sy,sx=utilityFunctions.loadShape()
            
        if densim.shape[1] > 2000:
            #domains in area
            gimtmp = np.copy(gim)
            gimtmp[sy:,:] = 0
            pc = len(np.where(gimtmp)[0])
            gimtmp = np.copy(gim)
            gimtmp[0:sy,:] = 0
            gimtmp[sy*2:,:] = 0
            ln = len(np.where(gimtmp)[0])
            gimtmp = np.copy(gim)
            gimtmp[0:sy*2,:] = 0
            gimtmp[sy*3:,:] = 0
            bt = len(np.where(gimtmp)[0])
            gimtmp = np.copy(gim)
            gimtmp[0:sy*3,:] = 0
            rv = len(np.where(gimtmp)[0])
            print('PC-3',pc,'LNCaP',ln,'BT-474',bt,'22Rv1',rv)
            writeRowToFile(['PC-3',pc],fname)
            writeRowToFile(['LNCaP',ln],fname)
            writeRowToFile(['BT-474',bt],fname)
            writeRowToFile(['22Rv1',rv],fname)
    
    am = len(gt)
    cellperc = [x/am for x in cellperc]
        
    return f1s,precs,recs,cellperc
    #visualize(masks)
    #visualize([densim,(densim<lims[1]).astype(int),(densim>=lims[1]).astype(int) - (densim>=lims[2]).astype(int),(densim>=lims[2]).astype(int)])

def plotScoresFromFile(fname='resultcsvs/unet_smer_13_14_15.csv'):
    f1s = []
    precs = []
    recs = []
    with open(fname,'r') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=',',lineterminator='\n')
        for row in csvreader:
            
            if len(row) == 5:
                f1s.append(row[2])
                precs.append(row[3])
                recs.append(row[4])
    #f1s = recs
    pci = list(range(0,len(f1s),8))            
    pc = [float(f1s[x]) for x in pci]
    pci = [x+1 for x in pci]
    ln = [float(f1s[x]) for x in pci]
    pci = [x+1 for x in pci]
    bt = [float(f1s[x]) for x in pci]
    pci = [x+1 for x in pci]
    rv = [float(f1s[x]) for x in pci]
    x = list(range(len(pc)))
    print(pc,ln,bt,rv)
    plt.figure()
    plt.plot(x,pc,'rx-',label='PC-3')
    plt.plot(x,ln,'bo-',label='LNCaP')
    plt.plot(x,bt,'g+-',label='BT-474')
    plt.plot(x,rv,'kv-',label='22Rv1')
    plt.legend(loc='best')
    #yt = list(range(5,10))
    #yt = [y/10 for y in yt]
    xlab = [10,20,30,40,50,60,10,20,30,40,50,60]
    xlab = [str(y) for y in xlab]
    plt.xticks(x,xlab)
    plt.title('Domain adaptation with 22Rv1 (recall)')
    #plt.yticks(yt)
        
