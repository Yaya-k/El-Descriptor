import numpy as np
import skimage.feature
import copy
import cv2
from utils import rgb2gray

objets=[9,12,42,51,125,156,200,257,642,787,925,959]

def computeLBPs(images,radius,n_points,compression=True):
    
    hs={}
    for j in images.keys(): #parcourt les objets

        moon=copy.copy(images[j])
        lbp=skimage.feature.local_binary_pattern(rgb2gray(moon[0]),n_points,radius,'uniform') #premiere image
        
        h1=np.histogram(lbp.reshape(lbp.size),255) #creation histogramme
        h1=np.take(h1[0],np.where(h1[0]>0))[0] 
        
        h=np.zeros((len(moon),n_points+2),dtype=np.int64)
        h[0]=h1
        for i in range(len(moon)): #parcourt de toutes les images
            
            lbp=skimage.feature.local_binary_pattern(rgb2gray(moon[i]),n_points,radius,'uniform')
            
            h1=np.histogram(lbp.reshape(lbp.size),255)
            h1=np.take(h1[0],np.where(h1[0]>0))[0]
            
            h[i]=h1
        if compression :
            h=(h[2:]+h[1:-1]+h[:-2])/3
            h=h[:-1:3]
            #h=sum(h,0)/len(h)
        hs[j]=h
    return hs


def comparaisonTest(hTest,hLearn):
    for i in hTest.keys():
        for j in range(len(hTest[i])):
            r=comparaison(hLearn,hTest[i][j])
            print(i,r)
    return 0

def comparaison(hLearn, h):
    r=np.zeros([len(hLearn.keys()),len(hLearn[9])])
    kp=0
    for k in hLearn.keys():
        for l in range(len(hLearn[k])):
            r[kp,l]=cv2.compareHist(np.float32(h),np.float32(hLearn[k][l]),0)
        kp+=1
    return hLearn.keys()[np.argmax(r)]




