import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import skimage.feature
from skimage import io as skio
from skimage import filters
from scipy import ndimage as ndi
from skimage import morphology
import copy
from load import *
import random
import cv2

random.seed()

objets=[9,12,42,125,151,156,200,257,642,787,925,959]
radius = 3
n_points = 8 * radius


T={}

images=load(objets)
def computeLBPs(images):
    
    hs={}
    for j in images.keys(): #parcourt les objets
        print(j)

        moon=copy.copy(images[j])
        lbp=skimage.feature.local_binary_pattern(segment(moon[0]),n_points,radius,'uniform') #premiere image
        
        h1=np.histogram(lbp.reshape(lbp.size),255) #creation histogramme
        h1=np.take(h1[0],np.where(h1[0]>0))[0] 
        
        h=np.zeros((72,n_points+2),dtype=np.int64)
        h[0]=h1
        for i in range(len(moon)): #parcourt de toutes les images
            
            lbp=skimage.feature.local_binary_pattern(segment(moon[i]),n_points,radius,'uniform')
            
            h1=np.histogram(lbp.reshape(lbp.size),255)
            h1=np.take(h1[0],np.where(h1[0]>0))[0]
            
            h[i]=h1
        hs[j]=h #moyenne : hs[j]=np.mean(h,0)
    return hs

plt.figure(0)

#hs=computeLBPs({k:images[k] for k in objets}) #compute all refs

#### image test aleatoire
##obj=objets[random.randint(0,12)]
##rotation=random.randint(0,73)
##htest=computeLBPs({0:[images[obj][rotation]]})
##
##
#### comparaison
##r=[]
##r2=[]
##plt.figure(0)
##for i in hs.keys():
##    for j in range(hs[i].shape[0]):
##        plt.plot(np.take(hs[i],np.where(hs[i]>0))[0],'.')
##        r.append(cv2.compareHist(np.float32(hs[i][j]),np.float32(htest[0]),0))
##        r2.append(cv2.compareHist(np.float32(hs[i][j]),np.float32(htest[0]),3))
##
##print(argmin(r),argmin(r2))
##plt.figure(1)
##plt.imshow(images[obj][rotation],cmap='gray')
##
##plt.show()
def segment(img):
    img=img*255
    mask=cv2.inRange(img,(18),(255));
    strel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, strel)

    kernel = np.ones((8,8),np.uint8)

    dilation = cv2.dilate(closing,kernel,iterations = 1)
    kernel = np.ones((1,10),np.uint8)
    erotion=cv2.erode(dilation,kernel,iterations=1)

    kernel=np.ones((7,7),np.uint8)
    dilation=cv2.dilate(erotion,kernel,iterations=1)

    kernel=np.ones((1,10),np.uint8)
    erosion=cv2.erode(dilation,kernel,iterations=1)

    kernel=np.ones((7,7),np.uint8)

    dilation=cv2.dilate(erosion,kernel,iterations=1)

    kernel=np.ones((2,2),np.uint8)
    erosion=cv2.erode(dilation,kernel,iterations=1)
    img[erosion==0]=(255)

return  np.float32(img/255)
        
