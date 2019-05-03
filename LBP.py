import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import skimage.feature
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

        moon=images[j]
        lbp=skimage.feature.local_binary_pattern(moon[0],n_points,radius,'uniform') #premiere image
        
        h1=np.histogram(lbp.reshape(lbp.size),255) #creation histogramme
        h1=np.take(h1[0],np.where(h1[0]>0))[0] 
        
        h=np.zeros((72,n_points+2),dtype=np.int64)
        h[0]=h1
        for i in range(len(moon)): #parcourt de toutes les images
            
            lbp=skimage.feature.local_binary_pattern(moon[i],n_points,radius,'uniform')
            
            h1=np.histogram(lbp.reshape(lbp.size),255)
            h1=np.take(h1[0],np.where(h1[0]>0))[0]
            
            h[i]=h1
        hs[j]=np.mean(h,0) #moyenne 
    return hs

plt.figure(0)

hs=computeLBPs({k:images[k] for k in objets}) #compute all refs

## image test aleatoire
obj=objets[random.randint(0,12)]
rotation=random.randint(0,73)
htest=computeLBPs({0:[images[obj][rotation]]})


## comparaison
r=[]
r2=[]
plt.figure(0)
for i in hs.keys():    
    plt.plot(np.take(hs[i],np.where(hs[i]>0))[0],'.')
    r.append(cv2.compareHist(np.float32(hs[i]),np.float32(htest[0]),0))
    r2.append(cv2.compareHist(np.float32(hs[i]),np.float32(htest[0]),3))


plt.figure(1)
plt.imshow(images[obj][rotation],cmap='gray')

plt.show()
