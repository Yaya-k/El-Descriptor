import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.image as mpimg

def extractColors(img,n_clusters):
    n_clusters=3
    image = img.reshape((img.shape[0] * img.shape[1], 3))
    np.take(image[0],np.where(image[0]>0))[0]
    
    clt = KMeans(n_clusters)
    clt.fit(image)

    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    print(numLabels)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)



    return (hist, clt.cluster_centers_)

def colorPrinc(hist,centers):
    n=max(hist)
    clr=[]
    colors={'rouge':[1,0,0], 'vert':[0,1,0], 'bleu':[0,0,1],'noir':[0,0,0],'blanc':[1,1,1],'cyan':[0,0.6,0.6],'magenta':[0.6,0.6,0],'jaune':[0.6,0,0.6]}
    for i in range(len(hist)):
        if hist[i]>n/3:
            tmpDist=3
            tmpClr=''
            for j in colors.keys():
                d=distance(colors[j],centers[i])
                if d<tmpDist:
                    tmpDist=d
                    tmpClr=i
            clr.append(tmpClr)
    return clr
                
            
def distance(a,b):
    d=0
    print(a,b)
    for i in range(len(a)):
        d=d+(a[i]-b[i])**2
    return np.sqrt(d)

img=mpimg.imread('bases/9/9_r0.png')
mask=mpimg.imread('masks/9/9_r0.png')
img[mask==0]=(1,1,1)
(his,centers)=extractColors(img,10)
clr=colorPrinc(his,centers)
