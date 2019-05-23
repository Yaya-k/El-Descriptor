import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
import matplotlib.colors as mpclr
import matplotlib.pyplot as plt

def extractColors(img,n_clusters):
    n_clusters=3
    image=img
    
    clt = KMeans(n_clusters)
    clt.fit(image)

    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)



    return (img,n_clustershist, clt.cluster_centers_)

def colorPrinc(img,n_clusters,MODE='hsv'):
    
    clr=[]
    colorsRGB={'rouge':[1,0,0], 'vert':[0,1,0], 'bleu':[0,0,1],'noir':[0,0,0],'blanc':[1,1,1],'cyan':[0,0.6,0.6],'magenta':[0.6,0.6,0],'jaune':[0.6,0,0.6], 'gris':[0.5,0.5,0.5]}
    colorsHSV={'rouge':0., 'vert':0.33, 'bleu':0.67,'cyan':0.5,'magenta':0.83,'jaune':0.17, 'rouge':1., 'blanc':-1.}
    if MODE=='rgb':
        colors=colorsRGB
        (_,hist, centers)=extractColors(img,n_clusters)
        
    elif MODE=='hsv':
        colors=colorsHSV
        blancs=(img[:,1]<0.1) * (img[:,2]>0.8)
        img1=img[blancs==False]
        (hist,centers)=np.histogram(img1[:,0],n_clusters)
        hist=np.append(hist,len(blancs[blancs==True]))
        centers=np.append(centers,-1)
        
    n=max(hist)
    for i in range(len(hist)):
        if hist[i]>n/4:
            tmpDist=3
            tmpClr=''
            for j in colors.keys():
                d=distance(colors[j],centers[i])
                if d<tmpDist:
                    tmpDist=d
                    tmpClr=j
            clr.append(tmpClr)
    return clr
                
            
def distance(a,b):
    d=0
    if type(a)!=float:
        for i in range(len(a)):
            d=d+(a[i]-b[i])**2
        return np.sqrt(d)
    else:
        return np.sqrt((a-b)**2)

image=mpimg.imread('bases/9/9_r0.png')
mask=mpimg.imread('masks/9/9_r0.png')

img=image
img=mpclr.rgb_to_hsv(img)
img=img[mask!=0]

clr=colorPrinc(img,5,'hsv')
