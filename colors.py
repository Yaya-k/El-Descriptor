import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
import matplotlib.colors as mpclr
import matplotlib.pyplot as plt

def extractColors(img):
    image=img
    n_clusters=8
    clt = KMeans(n_clusters,n_init=1,init=np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0],[1,1,1],[0,0.8,0.8],[0.8,0.8,0],[0.8,0,0.8]]))
    clt.fit(image)

    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)



    return (hist, clt.cluster_centers_)

def colorPrinc(img,MODE='hsv'):
    
    clr=[]
    colorsRGB={'rouge':[1,0,0], 'vert':[0,1,0], 'bleu':[0,0,1],'noir':[0,0,0],'blanc':[1,1,1],'cyan':[0,0.8,0.8],'magenta':[0.8,0.8,0],'jaune':[0.8,0,0.8]}
    colorsHSV={0:'rouge',0.4:'vert', 0.67:'bleu', -1:'blanc'} 
    if MODE=='rgb':
        colors=colorsRGB
        (hist, centers)=extractColors(img)

        n=max(hist)
        for i in range(len(hist)):
            if hist[i]>n/2:
                tmpDist=3
                tmpClr=''
                for j in colors.keys():
                    d=distance(colors[j],centers[i])
                    if d<tmpDist:
                        tmpDist=d
                        tmpClr=j
                clr.append(tmpClr)

        clr=list(set(clr))
        
    elif MODE=='hsv':
        colors=colorsHSV
        blancs=(img[:,1]<0.35) * (img[:,2]>0.65)
        img1=img[blancs==False]
        (hist,_)=np.histogram(img1[:,0],bins=[0,0.2,0.53,0.83,1])

        centers=[0,0.4,0.67]
        hist[0]+=hist[-1]
        hist=hist[:-1]
        hist=np.append(hist,len(blancs[blancs==True]))
        centers=np.append(centers,-1)

        n=max(hist)
        for i in range(len(hist)):
            if hist[i]>n/4:
                clr.append(colorsHSV[centers[i]])
        

    return clr
                
            
def distance(a,b):
    d=0
    for i in range(len(a)):
        d=d+(a[i]-b[i])**2
    return np.sqrt(d)

objets=[9,12,42,125,51,156,200,257,642,787,925,959]

for i in objets :
    image=mpimg.imread('bases/'+str(i)+'/'+str(i)+'_r0.png')
    mask=mpimg.imread('masks/'+str(i)+'/'+str(i)+'_r0.png')
    
    img=image
    img=mpclr.rgb_to_hsv(img)
    img1=img[mask!=0]
    clr=colorPrinc(img1,'rgb')
    print(i,clr)
    clr=colorPrinc(img1,'hsv')
    print(i,clr)
