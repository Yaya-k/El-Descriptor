import matplotlib.image as mpimg
import numpy as np


objets=[9,12,42,125,151,156,200,257,642,787,925,959]


def load(objets):
    images={}
    for j in objets:
        moon=[]      
        i=0
        filename='bases/'+str(j)+'/'+str(j)+'_r'
        moon.append(mpimg.imread(filename+str(int(i))+'.png'))
        for i in np.linspace(5,355,71):
            moon.append(mpimg.imread(filename+str(int(i))+'.png'))
        images[j]=moon
    return images
