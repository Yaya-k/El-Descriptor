import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import skimage.feature


objets=[9,12,42,125,151,156,200,257,642,787,925,959]


radius = 1
n_points = 8 * radius


hs={}
def load(objets):
    images={}
    for j in objets:
        moon=[]
        print(j)        
        i=0
        filename='bases/'+str(j)+'/'+str(j)+'_r'
        moon.append(mpimg.imread(filename+str(int(i))+'.png'))
        for i in np.linspace(5,355,71):
            moon.append(mpimg.imread(filename+str(int(i))+'.png'))
        images[j]=moon
    return images

images=load(objets)

for j in objets:
    moon=images[j]
    lbp=skimage.feature.local_binary_pattern(moon[0],n_points,radius,'ror')

    h1=plt.hist(lbp.reshape(lbp.size),255)
    h=h1[0]

    for i in np.linspace(5,355,71):
        lbp=skimage.feature.local_binary_pattern(moon[int(i/5)],n_points,radius,'ror')
        
        h1=plt.hist(lbp.reshape(lbp.size),255)
        h=np.vstack((h,h1[0]))    
    
    hs[j]=np.mean(h,0)

plt.figure(0)
plt.imshow(moon[int(i/5)],cmap='gray')
plt.show()
