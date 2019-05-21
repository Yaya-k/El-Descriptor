# -*- coding: cp1252 -*-
import matplotlib.image as mpimg
import numpy as np
import cv2
import scipy
import random
import matplotlib.pyplot as plt
import numpy
import scipy.spatial
objets=[9,12,42,125,51,156,200,257,642,787,925,959]


def load(type):
    imagesTest={}
    imagesLearn={}
    for j in objets:
        print(j)
                
        moonTest=[]
        moonLearn=[]
        i=0
        filename=type+'/'+str(j)+'/'+str(j)+'_r'
        moonTest.append(mpimg.imread(filename+str(int(i))+'.png'))
        for i in np.linspace(5,355,71):
            if i%4==0:
                moonTest.append(mpimg.imread(filename+str(int(i))+'.png'))
            else:
                moonLearn.append(mpimg.imread(filename+str(int(i))+'.png'))
            
        imagesTest[j]=moonTest
        imagesLearn[j]=moonLearn
    return (imagesTest,imagesLearn)

def segmente(imagesSource,masks):
    for j in imagesSource.keys(): # parcourt les objets
        print(j)

        moon=imagesSource[j] # la liste d'images 
        mask=masks[j] # la liste de masks
        for i in range(len(moon)): #parcourt de toutes les images
            img=moon[i]
            mk=mask[i]
            img[mk==0]=(1,1,1) # application des masks
    return imagesSource

# Feature extractor
def extract_features(image, vector_size=32):
  
    alg = cv2.KAZE_create()       
    kps = alg.detect(image) # les descripteurs
    # Obtenir les 32 premiers.
    # Le nombre de points-cles varie en fonction de la taille de l'image et de la palette de couleurs
    # Les trier en fonction de la valeur de réponse (plus c'est grand, mieux c'est)
    kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
    # vecteur de descripteursr
    kps, dsc = alg.compute(image, kps) # calcule le descripteur pour un ensemble de point clis calculer sur l'image
    # Les metre tous dans un grand vecteur 
    dsc = dsc.flatten() # le mettre sur la meme ligne
    # Descripteur de même taille
    # La taille du vecteur descripteur est de 64
    needed_size = (vector_size * 64)
    if dsc.size < needed_size:
        # Si nous avons moins de 32 descripteurs, il suffit d’ajouter des zéros au début
        # fin de notre vecteur de fonctionnalité
        dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    return dsc

def batch_extractor(imagesS):
    print('in extraction')
   # f= open("resultats.txt","w+")
    result = {}
    for i in imagesS.keys():
        moon=[];
        print (i)
        img=imagesS[i]
        for j in range(len(img)):
            moon.append(extract_features(img[j]))            
        result[i] = centeroidnp(moon)
       # f.write("" % (i))# j'ecris dans un fichier
       # f.write("\n" % (result[i]))
    return result

def centeroidnp(cararc):
    ca=numpy.array(cararc)
    ligne,colonne=ca.shape
    res=[]
    for j in range(colonne):
        s=0
        for i in range (ligne):
            s=s+cararc[i][j]/ligne
        res.append(s)
    return res
        
def cos_cdist(results, vector):    
    cosinus={}
    v = vector.reshape(1, -1)
    # je calcule la distance en cosinus entre mon image test et la base
    for i in results.keys():
        print(i)
        m=numpy.array(results[i])
        m=m.reshape(1, -1)
        cosinus[i]=scipy.spatial.distance.cdist(m,v,'cosine').reshape(-1)      
    return cosinus 
    


def match(res, image):
    print('in match')
    features = extract_features(image)
    img_distances = cos_cdist(res,features)        
    return img_distances
