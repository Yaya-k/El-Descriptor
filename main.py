import cv2
from utils import *
from LBP import *


objets=[9,12,42,51,125,156,200,257,642,787,925,959]
radius = 3
n_points = 8 * radius

#chargement base
print('chargement images')
(imagesTest,imagesLearn)=load('bases')
print('chargement masks')
(masksTest,masksLearn)=load('masks')

#Apprentissage
print('segmentation images apprentissage')
imagesLearn=segmente(imagesLearn,masksLearn)
print('calcul LBP images aprentissage')
hLearn=computeLBPs({k:imagesLearn[k] for k in objets},radius,n_points) #compute all refs


#### image test
print('segmentation images test')
imagesTest=segmente(imagesTest,masksTest)
print('calcul LBP images test')
hTest=computeLBPs({k:imagesTest[k] for k in objets},radius,n_points,False)

## comparaison
print('comparaison')
r=comparaisonTest(hTest,hLearn)
