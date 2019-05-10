import numpy as np
import cv2 
from skimage import io as skio
from skimage import filters
from scipy import ndimage as ndi
from skimage import morphology

# Load an color image in grayscale
# nous allons commencer par charger tous les images de la dase
path='bases image'
img = cv2.imread('bases image/9/9_r0.png')
mask=cv2.inRange(img,(18,18,18),(255,255,255));
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
# height = np.size(img, 0)
# width = np.size(img, 1)
# noir=[0,0,0];
# blanc=[255,255,255]
# print(np.size(img,0))
# print(np.size(img,1))
#closing=~closing
img[erosion==0]=(255,255,255)
#maskedImg = cv2.bitwise_and(img, img,mask=closing)

# for i in range(height):
	# for j in range(width):
		# print(closing[i,j].get)
		# if(closing[i,j].get==0):
			# img[i,j]=blanc


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



