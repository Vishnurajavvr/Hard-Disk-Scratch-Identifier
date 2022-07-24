from pydoc import classname
from skimage.feature import hog
from skimage.transform import pyramid_gaussian

import joblib 
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2
import os
import glob

import time
start = time.time()

orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

# sliding window:
def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])
#%%
# Upload the saved svm model:
model = joblib.load(r"C:\Users\vishn\Desktop\Hard disk Scratch Detection\model_name.npy") # Model path

scale = 0
detections = []
img= cv2.imread(r"C:\Users\vishn\Desktop\Hard disk Scratch Detection\test_image\10.jpg") # Test image path 

img= cv2.resize(img,(400, 200)) 


(winW, winH)= (64,128)
windowSize=(winW,winH)
downscale=1.4

for resized in pyramid_gaussian(img, downscale=1.4): 
    for (x,y,window) in sliding_window(resized, stepSize=10, windowSize=(winW,winH)): 
        if window.shape[0] != winH or window.shape[1] !=winW: 
            continue
        window=color.rgb2gray(window)
        fds = hog(window, orientations, pixels_per_cell, cells_per_block, block_norm='L2')
        fds = fds.reshape(1, -1) 
        pred = model.predict(fds) 
        
        if pred == 1:
            if model.decision_function(fds) > 0.9:  
                print("Detection:: Location -> ({}, {})".format(x, y))
                print("Scale ->  {} | Confidence Score {} \n".format(scale,model.decision_function(fds)))
                detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fds),
                                   int(windowSize[0]*(downscale**scale)), 
                                      int(windowSize[1]*(downscale**scale))))
    scale+=1
    
clone = resized.copy()

rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections]) 
sc = [score[0] for (x, y, score, w, h) in detections]
print("detection confidence score: ", sc)
sc = np.array(sc)
pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)

for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img,  (xA, yA), (xB, yB), (0,255,0), 1)
    cv2.putText(img,'Scratch',(xA-2,yA-2),1,0.75,(0,0,255),1) 
    
#

end = time.time()
print("The time of execution of above program is :", end-start)

cv2.imshow("Raw Detections after NMS", img)

cv2.imwrite('Scratch_image.jpg', img)
cv2.waitKey(0) 




