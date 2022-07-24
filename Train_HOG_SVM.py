# Importing the necessary modules:

from skimage.feature import hog#
from skimage.transform import pyramid_gaussian
from skimage.io import imread

import joblib 
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC 
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split 
from skimage import color 
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np 
import argparse
import cv2
import os
import glob
from PIL import Image 
from numpy import *

import time
start = time.time()

# define parameters of HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3


# define path to images:

pos_im_path = r"C:\Users\vishn\Desktop\Hard disk Scratch Detection\Positive_image" #positive input dataset
# define the same for negatives
neg_im_path= r"C:\Users\vishn\Desktop\Hard disk Scratch Detection\negative_image" # Negative input dataset

# read the image files:
pos_im_listing = os.listdir(pos_im_path)
neg_im_listing = os.listdir(neg_im_path)
num_pos_samples = size(pos_im_listing) 
num_neg_samples = size(neg_im_listing)
print(num_pos_samples) 
print(num_neg_samples)
data= []
labels = []

# compute HOG features and label them:

for file in pos_im_listing: 
    img = Image.open(pos_im_path + '\\' + file) 
    img = img.resize((64,128)) 
    gray = img.convert('L') 
    # calculate HOG for positive features
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)#, channel_axis=-1)
    data.append(fd)
    labels.append(1)
    
# negative images
for file in neg_im_listing:
    img= Image.open(neg_im_path + '\\' + file)
    img = img.resize((64,128))
    gray= img.convert('L') 
    # Now we calculate the HOG for negative features
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)#, channel_axis=-1)
    data.append(fd)
    labels.append(0)

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

#%%

print(" Constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	np.array(data), labels, test_size=0.20, random_state=50) ##42

#%% Train the linear SVM
print(" Training Linear SVM classifier...")
model = LinearSVC()
model.fit(trainData, trainLabels)

#%% Evaluate the classifier
print(" Evaluating classifier on test data ...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))


# Save the model:
#%% Save the Model
joblib.dump(model, 'model_name.npy') 


end = time.time()
print("The time of execution of above program is :", end-start)
