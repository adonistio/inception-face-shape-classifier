#This script detects the face(s) in the image, specifies the bounding box, detects the facial landmarks, and extracts the features for training

import numpy as np  
import cv2  
import dlib  
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
import os
import imutils
import math

def get_norm(image,x1,y1,x2,y2):
	x = (int(image[x1][y1][0])-int(image[x2][y2][0]))**2
	y = (int(image[x1][y1][1])-int(image[x2][y2][1]))**2
	z = (int(image[x1][y1][2])-int(image[x2][y2][2]))**2
	norm = x + y + z
	return norm

def get_min(image,x1,y1):
	x = int(image[x1][y1][0])
	y = int(image[x1][y1][1])
	z = int(image[x1][y1][2])
	return np.min([x,y,z])

def get_color(image,x,y,gray):
	if gray == 1: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return np.min(image[y,x])
	
def get_lum(image,x,y,w,h,k,gray):
	
	if gray == 1: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	i1 = range(int(-w/2),int(w/2))
	j1 = range(0,h)
	
	lumar = np.zeros((len(i1),len(j1)))
	for i in i1:
		for j in j1:
			lum = np.min(image[y+k*h,x+i])
			lumar[i][j] = lum
	
	return np.min(lumar)

def get_ave_down(image,x,y,h,w):
	ave = np.min(image[x-w:x+w,y-h:y])
	return int(ave)
def get_ave_up(image,x,y,h,w):
	ave = np.max(image[x-w:x+w,y:y+h])
	return int(ave)

def d(landmarks,index1,index2):
#get distance between i1 and i2

	x1 = landmarks[int(index1)][0]
	y1 = landmarks[int(index1)][1]
	x2 = landmarks[int(index2)][0]
	y2 = landmarks[int(index2)][1]
	
	x_diff = (x1 - x2)**2
	y_diff = (y1 - y2)**2
	
	dist = math.sqrt(x_diff + y_diff)
	
	return dist

def q(landmarks,index1,index2):
#get angle between a i1 and i2

	x1 = landmarks[int(index1)][0]
	y1 = landmarks[int(index1)][1]
	x2 = landmarks[int(index2)][0]
	y2 = landmarks[int(index2)][1]
	
	x_diff = float(x1 - x2)
	
	if (y1 == y2): y_diff = 0.1
	if (y1 < y2): y_diff = float(np.absolute(y1 - y2))
	if (y1 > y2): 
		y_diff = 0.1
		print("Error: Facial feature located below chin.")
	
	return np.absolute(math.atan(x_diff/y_diff))

	
#image_dir should contain sub-folders containing the images where features need to be extracted
#only one face should be present in each image
#if multiple faces are detected by OpenCV, image must be manually edited; the parameters of the face-detection routine can also be changed
	
image_dir = "C:/Users/Adonis Tio/Jupyter/Google Images/celebs_extra_sorted"
cascade_path = "C:/Users/Adonis Tio/Desktop/2017-01 EEE 298 Deep Learning/Project/CV2 FaceDetect/haarcascade_frontalface_default.xml" 
predictor_path= "C:/Users/Adonis Tio/Desktop/2017-01 EEE 298 Deep Learning/Project/CV2 FaceDetect/shape_predictor_68_face_landmarks.dat"

# Create the haar cascade  
faceCascade = cv2.CascadeClassifier(cascade_path)  

# create the landmark predictor  
predictor = dlib.shape_predictor(predictor_path)  

# Read the image  
sub_dir = [q for q in pathlib.Path(image_dir).iterdir() if q.is_dir()]

start_j = 0
end_j = len(sub_dir)

features = []

for j in range(start_j, end_j):
	images_dir = [p for p in pathlib.Path(sub_dir[j]).iterdir() if p.is_file()]
	
	start_i = 0
	end_i = len(images_dir)
	
	for i in range(start_i, end_i):
		print(j, i, images_dir[i])

		image = cv2.imread(str(images_dir[i]))
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		
		# convert the image to grayscale  
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

		# Detect faces in the image; you can change the parameters if multiple faces are detected for most images; otherwie, it is easier to edit the images if only a couple have multiple face detections  
		faces = faceCascade.detectMultiScale(  
			gray,  
			scaleFactor = 1.1, #1.1  
			minNeighbors = 9,  
			minSize = (30, 30),  
			flags = cv2.CASCADE_SCALE_IMAGE  
		)  
		
		string = str(i) + " " + Path(images_dir[i]).name
		print("Found {0} faces!".format(len(faces)))  
		   
		# Draw a rectangle around the faces  
		for (x, y, w, h) in faces:  
			cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)  

			# Converting the OpenCV rectangle coordinates to Dlib rectangle  
			dlib_rect = dlib.rectangle(int(x), int(0.95*y), int(x + w), int(y + 1.05*h))  
		  
			detected_landmarks = predictor(image, dlib_rect).parts()  		  
			landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])  
				
			# copying the image so we can see side-by-side  
			if rotated == 0: image_copy = image.copy() 
			
			for idx, point in enumerate(landmarks):  
				pos = (point[0, 0], point[0, 1])
				
				# draw points on the landmark positions  
				cv2.circle(image_copy, pos, 2, color=(255, 153, 0))
			
			#find hairline, p27 is upper point of nose
			#finding the hairline is done by iterating from landmark 27 (upper point of nose bridge) and looking at a significant color difference from the initial point; avoid pictures with bangs or small color differential between skin and hair color
			
			p27 = (landmarks[27][0,0],landmarks[27][0,1]) 
			x = p27[0]
			y1 = p27[1]
			
			gray = 0
			diff = get_lum(image,x,y1,8,2,-1,gray)
			limit = diff - 55

			while (diff > limit):
				y1 = int(y1 - 1)	
				diff = get_lum(image,x,y1,6,2,-1,gray)
						  
			cv2.circle(image_copy, (x,y1), 3, color=(255, 153, 0))	  
			
			#Show annotated image			
			plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
			cv2.imwrite("agreene.jpg", image_copy)
			plt.show()
			cv2.waitKey(0)
			
			lmark = landmarks.tolist()
			p68 = ((x,y1))
			lmark.append(p68)
			
			#Extract features from the facial landmark coordinates

			f = []
			f.append(i)
			f.append(j)
					
			fwidth = d(lmark,0,16)
			fheight = d(lmark,8,68)
			f.append(fheight/fwidth)
			
			jwidth = d(lmark,4,12)
			f.append(jwidth/fwidth)
			
			hchinmouth = d(lmark,57,8)
			f.append(hchinmouth/fwidth)
			ref = q(lmark,27,8)
			
			#get angle wrt vertical
			for k in range(0,17):
				if k != 8:
					theta = q(lmark,k,8)
					f.append(theta)
					
			#get facial widths
			for k in range(1,8):
				dist = d(lmark,k,16-k)
				f.append(dist/fwidth)
			
			features.append(f)
np.savetxt("features_celebs_extra_sorted_noref.txt", features)