"""IT IS REALLY IMPORTANT TO HAVE OPENCV PACKAGE INSTALLED  IN YOUR PYTHON DIRECTORY TO RUN THIS
    JUST PUT THE IMAGE IN THE SAME FOLDER AS THIS FILE OR GIVE THE PATH IN LINE NUMBER 15"""


import numpy as np
import cv2


# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


img=cv2.imread('cc.png') #GIVE THE PATH OF YOUR PICTURE HERE

while 1:
    
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
        
		
	cv2.imshow('img',img)
	cv2.waitKey()
	break
    

cv2.destroyAllWindows()
