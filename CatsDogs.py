"""Remember To uncomment line 116 and 117 for the first time because your model will train and store the trained model 
    and when your model is trained comment or delete out 116 and 117 """
#--------Dependencies--------
#TensorFlow
#Tflearn
#OpenCV
#tqdm(optional)
#numpy

import os
import numpy as np
import tensorflow as tf
import cv2
from random import shuffle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tqdm


traindr='E:\\tutorial\\Machine Learning\\cats vs dogs\\train' #Change the directory to wherever you download the data
testdr='E:\\tutorial\\Machine Learning\\cats vs dogs\\test'  #Ditto

imsize=50 #Size of the image
lr=1e-3  #Learning Rate 

model_name='catsvsdogs-{}-{}.model'.format(lr,'Xconvb') #your model name 

def label_img(img):  #Preprocessing to get the labels out of the name of the image 
	wrd=img.split('.')[0]
	if wrd=='cat':	return [0,1]
	elif wrd=='dog':	return [1,0]
	
def create_train():  #Creating the training data
	traindata=[]
	for img in os.listdir(traindr):  #Iterating through our training data .note that the name of the image will be iterated and not the image
		label=label_img(img)
		path=os.path.join(traindr,img)
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(imsize,imsize)) #Resizing the image
		traindata.append([np.array(img),label])
	shuffle(traindata)
	return traindata

def test_data():
	traindata=[]
	for img in os.listdir(testdr):
		label=label_img(img)
		path=os.path.join(testdr,img)
		imgnum=img.split('.')[0]
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(imsize,imsize))
		
		traindata.append([np.array(img),imgnum])
	
	return traindata
	
traindata=create_train()







# Building convolutional convnet
convnet = input_data(shape=[None, imsize, imsize, 1], name='input')
# http://tflearn.org/layers/conv/
# http://tflearn.org/activations/
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)

if os.path.exists('{}.meta'.format(model_name)):
	model.load(model_name)
	

traind=traindata[:-500]
testd=traindata[-500:]

X=np.array([i[0] for i in traind]).reshape(-1,50,50,1)

Y=[i[1]for i in traind]

test_x=np.array([i[0] for i in testd]).reshape(-1,50,50,1)

test_y=[i[1] for i in testd]




#model.fit({'input': X}, {'targets': Y}, n_epoch=6, validation_set=({'input': test_x}, {'targets': test_y}), 
   # snapshot_step=500, show_metric=True)

testdata=test_data()
with open('submit.csv','w') as f:
	f.write('id,label\n')
with open('submit.csv','w') as f:
	for data in (testdata):
		out=model.predict([data[0].reshape(50,50,1)])[0]
		f.write('{},{}\n'.format(data[1],out[0]))
	
	
	
	

		
