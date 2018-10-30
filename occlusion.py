from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import math
import pylab
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from numpy.random import permutation
from keras.optimizers import SGD



base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224, 3))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(6, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights('./inet.h5')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def Occlusion_exp(image_path, occluding_size, occluding_pixel, occluding_stride):
	image = cv2.imread(image_path)
	im = cv2.resize(image, (224, 224)).astype('uint8')
	print im.shape
	print im.shape
	im = np.expand_dims(im, axis=0)
	print im.shape
	out = model.predict(im)
	out = out[0]
	# Getting the index of the winning class:
	m = np.argmax(out)
	height, width, _ = image.shape
	print height
	print width
	output_height = int(math.ceil((height-occluding_size) / occluding_stride + 1))
	output_width = int(math.ceil((width-occluding_size) / occluding_stride + 1))
	heatmap = np.zeros((output_height, output_width))

	for h in xrange(output_height):
		for w in xrange(output_width):
			# Occluder region:
			h_start = h * occluding_stride
			w_start = w * occluding_stride
			h_end = min(height, h_start + occluding_size)
			w_end = min(width, w_start + occluding_size)
			# Getting the image copy, applying the occluding window and classifying it again:
			input_image = copy.copy(image)
			input_image[h_start:h_end, w_start:w_end,:] =  occluding_pixel            
			im = cv2.resize(input_image, (224, 224)).astype('uint8')
			# cv2.imshow('image', im)
			# cv2.waitKey(0)
			im = np.expand_dims(im, axis=0)
			out = model.predict(im)
			out = out[0]
			print('scanning position (%s, %s)'%(h,w))
			# It's possible to evaluate the VGG-16 sensitivity to a specific object.
			# To do so, you have to change the variable "index_object" by the index of
			# the class of interest. The VGG-16 output indices can be found here:
			# https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt
			prob = (out[m])
			print prob 
			heatmap[h,w] = prob

	# f = pylab.figure()
	# f.add_subplot(1, 2, 0)  # this line outputs images side-by-side    
	plt.imshow(heatmap, cmap='hot', interpolation='nearest')
	plt.colorbar()
	# f.add_subplot(1, 2, 1)
	# plt.imshow(image)
	plt.savefig('heatmap1.png')
	print ( 'Object index is %s'%m)


Occlusion_exp('./Sadguru_Jaggi_Vasudev.jpg', 30, 0, 1)