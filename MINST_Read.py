import struct
import numpy as np
import matplotlib as plt

filename_train = 'MINST/train-images.idx3-ubyte'
filename_train_label = 'MINST/train-labels.idx1-ubyte'
filename_test = 'MINST/t10k-images.idx3-ubyte'
filename_test_label = 'MINST/t10k-labels.idx1-ubyte'

def read_image_files(filename,num):
	with open(filename,'rb') as file_object:
		buf = file_object.read()
	index = 0
	magic,numImage,numRoses,numCols = struct.unpack_from('>IIII',buf,index)
	index += struct.calcsize('>IIII')
	image_sets = []
	
	for i in range(num):
		image = struct.unpack_from('784B',buf,index)            
		index += struct.calcsize('784B')
		image = np.array(image)
		image =  image/255.0
		image = image.tolist()
		image_sets.append(image)
	
	return image_sets

def read_label_files(filename):
	with open(filename,'rb') as file_object:
		buf = file_object.read()
	index = 0
	magic,nums = struct.unpack_from('>II',buf,index)
	index += struct.calcsize('>II')
	labels = struct.unpack_from('>%sB' % nums,buf,index)
	labels = np.array(labels)
	
	return labels

def fetch_traingset():
	images = read_image_files(filename_train,60000)
	labels = read_label_files(filename_train_label)
	return{'images':images,'labels':labels}
	
def fetch_testingset():
	images = read_image_files(filename_test,10000)
	labels = read_label_files(filename_test_label)
	return{'images':images,'labels':labels}	
	
	
	
	
	
	