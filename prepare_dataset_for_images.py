import numpy as np
from scipy import ndimage
import scipy.misc
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
from one_hot_matrix import *
import cv2
import csv
def prepare_dataset_for_images(No_of_image,image_size,output,folder_path):
	print(folder_path)
	X = np.zeros((No_of_image*output, image_size*image_size*1))#6632
	Y = np.zeros((No_of_image*output, 1))
	np.random.seed(0)
	names=["zero","one","two", "three", "four", "five", "six","seven", "eight", "nine", "sum", "subtract", "product", "divide"]
	index=0
	for i in range(0,output):
		for j in range(1,(No_of_image+1)):#1image_size9
				path = folder_path+names[i]+"/"+names[i].lower()+str(j)+".jpg"
				image=np.array(ndimage.imread(path,flatten=False))
				image=scipy.misc.imresize(image,size=(image_size,image_size)).reshape(1,image_size*image_size*1)
				X[index, :] = image
				Y[index] = i
				index=index+1
				print(index,image.shape)
			
	Y=one_hot_matrix(Y,output)
	with tf.Session() as sess:
		Y=sess.run(Y)
		sess.close()
	
	Y=np.reshape(Y,(Y.shape[0],output))
	permutation=list(np.random.permutation(X.shape[0]))
	X_shuffled=X[permutation,:]
	Y_shuffled=Y[permutation,:]
	X_shuffled=np.reshape(X_shuffled,(X_shuffled.shape[0],image_size,image_size,1))

	return X_shuffled,Y_shuffled

		
