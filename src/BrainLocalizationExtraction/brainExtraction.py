#!/usr/bin/env python

import os
import nibabel
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from IPython import display
import pylab as pl

import sys
from medpy.io import load
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import image
from scipy import ndimage
import math
from inspect import signature
import argparse, getopt

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d

def extractBrain(dataPath, modelCkpt, threshold,out_postfix):
    normalize = "local_max"
    width = 128
    height = 128
    n_channels = 1

    img_nib = nibabel.load(os.path.join(dataPath))
    image_data = img_nib.get_data()
    images = np.zeros((image_data.shape[2], width, height, n_channels))
    
    slice_counter = 0
    for ii in range(image_data.shape[2]):
        img_patch = cv2.resize(image_data[:, :, ii], dsize=(width, height), fx=width,
                               fy=height)  # , interpolation=cv2.INTER_CUBIC)

        if normalize:
            if normalize == "local_max":
                 images[slice_counter, :, :, 0] = img_patch / np.max(img_patch)
            elif normalize == "global_max":
                 images[slice_counter, :, :, 0] = img_patch / max_val
            elif normalize ==  "mean_std":
                 images[slice_counter, :, :, 0] = (img_patch-np.mean(img_patch))/np.std(img_patch)
            else:
                 raise ValueError('Please select a valid normalization')
        else:
            images[slice_counter, :, :, 0] = img_patch

        slice_counter += 1

    #Tensorflow graph

    g = tf.Graph()
    with g.as_default():

        with tf.name_scope('inputs'):

        x = tf.placeholder(tf.float32, [None, width, height, n_channels])        

        conv1 = conv_2d(x, 32, 3, activation='relu', padding='same', regularizer="L2")
        conv1 = conv_2d(conv1, 32, 3, activation='relu', padding='same', regularizer="L2")
        pool1 = max_pool_2d(conv1, 2)

        conv2 = conv_2d(pool1, 64, 3, activation='relu', padding='same', regularizer="L2")
        conv2 = conv_2d(conv2, 64, 3, activation='relu', padding='same', regularizer="L2")
        pool2 = max_pool_2d(conv2, 2)

        conv3 = conv_2d(pool2, 128, 3, activation='relu', padding='same', regularizer="L2")
        conv3 = conv_2d(conv3, 128, 3, activation='relu', padding='same', regularizer="L2")
        pool3 = max_pool_2d(conv3, 2)

        conv4 = conv_2d(pool3, 256, 3, activation='relu', padding='same', regularizer="L2")
        conv4 = conv_2d(conv4, 256, 3, activation='relu', padding='same', regularizer="L2")
        pool4 = max_pool_2d(conv4, 2)

        conv5 = conv_2d(pool4, 512, 3, activation='relu', padding='same', regularizer="L2")
        conv5 = conv_2d(conv5, 512, 3, activation='relu', padding='same', regularizer="L2")

        up6 = upsample_2d(conv5,2)
        up6 = tflearn.layers.merge_ops.merge([up6, conv4], 'concat',axis=3)
        conv6 = conv_2d(up6, 256, 3, activation='relu', padding='same', regularizer="L2")
        conv6 = conv_2d(conv6, 256, 3, activation='relu', padding='same', regularizer="L2")

        up7 = upsample_2d(conv6,2)
        up7 = tflearn.layers.merge_ops.merge([up7, conv3],'concat', axis=3)
        conv7 = conv_2d(up7, 128, 3, activation='relu', padding='same', regularizer="L2")
        conv7 = conv_2d(conv7, 128, 3, activation='relu', padding='same', regularizer="L2")

        up8 = upsample_2d(conv7,2)
        up8 = tflearn.layers.merge_ops.merge([up8, conv2],'concat', axis=3)
        conv8 = conv_2d(up8, 64, 3, activation='relu', padding='same', regularizer="L2")
        conv8 = conv_2d(conv8, 64, 3, activation='relu', padding='same', regularizer="L2")

        up9 = upsample_2d(conv8,2)
        up9 = tflearn.layers.merge_ops.merge([up9, conv1],'concat', axis=3)
        conv9 = conv_2d(up9, 32, 3, activation='relu', padding='same', regularizer="L2")
        conv9 = conv_2d(conv9, 32, 3, activation='relu', padding='same', regularizer="L2")

        pred = conv_2d(conv9, 2, 1,  activation='linear', padding='valid')


    #Thresholding parameter to binarize predictions
    percentile = threshold*100

    im = np.zeros((1, width, height, n_channels))
    pred3d = []
    with tf.Session(graph=g) as sess_test:
        # Restore the model
        tf_saver = tf.train.Saver()
        tf_saver.restore(sess_test, modelCkpt)

        for idx in range(images.shape[0]):

            im = np.reshape(images[idx, :, :, :], [1, width, height, n_channels])

            feed_dict = {x: im}
            pred_ = sess_test.run(pred, feed_dict=feed_dict)

            theta = np.percentile(pred_,percentile)
            pred_bin=np.where(pred_>theta,1,0)
            pred3d.append(cv2.resize(pred_bin[0,:,:,0].astype('float64'),dsize=(image_data.shape[0], image_data.shape[1]),fx=1/width,fy=1/height,interpolation=cv2.INTER_NEAREST))
            upsampled = np.swapaxes(np.swapaxes(pred3d,0,2),0,1) #if Orient module applied, no need for this line
            up_mask = nibabel.Nifti1Image(upsampled,img_nib.affine)
            nibabel.save(up_mask, dataPath[:-4]+out_postfix)

if __name__ == "__main__":

	# Parse command line args

	parser = argparse.ArgumentParser(description='Brain extraction based on U-Net convnet')
	parser.add_argument('-i','--input', required=True, action='append', help='Input image(s)')
	parser.add_argument('-c','--checkpoint', required=True, action='append', help='Network checkpoint')
	parser.add_argument('-t','--threshold', required=True, action='append', help='Threshold')
	parser.add_argument('-o','--out_postfix', required=True, action='append', help='Suffix to masked images')

	args = parser.parse_args()

	# print(len(args.input)>0)
	# print(len(args.mask)>0)
	# print(len(args.output)>0)
	# print('Inputs: {}'.format(args.input))
	# print('Masks: {}'.format(args.mask))
	# print('Outputs: {}'.format(args.output))
    #
	# if len(args.input)==0:
	#     print("Error: No input images provided")
	#     print("Usage: %s -i input_image1 -m input_image1_mask -o output_image1 -i input_image2 -m input_image2_mask -o output_image2 " % sys.argv[0])
	#     sys.exit(2)
    #
	# if len(args.mask)==0:
	#     print("Error: No masks provided")
	#     print("Usage: %s -i input_image1 -m input_image1_mask -o output_image1 -i input_image2 -m input_image2_mask -o output_image2 " % sys.argv[0])
	#     sys.exit(2)
    #
	# if len(args.output)==0:
	#     print("Error: No output provided")
	#     print("Usage: %s -i input_image1 -m input_image1_mask -o output_image1 -i input_image2 -m input_image2_mask -o output_image2 " % sys.argv[0])
	#     sys.exit(2)
    #
	# if (len(args.input)!=len(args.mask)):
	#     print("Error: Number of inputs and masks are not equal")
	#     print("Usage: %s -i input_image1 -m input_image1_mask -o output_image1 -i input_image2 -m input_image2_mask -o output_image2 " % sys.argv[0])
	#     sys.exit(2)
    #
	# if (len(args.input)!=len(args.output)):
	#     print("Error: Number of inputs and outputs are not equal")
	#     print("Usage: %s -i input_image1 -m input_image1_mask -o output_image1 -i input_image2 -m input_image2_mask -o output_image2 " % sys.argv[0])
	#     sys.exit(2)
    #
    #
	# print('Inputs: {}'.format(args.input))
	# print('Masks: {}'.format(args.mask))
	# print('Outputs: {}'.format(args.output))
	extractBrain(args.input[0],args.checkpoint[0],float(args.threshold[0]),args.out_postfix[0]) #I had to add [0] index to extract string from list
