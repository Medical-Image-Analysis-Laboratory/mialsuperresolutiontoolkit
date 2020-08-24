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

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d

import scipy.ndimage as snd
from skimage import morphology
from scipy.signal import argrelextrema


def extractBrain(dataPath, modelCkptLoc, thresholdLoc,modelCkptSeg,thresholdSeg, out_postfix):
    
    #Step1: Main part brain localization
    normalize = "local_max"
    width = 128
    height = 128
    border_x = 15 
    border_y = 15
    n_channels = 1

    img_nib = nibabel.load(os.path.join(dataPath))
    image_data = img_nib.get_data()
    images = np.zeros((image_data.shape[2], width, height, n_channels))
    pred3dFinal = np.zeros((image_data.shape[2], width, height, n_channels))

    slice_counter = 0
    for ii in range(image_data.shape[2]):
        img_patch = cv2.resize(image_data[:, :, ii], dsize=(width, height), fx=width,
                               fy=height)

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
    percentileLoc = thresholdLoc*100

    im = np.zeros((1, width, height, n_channels))
    pred3d = []
    with tf.Session(graph=g) as sess_test_loc:
        # Restore the model
        tf_saver = tf.train.Saver()
        tf_saver.restore(sess_test_loc, modelCkptLoc)

        for idx in range(images.shape[0]):

            im = np.reshape(images[idx, :, :, :], [1, width, height, n_channels])

            feed_dict = {x: im}
            pred_ = sess_test_loc.run(pred, feed_dict=feed_dict)

            theta = np.percentile(pred_,percentileLoc)
            pred_bin = np.where(pred_>theta,1,0)
            pred3d.append(pred_bin[0, :, :, 0].astype('float64'))

	#####

        pred3d=np.asarray(pred3d)
        heights = []
        widths = []
        coms_x = []
        coms_y= []

	#Apply PPP
        ppp = True
        if ppp:
            pred3d = post_processing(pred3d)

        pred3d = [cv2.resize(elem,dsize=(width, height),interpolation=cv2.INTER_NEAREST) for elem in pred3d]
        pred3d = np.asarray(pred3d)
        for i in range(np.asarray(pred3d).shape[0]):
            if np.sum(pred3d[i,:,:])!=0:	  
                pred3d[i,:,:] = extractLargestCC(pred3d[i,:,:].astype('uint8'))
                contours, hierarchy = cv2.findContours(pred3d[i,:,:].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                area = cv2.minAreaRect(np.squeeze(contours))
                heights.append(area[1][0])
                widths.append(area[1][1])
                bbox = cv2.boxPoints(area).astype('int')
                coms_x.append(int((np.max(bbox[:,1])+np.min(bbox[:,1]))/2))
                coms_y.append(int((np.max(bbox[:,0])+np.min(bbox[:,0]))/2))
	#Saving localization points
        med_x = int(np.median(coms_x))
        med_y = int(np.median(coms_y))
        half_max_x = int(np.max(heights)/2)
        half_max_y = int(np.max(widths)/2)
        x_beg = med_x-half_max_x-border_x
        x_end = med_x+half_max_x+border_x
        y_beg = med_y-half_max_y-border_y
        y_end = med_y+half_max_y+border_y

    #Step2: Brain segmentation
    width = 96
    height = 96

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


    subImages = np.zeros((images.shape[0], width, height))
    print(images.shape)
    for ii in range(images.shape[0]):
        subImages[ii, :, :] = cv2.resize(images[ii, x_beg:x_end, y_beg:y_end,:], dsize=(width, height))
    print(images.shape)
    with tf.Session(graph=g) as sess_test_seg:
    # Restore the model
        tf_saver = tf.train.Saver()
        tf_saver.restore(sess_test_seg, modelCkptSeg)
    
        for idx in range(images.shape[0]):
        
            im = np.reshape(subImages[idx, :, :], [1, width, height, n_channels])
        
            feed_dict = {x: im}
            pred_ = sess_test_seg.run(pred, feed_dict=feed_dict)
            percentileSeg = thresholdSeg*100
            theta = np.percentile(pred_,percentileSeg)
            pred_bin = np.where(pred_>theta,1,0)
	    #Map predictions to original indices and size

            pred_bin = cv2.resize(pred_bin[0, :, :, 0], dsize=(y_end-y_beg, x_end-x_beg), interpolation=cv2.INTER_NEAREST)

            pred3dFinal[idx, x_beg:x_end, y_beg:y_end,0] = pred_bin.astype('float64')
            
            #pred3d.append(pred_bin[0, :, :, 0].astype('float64'))
        pppp = False
        if pppp:
            pred3dFinal = post_processing(np.asarray(pred3dFinal))
        pred3d = [cv2.resize(elem, dsize=(image_data.shape[1], image_data.shape[0]), interpolation=cv2.INTER_NEAREST) for elem in pred3dFinal]
        pred3d = np.asarray(pred3d)
        upsampled = np.swapaxes(np.swapaxes(pred3d,1,2),0,2) #if Orient module applied, no need for this line(?)
        up_mask = nibabel.Nifti1Image(upsampled,img_nib.affine)
        nibabel.save(up_mask, dataPath.split('.')[0]+out_postfix)

#Funnction returning largest connected component of an object
def extractLargestCC(image):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]
    max_label = 1
    if len(sizes)<2: #in case no segmentation
        return image
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    largest_cc = np.zeros(output.shape)
    largest_cc[output == max_label] = 255
    return largest_cc.astype('uint8')

#Post-processing the binarized network output by PGD
def post_processing(pred_lbl):
    post_proc = True
    post_proc_cc = True
    post_proc_fill_holes = True

    post_proc_closing_minima = True
    post_proc_opening_maxima = True
    post_proc_extremity = False
    stackmodified = True

    crt_stack = pred_lbl.copy()
    crt_stack_pp = crt_stack.copy()

    if 1:

        distrib = []
        for iSlc in range(crt_stack.shape[0]):
            distrib.append(np.sum(crt_stack[iSlc]))

        if post_proc_cc:
            # print("post_proc_cc")
            crt_stack_cc = crt_stack.copy()
            labeled_array, num_features = snd.measurements.label(crt_stack_cc)
            unique, counts = np.unique(labeled_array, return_counts=True)

            # Try to remove false positives seen as independent connected components #2ndBrain
            for ind in range(len(unique)):
                if 5 < counts[ind] and counts[ind] < 300:
                    wherr = np.where(labeled_array == unique[ind])
                    for ii in range(len(wherr[0])):
                        crt_stack_cc[wherr[0][ii], wherr[1][ii], wherr[2][ii]] = 0

            crt_stack_pp = crt_stack_cc.copy()

        if post_proc_fill_holes:
            # print("post_proc_fill_holes")
            crt_stack_holes = crt_stack_pp.copy()

            inv_mask = 1 - crt_stack_holes
            labeled_holes, num_holes = snd.measurements.label(inv_mask)
            unique, counts = np.unique(labeled_holes, return_counts=True)

            for lbl in unique[2:]:
                trou = np.where(labeled_holes == lbl)
                for ind in range(len(trou[0])):
                    inv_mask[trou[0][ind], trou[1][ind], trou[2][ind]] = 0

            crt_stack_holes = 1 - inv_mask
            crt_stack_cc = crt_stack_holes.copy()
            crt_stack_pp = crt_stack_holes.copy()

            distrib_cc = []
            for iSlc in range(crt_stack_pp.shape[0]):
                distrib_cc.append(np.sum(crt_stack_pp[iSlc]))

        if post_proc_closing_minima or post_proc_opening_maxima:

            if 0:  # closing GLOBAL
                crt_stack_closed_minima = crt_stack_pp.copy()
                crt_stack_closed_minima = morphology.binary_closing(crt_stack_closed_minima)
                crt_stack_pp = crt_stack_closed_minima.copy()

                distrib_closed = []
                for iSlc in range(crt_stack_closed_minima.shape[0]):
                    distrib_closed.append(np.sum(crt_stack_closed_minima[iSlc]))

            if post_proc_closing_minima:
                # if 0:
                crt_stack_closed_minima = crt_stack_pp.copy()

                # for local minima
                local_minima = argrelextrema(np.asarray(distrib_cc), np.less)[0]
                local_maxima = argrelextrema(np.asarray(distrib_cc), np.greater)[0]

                for iMin in range(len(local_minima)):
                    for iMax in range(len(local_maxima) - 1):
                        # print(local_maxima[iMax], "<", local_minima[iMin], "AND", local_minima[iMin], "<", local_maxima[iMax+1], "   ???")

                        # find between which maxima is the minima localized
                        if local_maxima[iMax] < local_minima[iMin] and local_minima[iMin] < local_maxima[iMax + 1]:

                            # check if diff max-min is large enough to be considered
                            if distrib_cc[local_maxima[iMax]] - distrib_cc[local_minima[iMin]] > 50 and distrib_cc[
                                local_maxima[iMax + 1]] - distrib_cc[local_minima[iMin]] > 50:
                                sub_stack = crt_stack_closed_minima[local_maxima[iMax] - 1:local_maxima[iMax + 1] + 1,
                                            :, :]

                                # print("We did 3d close.")
                                sub_stack = morphology.binary_closing(sub_stack)
                                crt_stack_closed_minima[local_maxima[iMax] - 1:local_maxima[iMax + 1] + 1, :,
                                :] = sub_stack

                crt_stack_pp = crt_stack_closed_minima.copy()

                distrib_closed = []
                for iSlc in range(crt_stack_closed_minima.shape[0]):
                    distrib_closed.append(np.sum(crt_stack_closed_minima[iSlc]))

            if post_proc_opening_maxima:
                crt_stack_opened_maxima = crt_stack_pp.copy()

                local = True
                if local:
                    local_maxima_n = argrelextrema(np.asarray(distrib_closed), np.greater)[
                        0]  # default is mode='clip'. Doesn't consider extremity as being an extrema

                    for iMax in range(len(local_maxima_n)):

                        # Check if this local maxima is a "peak"
                        if distrib[local_maxima_n[iMax]] - distrib[local_maxima_n[iMax] - 1] > 50 and distrib[
                            local_maxima_n[iMax]] - distrib[local_maxima_n[iMax] + 1] > 50:

                            if 0:
                                print("Ceci est un pic de au moins 50.", distrib[local_maxima_n[iMax]], "en",
                                      local_maxima_n[iMax])
                                print("                                bordé de", distrib[local_maxima_n[iMax] - 1],
                                      "en", local_maxima_n[iMax] - 1)
                                print("                                et", distrib[local_maxima_n[iMax] + 1], "en",
                                      local_maxima_n[iMax] + 1)
                                print("")

                            sub_stack = crt_stack_opened_maxima[local_maxima_n[iMax] - 1:local_maxima_n[iMax] + 2, :, :]
                            sub_stack = morphology.binary_opening(sub_stack)
                            crt_stack_opened_maxima[local_maxima_n[iMax] - 1:local_maxima_n[iMax] + 2, :, :] = sub_stack
                else:
                    crt_stack_opened_maxima = morphology.binary_opening(crt_stack_opened_maxima)

                crt_stack_pp = crt_stack_opened_maxima.copy()

                distrib_opened = []
                for iSlc in range(crt_stack_pp.shape[0]):
                    distrib_opened.append(np.sum(crt_stack_pp[iSlc]))

            if post_proc_extremity:

                crt_stack_extremity = crt_stack_pp.copy()

                # check si y a un maxima sur une extremite
                maxima_extrema = argrelextrema(np.asarray(distrib_closed), np.greater, mode='wrap')[0]
                # print("maxima_extrema", maxima_extrema, "     numslices",numslices, "     numslices-1",numslices-1)

                if distrib_opened[0] - distrib_opened[1] > 40:
                    # print("First slice of ", distrib_opened, " is a maxima")
                    sub_stack = crt_stack_extremity[0:2, :, :]
                    sub_stack = morphology.binary_opening(sub_stack)
                    crt_stack_extremity[0:2, :, :] = sub_stack
                    # print("On voulait close 1st slices",  sub_stack.shape[0])

                if pred_lbl.shape[0] - 1 in maxima_extrema:
                    # print(numslices-1, "in maxima_extrema", maxima_extrema )

                    sub_stack = crt_stack_opened_maxima[-2:, :, :]
                    sub_stack = morphology.binary_opening(sub_stack)
                    crt_stack_opened_maxima[-2:, :, :] = sub_stack

                    # print("On voulait close last slices",  sub_stack.shape[0])

                crt_stack_pp = crt_stack_extremity.copy()

                distrib_opened_border = []
                for iSlc in range(crt_stack_pp.shape[0]):
                    distrib_opened_border.append(np.sum(crt_stack_pp[iSlc]))

    return crt_stack_pp


def get_parser():
	import argparse

	parser = argparse.ArgumentParser(description='Brain extraction based on U-Net convnet')
	parser.add_argument('-i','--input', required=True, action='append', help='Input image(s)')
	parser.add_argument('-c','--checkpoint_loc', required=True, action='append', help='Network checkpoint localization')
	parser.add_argument('-t','--threshold_loc', required=True, action='append', help='Threshold localization')
	parser.add_argument('-C','--checkpoint_seg', required=True, action='append', help='Network checkpoint segmentation')
	parser.add_argument('-T','--threshold_seg', required=True, action='append', help='Threshold segmentation')
	parser.add_argument('-o','--out_postfix', required=True, action='append', help='Suffix to masked images')
	
	return parser


if __name__ == "__main__":

	# Parse command line args

	parser = get_parser()
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
	extractBrain(args.input[0],args.checkpoint_loc[0],float(args.threshold_loc[0]),args.checkpoint_seg[0],float(args.threshold_seg[0]),args.out_postfix[0]) #I had to add [0] index to extract string from list
