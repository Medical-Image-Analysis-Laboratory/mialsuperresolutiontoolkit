#!/usr/bin/env python

import argparse, getopt, sys

#import scipy.interpolate

import numpy as np
#from matplotlib import pyplot
import glob
import nibabel as nib
import scipy.ndimage

#from scipy.ndimage.filters import percentile_nonzero_filter

import sys
import os
import re
import fnmatch
import ntpath
import copy

import pdb

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def percentile_nonzero(image, percentile_nonzero):
        #pdb.set_trace()
    if len(image) < 1:
        value = None
    elif (percentile_nonzero >= 100):
        sys.stderr.write('ERROR: percentile_nonzero must be < 100.  you supplied: %s\n'% percentile_nonzero)
        value = None
    else:
        image_nonzero=image[image!=0]
        element_idx = int(len(image_nonzero) * (percentile_nonzero / 100.0))
        image_nonzero.sort()
        value = image_nonzero[element_idx]
    return value

def mean_nonzero(image):
    image_nonzero=image[image!=0]
    mean=np.sum(image_nonzero)/len(image_nonzero)
    return mean

def intensityNormalization(image,landmarks):
    print 'min ='+str(landmarks['p1'])
    print 'max (99.8%) ='+str(landmarks['p2'])
        #print 'mean ='+str(landmarks['mean'])
    print 'quartiles [25%,50%,75%] ='+str(landmarks['quartiles'])
    return 1

def displayHistogram(image,image_name,loffset,roffset):
    bins = np.round(np.arange(loffset,np.max(image)-roffset,40))
    histo, bins = np.histogram(image, bins=bins) 
    bins_center = 0.5*(bins[1:] + bins[:-1])
    #pyplot.plot(bins_center,histo,alpha=0.5)
    ##pyplot.hist(fit(np.random.uniform(x[0],x[-1],len(image))),bins=y)
    ##pyplot.hist(image,bins,histtype='step',alpha=0.5,label=image_name)
    return 1

def extractImageLandmarks(image):
    landmarks={}
    landmarks['p1']=percentile_nonzero(image,0)
    landmarks['p2']=percentile_nonzero(image,99.8)
    #landmarks['mean']=mean_nonzero(image)
    landmarks['quartiles']=[percentile_nonzero(image,25),percentile_nonzero(image,50),percentile_nonzero(image,75)]
    #landmarks['quartiles']=[percentile_nonzero(image,10),percentile_nonzero(image,20),percentile_nonzero(image,30),percentile_nonzero(image,40),percentile_nonzero(image,50),percentile_nonzero(image,60),percentile_nonzero(image,70),percentile_nonzero(image,80),percentile_nonzero(image,90)]
    #pdb.set_trace()
    return landmarks

def trainImageLandmarks(list_landmarks):
    mup_l=[]
    mup_L=[]
    mup_r=[]
    mup_R=[]
    maxLR=[]
    index=0
    while index<len(list_landmarks):
        landmarks=list_landmarks[index]['quartiles']
        mup_l.append(np.min(landmarks-list_landmarks[index]['p1']))
        mup_L.append(np.max(landmarks-list_landmarks[index]['p1']))
        mup_r.append(np.min(list_landmarks[index]['p2']-landmarks))
        mup_R.append(np.max(list_landmarks[index]['p2']-landmarks))
        maxLR.append(np.max([float(mup_L[index])/mup_l[index],float(mup_R[index])/mup_r[index]]))
        #print 'mup_l  =  '+str(mup_l[index])
        #print 'mup_L  =  '+str(mup_L[index])
        #print 'mup_r  =  '+str(mup_r[index])
        #print 'mup_R  =  '+str(mup_R[index])
        #print 'maxLR  =  '+str(maxLR[index])
        index+=1
    ymax=np.max(maxLR)
    ymax_index=maxLR.index(max(maxLR))
    dS = float(ymax*(mup_L[ymax_index]+mup_R[ymax_index]))
    print 'Ymax  =  '+str(ymax)+'  at position '+str(ymax_index)+'  ,  dS = '+str(dS)+' (=s2 when s1=0)'
    return list_landmarks,dS

def mapImageLandmarks(list_landmarks,s1,s2):
    list_landmarks_mapped = copy.deepcopy(list_landmarks)
    index=0
    while index<len(list_landmarks):
        land_index=0
        print 'Image index: '+str(index)
        while land_index<len(list_landmarks[index]['quartiles']):
            print 'old landmark: '+str(list_landmarks_mapped[index]['quartiles'][land_index])
            list_landmarks_mapped[index]['quartiles'][land_index]=s1+float((list_landmarks_mapped[index]['quartiles'][land_index]-list_landmarks_mapped[index]['p1'])/float(list_landmarks_mapped[index]['p2']-list_landmarks_mapped[index]['p1']))*float((s2-s1))
            print 'new landmark: '+str(list_landmarks_mapped[index]['quartiles'][land_index])
            land_index+=1
        print 'p1, p2 = '+str(list_landmarks_mapped[index]['p1'])+', '+str(list_landmarks_mapped[index]['p2'])
        index+=1
    return list_landmarks_mapped

def verifyOne2OneMapping(s1,s2,list_landmarks,lmap_mean):
    landmarks=list_landmarks['quartiles']
    mup_L=np.max(landmarks-list_landmarks['p1'])
    mup_R=np.max(list_landmarks['p2']-landmarks)
    #print 'mup_L  =  '+str(mup_L)
    #print 'mup_R  =  '+str(mup_R)

    land_index=0
    while land_index<len(lmap_mean):
        #pdb.set_trace()
        if np.logical_and((lmap_mean[str(0)]-s1)>=mup_L,(s2-lmap_mean[str(len(lmap_mean)-1)])>=mup_R):
            cond=1;
        else:
            cond=0;
        land_index+=1
    return cond

def mapImage(image,lmap_mean,list_landmarks,s1,s2,p1,p2):
    image_out=image.copy().astype('float')
    tmp=image.copy().astype('float')
    index=0
    ##pyplot.figure(2)
    #pdb.set_trace()
    while index < len(lmap_mean)+1:
        if index ==0:
            x=np.array([int(p1),int(list_landmarks[index])])
            y=np.array([int(s1),int(lmap_mean[str(index)])])
            coefs=np.polyfit(x,y,1)
            mask=np.logical_and(image > 0, image <= list_landmarks[index])
            image_out[mask]=coefs[0]*image[mask]+coefs[1]
            ##pyplot.plot(x,y,marker='o', linestyle='--');
        else:
            if index==(len(lmap_mean)):
                x=np.array([int(list_landmarks[index-1]),int(p2)])
                y=np.array([int(lmap_mean[str(index-1)]),int(s2)])
                coefs=np.polyfit(x,y,1)
                mask=image>list_landmarks[index-1]
                image_out[mask]=coefs[0]*image[mask]+coefs[1]
                ##pyplot.plot(x,y,marker='o', linestyle='--');
            else:
                x=np.array([int(list_landmarks[index-1]),int(list_landmarks[index])])
                y=np.array([int(lmap_mean[str(index-1)]),int(lmap_mean[str(index)])])
                coefs=np.polyfit(x,y,1)
                mask=np.logical_and(image>list_landmarks[index-1], image<=list_landmarks[index])
                image_out[mask]=coefs[0]*image[mask]+coefs[1]
                ##pyplot.plot(x,y,marker='o', linestyle='--');
        index+=1
    ##pyplot.show()
    return image_out

def computeMeanMapImageLandmarks(list_landmarks):
    mean_landmarks={}
    index=0
    while index < len(list_landmarks):
        land_index=0
        while land_index < len(list_landmarks[index]['quartiles']):
            if(index==0):
                mean_landmarks[str(land_index)] = list_landmarks[index]['quartiles'][land_index];
            else:
                mean_landmarks[str(land_index)]+= list_landmarks[index]['quartiles'][land_index];
            land_index+=1
        index+=1

    land_index=0
    while land_index < len(mean_landmarks):
        mean_landmarks[str(land_index)] = mean_landmarks[str(land_index)] / len(list_landmarks)
        land_index+=1

    print 'Final landmark average : '
    print mean_landmarks
    return mean_landmarks


def main(images,masks,outputs):
    
    image_paths= sorted(images)
    mask_paths=sorted(masks)
    output_paths = sorted(outputs)

    if(len(image_paths)!=len(mask_paths)):
        print 'Loading failed: Number of images and masks are different (# images = '+str(len(image_paths))+' \ # masks = '+str(len(mask_paths))+')'
        return
    else:
        print 'Loading passed: Number of images and masks are equal (# images = '+str(len(image_paths))+' \ # masks = '+str(len(mask_paths))+')'
    
    list_landmarks=[]
   
    s1=1
    #pyplot.figure(1)
    #pyplot.subplot(211)
    index=0
    while index<len(image_paths):
        image_name = image_paths[index].split("/")[-1].split(".")[0]
        print 'Process image '+image_name
        image = nib.load(image_paths[index]).get_data()
        #image = scipy.ndimage.filters.gaussian_filter(image,1.0)
        mask = nib.load(mask_paths[index]).get_data()
        maskedImage = np.reshape(image*mask,image.shape[0]*image.shape[1]*image.shape[2])
        displayHistogram(maskedImage,image_name,1,0)
        list_landmarks.append(extractImageLandmarks(maskedImage))
        index+=1

    #pyplot.legend()
    #pyplot.xlabel('Intensity')
    #pyplot.ylabel('# of voxels')
    #pyplot.title('Histograms: Overview before normalization')
    list_landmarks,dS = trainImageLandmarks(list_landmarks)

    s2 = np.ceil(dS - s1)
    print 'Standard scale estimated: ['+str(s1)+','+str(s2)+']'

    list_landmarks_mapped = mapImageLandmarks(list_landmarks,s1,s2)

    mean_landmarks = computeMeanMapImageLandmarks(list_landmarks_mapped)

    index = 0
    #pyplot.figure(1)
    #pyplot.subplot(212)
    while index<len(image_paths):
        image_name = image_paths[index].split("/")[-1].split(".")[0]
        print 'Map image '+image_name
        image = nib.load(image_paths[index])
        image_data = image.get_data()
        mask_data = nib.load(mask_paths[index]).get_data()
        dimY=image.shape[0]
        dimX=image.shape[1]
        dimZ=image.shape[2]
        #maskedImage = np.reshape(image_data*mask_data,dimX*dimY*dimZ)
        maskedImage = np.reshape(image_data,dimX*dimY*dimZ)
        #pdb.set_trace()
        maskedImageMapped = mapImage(maskedImage,mean_landmarks,list_landmarks[index]['quartiles'],s1,s2,list_landmarks[index]['p1'],list_landmarks[index]['p2'])
        displayHistogram(maskedImageMapped,image_name,1,0)
        o2o=verifyOne2OneMapping(s1,s2,list_landmarks[index],mean_landmarks)
        new_image = nib.Nifti1Image(np.reshape(maskedImageMapped,np.array([dimY,dimX,dimZ])),image.get_affine(),header=image.get_header())
        print 'Save normalized image '+str(image_name)+ ' as '+str(output_paths[index]) + '(one 2 one mapping :'+str(o2o)+')'
        nib.save(new_image,output_paths[index])
        index+=1
    #pyplot.legend()
    #pyplot.xlabel('Intensity')
    #pyplot.ylabel('# of voxels')
    #pyplot.title('Histograms: Overview after normalization')
    ## To be uncommented to display plot before/after histogram normalizatiopn
    ##pyplot.show()

# Parse command line args

parser = argparse.ArgumentParser(description='Intensity histogram normalization based on percentiles')
parser.add_argument('-i','--input', required=True, action='append', help='Input image(s)')
parser.add_argument('-m','--mask', required=True, action='append', help='Input mask(s)')
parser.add_argument('-o','--output', required=True, action='append', help='Output normalized image(s)')

args = parser.parse_args()

print(len(args.input)>0)
print(len(args.mask)>0)
print(len(args.output)>0)
print('Inputs: {}'.format(args.input))
print('Masks: {}'.format(args.mask))
print('Outputs: {}'.format(args.output))

if len(args.input)==0:
    print("Error: No input images provided")
    print("Usage: %s -i input_image1 -m input_image1_mask -o output_image1 -i input_image2 -m input_image2_mask -o output_image2 " % sys.argv[0])
    sys.exit(2)

if len(args.mask)==0:
    print("Error: No masks provided")
    print("Usage: %s -i input_image1 -m input_image1_mask -o output_image1 -i input_image2 -m input_image2_mask -o output_image2 " % sys.argv[0])
    sys.exit(2)

if len(args.output)==0:
    print("Error: No output provided")
    print("Usage: %s -i input_image1 -m input_image1_mask -o output_image1 -i input_image2 -m input_image2_mask -o output_image2 " % sys.argv[0])
    sys.exit(2)

if (len(args.input)!=len(args.mask)):
    print("Error: Number of inputs and masks are not equal")
    print("Usage: %s -i input_image1 -m input_image1_mask -o output_image1 -i input_image2 -m input_image2_mask -o output_image2 " % sys.argv[0])
    sys.exit(2)

if (len(args.input)!=len(args.output)):
    print("Error: Number of inputs and outputs are not equal")
    print("Usage: %s -i input_image1 -m input_image1_mask -o output_image1 -i input_image2 -m input_image2_mask -o output_image2 " % sys.argv[0])
    sys.exit(2)


print('Inputs: {}'.format(args.input))
print('Masks: {}'.format(args.mask))
print('Outputs: {}'.format(args.output))
main(args.input,args.mask,args.output)
