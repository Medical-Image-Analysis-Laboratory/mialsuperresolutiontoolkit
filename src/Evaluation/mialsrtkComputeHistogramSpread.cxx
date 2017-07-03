/*=========================================================================

Program: Computes Spread of Image Intensity Histogram
Language: C++
Date: $Date$
Version: 1.0
Author: Sebastien Tourbier

Copyright (c) 2017 Medical Image Analysis Laboratory (MIAL), Lausanne
  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

=========================================================================*/
/* Standard includes */
#include <tclap/CmdLine.h>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImage.h"

#include "itkChangeInformationImageFilter.h"

#include "itkStatisticsImageFilter.h"
#include <itkLabelOverlapMeasuresImageFilter.h>

#include "itkMultiThreader.h"

#include "vcl_algorithm.h"

#include "mialsrtkMaths.h"
#include "itkImageRegionConstIteratorWithIndex.h"




int main(int argc, char *argv[])
{

    const char *inputFileName1 = NULL;
    const char *maskFileName = NULL;
    const char *csvFileName = NULL;
    const char *subjName = NULL;
    const char *methodName = NULL;
    const char *gmmBeta = NULL;
    const char *iteration = NULL;

    // Parse arguments

    TCLAP::CmdLine cmd("Computes the Histogram spread of an Image", ' ', "Unversioned");

    TCLAP::ValueArg<std::string> input1Arg  ("i","input","Input image",true,"","string",cmd);
    TCLAP::ValueArg<std::string> maskArg  ("m","mask","Input mask",true,"undefined","string",cmd);
    TCLAP::ValueArg<std::string> csvArg  ("c","csv","CSV file",true,"undefined","string",cmd);
    TCLAP::ValueArg<std::string> subjArg  ("s","subject-name","Name of subject",true,"undefined","string",cmd);
    TCLAP::ValueArg<std::string> methArg  ("t","method-type","Type of method",true,"undefined","string",cmd);
    TCLAP::ValueArg<std::string> betaArg  ("b","beta","Contribution of GMM",true,"undefined","string",cmd);
    TCLAP::ValueArg<std::string> iterArg  ("","it","Iteration",true,"undefined","string",cmd);


    // Parse the argv array.
    cmd.parse( argc, argv );

    inputFileName1 = input1Arg.getValue().c_str();
    maskFileName  = maskArg.getValue().c_str();

    csvFileName  = csvArg.getValue().c_str();
    subjName = subjArg.getValue().c_str();
    methodName = methArg.getValue().c_str();
    gmmBeta = betaArg.getValue().c_str();
    iteration = iterArg.getValue().c_str();


    const unsigned int dimension = 3;

    typedef  float PixelType;

    typedef itk::Image<PixelType, dimension> ImageType;
    typedef itk::ImageFileReader<ImageType> ReaderType;

    typedef itk::ImageRegionConstIteratorWithIndex<ImageType> IteratorType;

    ImageType::Pointer image = ImageType::New();
    ImageType::Pointer mask = ImageType::New();

    //Read the images
    //
    //
    std::cout << "Load input  image: " << inputFileName1 << std::endl;;
    ReaderType::Pointer imReader = ReaderType::New();
    imReader->SetFileName(inputFileName1);
    try
    {
        imReader->Update();
        image = imReader->GetOutput();
        image->Update();

    }
    catch( itk::ExceptionObject & err )
    {
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Load input  mask: " << maskFileName << std::endl;;
    ReaderType::Pointer maskReader = ReaderType::New();
    maskReader->SetFileName(maskFileName);
    try
    {
        maskReader->Update();
        mask = imReader->GetOutput();
        mask->Update();
    }
    catch( itk::ExceptionObject & err )
    {
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }

    IteratorType itImage(image,image->GetLargestPossibleRegion());
    IteratorType itMask(mask,mask->GetLargestPossibleRegion());

    int voxBrainNumber = 0;
    for( itMask.GoToBegin(); !itMask.IsAtEnd(); ++itMask)
    {
        if(itMask.Get()>0.0) voxBrainNumber++;
    }

    std::cout << "Number of voxels in the mask : " << voxBrainNumber << std::endl;


    typedef vnl_vector<float>::iterator VnlIterator;

    int nbins = 256;
    vnl_vector<float> histogram;
    histogram.set_size(nbins);
    histogram.fill(0.0);

    vnl_vector<float> x_orig;
    x_orig.set_size(voxBrainNumber);
    x_orig.fill(0.0);

    int index = 0;
    for( itMask.GoToBegin(),  itImage.GoToBegin() ; !itMask.IsAtEnd(); ++itMask, ++itImage)
    {
        if(itMask.Get()>0.0)
        {
            x_orig[index]=itImage.Get();
            index++;
        }
    }

    vnl_vector<float>  x = ((x_orig - x_orig.min_value())/x_orig.max_value()) * (nbins-1.0);

    VnlIterator itX;
    int count = 0;
    for(itX = x.begin(); itX != x.end(); ++itX)
    {
        if(*itX>0.0)
        {
            histogram[(int)floor(*itX)]++;
            count++;
        }
    }

    vnl_vector<float> PDF = histogram / count;

    vnl_vector<float> CDF;
    CDF.set_size(nbins);
    CDF.fill(0.0);

    VnlIterator itPDF;

    index = 0;
    for(itPDF = PDF.begin(); itPDF != PDF.end(); ++itPDF)
    {
        if(index == 0)
        {
            CDF[index] = *itPDF;
        }
        else
        {
            CDF[index] = CDF[index-1] + *itPDF;
        }
        index++;
    }


    VnlIterator itCDF;
    index=0;
    float q1 = 0.0;
    for(itCDF = CDF.begin(); itCDF != CDF.end(); ++itCDF)
    {
        if(*itCDF>0.25)
        {
            q1 = index-1;
            break;
        }
        index++;
    }

    index=0;
    float q3 = 0.0;
    for(itCDF = CDF.begin(); itCDF != CDF.end(); ++itCDF)
    {
        if(*itCDF>0.75)
        {
            q3 = index-1;
            break;
        }
        index++;
    }


    std::cout << "Histogram Q1 / Q3 = " << q1 << " / " << q3 << std::endl;

    //Computes the histogram spread: ratio of the quartile distance to the range of the histogram. Performance metrics for image contrast, 2011, Tripathi et al.
    float range = (x_orig.max_value()-x_orig.min_value());
    float hs = (q3 - q1) / range;

    std::cout << "Histogram range : " << range << std::endl;
    std::cout << "Histogram Spread : " << hs << std::endl;

    std::ofstream fout(csvFileName, std::ios_base::out | std::ios_base::app);
    //fout << "Subject" << "," << "Resolution" << "," << "Beta" << "," << "Tissue" << "," << "Volume" <<std::endl;
    fout << subjName << "," << methodName << "," << iteration << "," << gmmBeta << "," << hs << std::endl;
    fout.close();

}
