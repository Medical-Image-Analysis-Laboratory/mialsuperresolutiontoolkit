/*=========================================================================

Program: Computes Overlap Measures between Segmentations
Language: C++
Date: $Date$
Version: 2.0
Author: Sebastien Tourbier

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

#include "mialtkMaths.h"
#include "itkGradientMagnitudeImageFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"

#include "itkBinaryThresholdImageFilter.h"
#include "itkMaskImageFilter.h"
#include "itkMultiplyImageFilter.h"

#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryMorphologicalClosingImageFilter.h"
#include "itkBinaryErodeImageFilter.h"


int main(int argc, char *argv[])
{

    const char *inputFileName1 = NULL;
    const char *refFileName = NULL;
    const char *patientName = NULL;
    const char *stackName = NULL;
    const char *algoName = NULL;
    const char *gradmagName = NULL;
    const char *outputFileName = NULL;

    // Parse arguments

    TCLAP::CmdLine cmd("Evaluate M1 and M2 Sharpness Measures", ' ', "Unversioned");

    TCLAP::ValueArg<std::string> input1Arg  ("i","input","Input image",true,"","string",cmd);
    TCLAP::ValueArg<std::string> refArg  ("r","ref-mask","Ref image used for head masking",false,"undefined","string",cmd);
    TCLAP::ValueArg<std::string> patientArg  ("p","patient-name","Patient name",false,"undefined","string",cmd);
    TCLAP::ValueArg<std::string> stackArg  ("s","stack-name","Patient name",false,"undefined","string",cmd);
    TCLAP::ValueArg<std::string> algoArg  ("a","algo-name","Algo name",false,"undefined","string",cmd);
    TCLAP::ValueArg<std::string> gradmagArg  ("g","output-gradmag","Gradient magnitude image name",false,"undefined","string",cmd);
    TCLAP::ValueArg<std::string> outArg  ("o","output-csv","Output csv file",false,"undefined","string",cmd);

    // Parse the argv array.
    cmd.parse( argc, argv );

    inputFileName1 = input1Arg.getValue().c_str();
    refFileName  = refArg.getValue().c_str();
    patientName = patientArg.getValue().c_str();
    stackName = stackArg.getValue().c_str();
    algoName = algoArg.getValue().c_str();
    gradmagName = gradmagArg.getValue().c_str();
    outputFileName = outArg.getValue().c_str();


    const unsigned int dimension = 3;

    typedef unsigned char InputPixelType;

    typedef itk::Image<InputPixelType, dimension> InputImageType;
    typedef itk::ImageFileReader<InputImageType> ReaderType;

    typedef itk::MaskImageFilter<InputImageType,InputImageType,InputImageType> MaskFilterType;
    typedef itk::MultiplyImageFilter<InputImageType,InputImageType,InputImageType> MultiplyFilterType;
    typedef itk::BinaryThresholdImageFilter<InputImageType,InputImageType> ThresholdFilterType;

    typedef itk::BinaryBallStructuringElement<InputImageType::PixelType, dimension> StructuringElementType;
    typedef itk::BinaryMorphologicalClosingImageFilter< InputImageType, InputImageType, StructuringElementType > BinaryClosingFilterType;
    typedef itk::BinaryErodeImageFilter<InputImageType,InputImageType,StructuringElementType> BinaryErodeFilterType;


    //Read the images
    //
    //
    std::cout << "Load input  image: " << inputFileName1 << std::endl;;
    ReaderType::Pointer imReader = ReaderType::New();
    imReader->SetFileName(inputFileName1);
    try
    {
        imReader->Update();
    }
    catch( itk::ExceptionObject & err )
    {
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }
    //imReader->GetOutput()->Print(std::cout);
    InputImageType::Pointer image =  imReader -> GetOutput();
    InputImageType::Pointer origImage =  imReader -> GetOutput();
    InputImageType::Pointer refMaskImage = InputImageType::New();

    if(strncmp( refFileName , "undefined" , sizeof(refFileName) - 1))
    {
        std::cout << "Load the reference image used for head masking" << std::endl;
        ReaderType::Pointer imRefReader = ReaderType::New();
        imRefReader->SetFileName(refFileName);

        try
        {
            imRefReader->Update();
        }
        catch( itk::ExceptionObject & err )
        {
            std::cerr << err << std::endl;
            return EXIT_FAILURE;
        }

        //imRefReader->GetOutput()->Print(std::cout);


        ThresholdFilterType::Pointer thresholder = ThresholdFilterType::New();
        thresholder->SetInput(imRefReader->GetOutput());
        thresholder->SetLowerThreshold(1.0);
        thresholder->SetInsideValue(1.0);
        thresholder->SetOutsideValue(0.0);
        thresholder->Update();

        //thresholder->GetOutput()->Print(std::cout);

        //Removes small holes and erode the mask (to remove part of Tikhonov data coming from diffusion on the border)
        unsigned int radius = 1;
        StructuringElementType structuringElement;
        structuringElement.SetRadius(radius);
        structuringElement.CreateStructuringElement();

        unsigned int radius2 = 1;
        StructuringElementType structuringElement2;
        structuringElement2.SetRadius(radius2);
        structuringElement2.CreateStructuringElement();

        BinaryClosingFilterType::Pointer closingFilter = BinaryClosingFilterType::New();
        closingFilter->SetInput(thresholder->GetOutput());
        closingFilter->SetForegroundValue(1.0);
        closingFilter->SetKernel(structuringElement);

        BinaryErodeFilterType::Pointer erodeFilter = BinaryErodeFilterType::New();
        erodeFilter->SetInput(closingFilter->GetOutput());
        erodeFilter->SetKernel(structuringElement2);
        erodeFilter->SetErodeValue(1.0);
        erodeFilter->Update();

        refMaskImage = erodeFilter->GetOutput();

        itk::ImageFileWriter<InputImageType>::Pointer writer = itk::ImageFileWriter<InputImageType>::New();
        writer->SetFileName("/home/ch176971/Desktop/mask.nii.gz");
        writer->SetInput(refMaskImage);
        writer->Update();

        //refMaskImage->Print(std::cout);

        MaskFilterType::Pointer masker = MaskFilterType::New();
        masker->SetInput(image);
        masker->SetMaskImage(refMaskImage);
        masker->Update();

        image = masker->GetOutput();

        /*MultiplyFilterType::Pointer multiplier = MultiplyFilterType::New();
        multiplier->SetInput1(image);
        multiplier->SetInput2(refMaskImage);
        multiplier->Update();

        image = multiplier->GetOutput();*/
    }
    else
    {
        image = imReader -> GetOutput();
    }

    //image->Print(std::cout);

    double M1 = 0.0;
    double M2 = 0.0;

    //Vectorize the image
    std::vector<float> imageData(image->GetRequestedRegion().GetNumberOfPixels());
    int linearIndex = 0;
    //imageData.set_size(image->GetRequestedRegion().GetNumberOfPixels());
    itk::ImageRegionConstIteratorWithIndex< InputImageType > it( image,image->GetRequestedRegion());
    itk::ImageRegionConstIteratorWithIndex< InputImageType > itMask( refMaskImage,image->GetRequestedRegion());
    
    
    //Compute the mean intensity within the mask / count the number of pixels within the mask
    double mean = 0.0;
    int numberOfPixels = 0;
    for (it.GoToBegin(), itMask.GoToBegin(); !it.IsAtEnd(); ++it, ++itMask)
    {
        if(itMask.Get()>0.0)
        {
            mean += it.Get();
            numberOfPixels++;
        }
    }
    mean = mean / (double)numberOfPixels;
    
    //Compute the M1 measure
    for (it.GoToBegin(), itMask.GoToBegin(); !it.IsAtEnd(); ++it, ++itMask)
    {
        if(itMask.Get()>0.0)
        {
            M1 += (it.Get() - mean) * (it.Get() - mean);
        }
    } 
    
    //variance = variance / ((double)numberOfPixels);
    M1 = M1 / ((double)numberOfPixels - 1.0);

    std::cout << "M1 : " << M1 << " (#pixels = " << numberOfPixels << ")" << std::endl;

    //Compute the M2 measure by: 1) Computing the magnitude of the image gradient and 2) Integrating the magnitude squared over all voxels
    itk::GradientMagnitudeImageFilter<InputImageType,InputImageType>::Pointer filter = itk::GradientMagnitudeImageFilter<InputImageType,InputImageType>::New();
    filter->SetInput(origImage);

    filter->Update();

    InputImageType::Pointer gradMagImage = InputImageType::New();

    if(strncmp( refFileName , "undefined" , sizeof(refFileName) - 1))
    {
        MaskFilterType::Pointer masker2 = MaskFilterType::New();
        masker2->SetInput(filter->GetOutput());
        masker2->SetMaskImage(refMaskImage);
        masker2->Update();

        gradMagImage = masker2->GetOutput();
    }
    else
    {
        gradMagImage = filter->GetOutput();
    }

    if( strncmp( gradmagName , "undefined" , sizeof(gradmagName) - 1) )
        try{
        std::cout << "Save the gradient magnitude image: " << gradmagName << std::endl;
        itk::ImageFileWriter<InputImageType>::Pointer writer = itk::ImageFileWriter<InputImageType>::New();
        writer->SetFileName(gradmagName);
        writer->SetInput(gradMagImage);
        writer->Update();
    }
    catch(itk::ImageFileWriterException e)
    {
        std::cerr << "Error: save gradient magnitude image" << std::endl;
        return 0;
    }

    //Vectorize the image
    std::vector<float> gradMagData(image->GetRequestedRegion().GetNumberOfPixels());
    double sum = 0.0;
    linearIndex = 0;
    //std::cout << "Image region" << image->GetRequestedRegion() << std::endl;
    //std::cout << "GradMag region" <<gradMagImage->GetRequestedRegion() << std::endl;
    //gradMagData.set_size(gradMagImage->GetRequestedRegion().GetNumberOfPixels());
    itk::ImageRegionConstIteratorWithIndex< InputImageType > gradIt( gradMagImage,image->GetRequestedRegion());

    int counter = 0;

    for (gradIt.GoToBegin(), itMask.GoToBegin(); !gradIt.IsAtEnd(); ++gradIt, ++itMask)
    {
        if(itMask.Get()>0.0)
        {
            sum += gradIt.Get()*gradIt.Get();
            counter++;
        }
    }

    M2 = sum;

    std::cout << "M2 : " << M2 << " ( " << counter << " )" << std::endl;

    if( strncmp( outputFileName , "undefined" , sizeof(outputFileName) - 1) )
    {
        std::cout << "Write evaluation to file" << outputFileName << std::endl;
        std::ofstream fout(outputFileName, std::ios_base::out | std::ios_base::app);
        fout << patientName << ',' << stackName << ',' <<algoName << ',';
        fout << M1 << ','<< M2 <<std::endl;
        fout.close();
    }

    //diceCoefFilter->Print(std::cout);

    std::cout << "Done! " << std::endl;
    return 0;
}

