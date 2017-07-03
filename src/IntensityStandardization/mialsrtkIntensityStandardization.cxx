
/*=========================================================================

Program: Rescale image intensity by linear transformation
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

#include "itkStatisticsImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"


#include "itkMultiThreader.h"

#include "vcl_algorithm.h"

#include "mialsrtkMaths.h"

void prompt_start(std::vector< std::string > & inputFileNames, float maxIntensity)
{
    unsigned int numberOfImages = inputFileNames.size();

    std::cout << std::endl << "----------------------------------------------------------------"<<std::endl;
    std::cout << " Intensity Standardization Program " << std::endl;
    std::cout << "----------------------------------------------------------------"<<std::endl<<std::endl;
    std::cout << std::endl << "Number of images : " << inputFileNames.size() ;

    std::cout << std::endl << "Intensity max : " << maxIntensity ;


    for(unsigned int i=0; i < numberOfImages; i++)
    {
        std::cout << "Input image " << int2str(i) << ":" <<inputFileNames[i] << std::endl;
    }

    std::cout << "###################################################### \n" << std::endl;

};

int main( int argc, char * argv [] )
{
    try {
        std::vector< std::string > inputFileNames;
        std::vector< std::string > outputFileNames;

        float maxIntensity = 255.0;

        // Parse arguments

        TCLAP::CmdLine cmd("Intensity standardization", ' ', "Unversioned");

        TCLAP::MultiArg<std::string> inputArg("i","input","input image file",true,"string",cmd);
        TCLAP::MultiArg<std::string> outputArg("o","output","output image file",false,"string",cmd);
        TCLAP::ValueArg<float> maxArg  ("","max","max intensity (255 by default)",false,255.0,"float",cmd);

        // Parse the argv array.
        cmd.parse( argc, argv );

        inputFileNames = inputArg.getValue();
        outputFileNames = outputArg.getValue();
        maxIntensity = maxArg.getValue();

        prompt_start(inputFileNames,maxIntensity);

        //Typedef
        const unsigned int Dimension = 3;
        typedef float PixelType;

        typedef itk::Image< PixelType, Dimension > ImageType;
        typedef ImageType::Pointer ImagePointer;

        typedef ImageType::RegionType RegionType;
        typedef std::vector< RegionType > RegionArrayType;

        typedef itk::ImageFileReader< ImageType > ImageReaderType;
        typedef itk::ImageFileWriter< ImageType > ImageWriterType;

        typedef itk::StatisticsImageFilter<ImageType> StatisticsImageFilterType;

        typedef itk::RescaleIntensityImageFilter< ImageType, ImageType > RescaleFilterType;


        // Number of images being evaluated
        unsigned int numberOfImages = inputFileNames.size();

        //Read reconstructed images
        std::vector<ImagePointer> images(numberOfImages);
        std::vector<float> maximums(numberOfImages);
        float global_max = 0.0;

        //Extract maximum intensity in all images
        for(unsigned int i=0; i< numberOfImages; i++)
        {
            ImageReaderType::Pointer imReader = ImageReaderType::New();
            imReader -> SetFileName( inputFileNames[i].c_str() );
            imReader -> Update();
            images[i] = imReader -> GetOutput();

            StatisticsImageFilterType::Pointer statisticsImageFilter = StatisticsImageFilterType::New ();
            statisticsImageFilter->SetInput(images[i]);
            statisticsImageFilter->Update();

            maximums[i] = statisticsImageFilter -> GetMaximum();

            //Save the max intensity in the overall set of images
            if(maximums[i]>global_max)
                global_max = maximums[i];

        }

        //Rescale intensity in all images.
        for(unsigned int i=0; i< numberOfImages; i++)
        {
            float new_max = ( maximums[i] / global_max ) * maxIntensity;
            std::cout << "Rescale intensity of image # " << int2str(i) << std::endl;
            RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
            rescaleFilter->SetInput(images[i]);
            rescaleFilter->SetOutputMinimum(0);
            rescaleFilter->SetOutputMaximum(new_max);
            rescaleFilter->Update();
            images[i] = rescaleFilter -> GetOutput();
            std::cout << "Old range = [0," << maximums[i] << "],  New range = [0," << new_max<< "]" << std::endl;
            std::cout << "----------------------------------------------------------------------------------------------------------- \n" << std::endl;
        }

        //Write output images
        for(unsigned int i=0; i< numberOfImages; i++)
        {
            std::string outputFileName = "";

            if(outputFileNames.size() == 0)
            {
                int lastdot= inputFileNames[i].find_last_of(".");

                if (lastdot != std::string::npos)
                {
                    outputFileName =  inputFileNames[i].substr(0,lastdot) + "_res.nii";
                }
                else
                {
                    outputFileName =  inputFileNames[i] + "_res.nii";
                }
            }
            else
            {
                outputFileName = outputFileNames[i];
            }

            ImageWriterType::Pointer writer = ImageWriterType::New();
            writer -> SetFileName( outputFileName );
            writer -> SetInput( images[i] );
            writer -> Update();

            std::cout << "Output Image # " << int2str(i) << " saved as " << outputFileName << std::endl;
        }

        return EXIT_SUCCESS;

    }
    catch (TCLAP::ArgException &e) // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }
};
