/*=========================================================================
Program: Project all negative intensities in the input image to zero

Language: C++
Date: $Date: 2012-28-12 $
Version: $Revision: 1 $
Author: $Sebastien Tourbier$

Copyright (c) 2017 Medical Image Analysis Laboratory (MIAL), Lausanne
  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
     
==========================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

/* Standard includes */
#include <tclap/CmdLine.h>
#include "stdio.h"

/* Itk includes */
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageDuplicator.h"
#include "itkImageRegionIterator.h"

/* mialsrtk includes */
#include "mialsrtkMaths.h"

int main( int argc, char *argv[] )
{
    try {

        std::string  input;
        std::string  output;

        // Parse arguments
        TCLAP::CmdLine cmd("Project all negative intensities in the input image to zero", ' ', "Unversioned");

        TCLAP::ValueArg<std::string> inputArg("i","input","Input Image file",true,"","string",cmd);
        TCLAP::ValueArg<std::string> outputArg("o","output","Output Image file",true,"","string",cmd);


        // Parse the argv array.
        cmd.parse( argc, argv );

        input = inputArg.getValue();
        output = outputArg.getValue();

        //Typedefs
        const    unsigned int    Dimension3D = 3;
        typedef  float           PixelType;

        typedef itk::Image< PixelType, Dimension3D >  ImageType;
        typedef ImageType::Pointer                  ImagePointer;

        typedef ImageType::RegionType               ImageRegionType;
        typedef std::vector< ImageRegionType >           ImageRegionArrayType;

        typedef itk::ImageFileReader< ImageType  >  ImageReaderType;
         typedef itk::ImageFileWriter< ImageType  >  ImageWriterType;

        //Load image
        ImageReaderType::Pointer imageReader = ImageReaderType::New();
        imageReader -> SetFileName( input.c_str() );
        imageReader -> Update();

        ImagePointer inputImage = imageReader -> GetOutput();

        itk::ImageRegionIterator< ImageType >inImIt( inputImage, inputImage->GetLargestPossibleRegion() );

        //Duplicate input image for output
        itk::ImageDuplicator<ImageType>::Pointer duplicator = itk::ImageDuplicator<ImageType>::New();
        duplicator -> SetInputImage(inputImage);
        duplicator -> Update();
        ImagePointer outputImage = duplicator -> GetOutput();

        itk::ImageRegionIterator< ImageType > outImIt( outputImage, outputImage->GetLargestPossibleRegion() );

        //Set to zero all negative values
        int counter = 0;
        for(inImIt.GoToBegin(),outImIt.GoToBegin();!inImIt.IsAtEnd();++inImIt,++outImIt)
        {
            if(inImIt.Get()<0.0)
            {
                outImIt.Set(0.0);
                counter++;
            }
        }

        std::cout << "Number of voxels set to 0 : " << int2str(counter) << std::endl;

        ImageWriterType::Pointer writer = ImageWriterType::New();
        writer -> SetFileName(output.c_str());
        writer -> SetInput(outputImage);
        writer -> Update();

        return EXIT_SUCCESS;

    } catch (TCLAP::ArgException &e)  // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }
}

