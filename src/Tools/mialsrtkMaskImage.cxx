/*=========================================================================

Program: Apply mask to input image and return the output masked image
Language: C++
Date: $Date: 2015-05-07 $
Version: $Revision: 1.0 $
Author: $Sebastien Tourbier$

==========================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

/* Standard includes */
#include <tclap/CmdLine.h>
#include <sstream>  
#include <string>
#include <stdlib.h> 

/* Itk includes */
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkEuler3DTransform.h"

#include "itkMultiplyImageFilter.h"

#include "itkResampleImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"

int main( int argc, char *argv[] )
{

    try {

        std::string input;
        std::string mask;
        std::string output;

        // Parse arguments
        TCLAP::CmdLine cmd("Apply binary mask to input image and return the output masked image.", ' ', "1.0");

        // Ouput HR image
        TCLAP::ValueArg<std::string> inputArg  ("i","input","Input image file",true,"","string",cmd);

        // Ouput HR image
        TCLAP::ValueArg<std::string> maskArg  ("m","mask","Input binary mask image file",true,"","string",cmd);

        // Ouput HR image
        TCLAP::ValueArg<std::string> outputArg  ("o","output","Output masked image",true,"","string",cmd);

        // Parse the argv array.
        cmd.parse( argc, argv );

        input = inputArg.getValue();
        mask = maskArg.getValue();
        output = outputArg.getValue();

         // typedefs
        const   unsigned int    Dimension = 3;
        typedef float  PixelType;
        typedef unsigned char  MaskPixelType;

        typedef itk::Image< PixelType, Dimension >  ImageType;
        typedef itk::ImageFileReader< ImageType > ImageReaderType;
        typedef itk::ImageFileWriter< ImageType > ImageWriterType;

        typedef itk::Image< MaskPixelType, Dimension >  ImageMaskType;
        typedef itk::ImageFileReader< ImageMaskType > MaskReaderType;

        std::cout<<"Reading image : "<<input.c_str()<<std::endl;
        ImageReaderType::Pointer imageReader = ImageReaderType::New();
        imageReader -> SetFileName( input.c_str() );
        imageReader->Update();

        ImageType::Pointer im = imageReader->GetOutput();

        std::cout<<"Reading mask image : "<<mask.c_str()<<std::endl;
        MaskReaderType::Pointer maskReader = MaskReaderType::New();
        maskReader -> SetFileName( mask.c_str() );
        maskReader->Update();

        typedef itk::ResampleImageFilter<ImageMaskType,ImageMaskType> ResampleImageMaskFilterType;
        typedef itk::NearestNeighborInterpolateImageFunction<ImageMaskType> NNInterpolatorType;

        ResampleImageMaskFilterType::Pointer maskUpsampler = ResampleImageMaskFilterType::New();
        NNInterpolatorType::Pointer nnInterpolator = NNInterpolatorType::New();

        typedef  itk::Euler3DTransform<double> Euler3DTransformType;
        Euler3DTransformType::Pointer idTransform = Euler3DTransformType::New();
        idTransform->SetIdentity();

        maskUpsampler -> SetInterpolator(nnInterpolator);
        maskUpsampler -> SetInput(maskReader->GetOutput());
        maskUpsampler -> SetTransform(idTransform);
        maskUpsampler -> SetOutputParametersFromImage(im);
        maskUpsampler -> SetOutputSpacing(im->GetSpacing());
        maskUpsampler -> SetSize(im->GetLargestPossibleRegion().GetSize());

        maskUpsampler -> Update();


        typedef itk::MultiplyImageFilter< ImageType, ImageMaskType, ImageType > MultiplyImageFilterType;
        MultiplyImageFilterType::Pointer filter = MultiplyImageFilterType::New();
        filter -> SetInput1(imageReader -> GetOutput());
        filter -> SetInput2(maskUpsampler -> GetOutput());

        std::cout<<"Writing masked image : "<<output.c_str()<<std::endl;
        ImageWriterType::Pointer writer =  ImageWriterType::New();
        writer -> SetFileName( output.c_str() );
        writer -> SetInput( filter->GetOutput() );

        try
	    {
	    	writer->Update();
	    }
	  	catch( itk::ExceptionObject & error )
	    {
	    	std::cerr << "Error: " << error << std::endl;
	    	return EXIT_FAILURE;
	    }

    } 
    catch (TCLAP::ArgException &e)  // catch any exceptions
    { 
    	std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl; 
    }

    return EXIT_SUCCESS;
}
