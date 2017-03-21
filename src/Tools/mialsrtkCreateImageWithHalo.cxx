/*==========================================================================

  © 

  Date: 01/05/2015
  Author(s): Sebastien Tourbier (sebastien.tourbier@unil.ch)

==========================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

/* Standard includes */
#include <tclap/CmdLine.h>
#include <sstream>  

#include <iostream>
#include <fstream> 
#include <string>
#include <stdlib.h> 

/* Itk includes */
#include "itkEuler3DTransform.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageMaskSpatialObject.h"
#include "itkTransformFileReader.h"
#include "itkTransformFactory.h"
#include "itkCastImageFilter.h"

#include "itkPermuteAxesImageFilter.h"
#include "itkFlipImageFilter.h"
#include "itkOrientImageFilter.h"  

/*Btk includes*/
//#include "btkSliceBySliceTransform.h"
//#include "btkSuperResolutionImageFilter.h"

#include "itkResampleImageFilter.h"

#include "itkExtractImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkMultiplyImageFilter.h"

#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"

#include "itkImageDuplicator.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIteratorWithIndex.h"

#include "itkBinaryThresholdImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryMorphologicalClosingImageFilter.h"

#include "itkBinaryDilateImageFilter.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryFillholeImageFilter.h"

//#include "../Utilities/mialtkMaths.h"


/* Time profiling */
/*
#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 0
#else
#include <time.h>
#endif

double getTime(void)
{
    struct timespec tv;

#ifdef __MACH__
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    if(clock_get_time(cclock, &mts) != 0) return 0;
    mach_port_deallocate(mach_task_self(), cclock);
    tv.tv_sec = mts.tv_sec;
    tv.tv_nsec = mts.tv_nsec;
#else
    if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;
#endif
    return (((double) tv.tv_sec) + (double) (tv.tv_nsec / 1000000000.0));
}
*/

int main( int argc, char *argv[] )
{

    try {

        const char *inputIm = NULL;
        const char *inputMask = NULL;
        const char *output = NULL;

        const char *test = "undefined";

        std::vector< int > x1, y1, z1, x2, y2, z2;
   
        double start_time_unix, end_time_unix, diff_time_unix;

        // Parse arguments
        TCLAP::CmdLine cmd("Adds an halo (2 pixels) of intensity value of 255 around the brain.", ' ', "Unversioned");
        
        // Input image with tissue labels
        TCLAP::ValueArg<std::string> inputArg  ("i","input","Input image",true,"","string",cmd);
        
        // Input brain mask
        TCLAP::ValueArg<std::string> maskArg  ("m","mask","Input brain mask",true,"","string",cmd);

        // Input reconstructed image for initialization
        TCLAP::ValueArg<std::string> outputArg  ("o","output","Output image with halo." ,true,"","string",cmd);
      
        //TCLAP::ValueArg<std::string>debugDirArg("","debug","Directory where  SR reconstructed image at each outer loop of the reconstruction optimization is saved",false,"","string",cmd);

        // Parse the argv array.
        cmd.parse( argc, argv );
    
        inputIm = inputArg.getValue().c_str();
        inputMask = maskArg.getValue().c_str();
        output = outputArg.getValue().c_str();
        

        // typedefs
        const   unsigned int    Dimension = 3;
        typedef float  PixelType;

        typedef itk::Image< PixelType, Dimension >  ImageType;
        typedef ImageType::Pointer                  ImagePointer;
        typedef std::vector<ImagePointer>           ImagePointerArray;

        typedef itk::ImageFileReader< ImageType >   ImageReaderType;
        typedef itk::ImageFileWriter< ImageType >   ImageWriterType;

        //typedef itk::CastImageFilter<ImageType,ImageMaskType> CasterType;

        // A helper class which creates an image which is perfect copy of the input image
        typedef itk::ImageDuplicator<ImageType> DuplicatorType;

        typedef itk::AddImageFilter< ImageType, ImageType, ImageType > AddImageFilterType;
        typedef itk::SubtractImageFilter< ImageType, ImageType, ImageType > SubtractImageFilterType;
        typedef itk::MultiplyImageFilter< ImageType, ImageType, ImageType > MultiplyImageFilterType;

        //std::vector<OrientImageFilterType::Pointer> orientImageFilter(numberOfImages);
        //std::vector<OrientImageMaskFilterType::Pointer> orientMaskImageFilter(numberOfImages);


        // Filter setup
        std::cout<<"Reading label image: "<<inputIm<<std::endl;
        ImageReaderType::Pointer imageReader = ImageReaderType::New();
        imageReader -> SetFileName( inputIm );
        imageReader -> Update();
        
        ImageType::Pointer image = imageReader -> GetOutput();

        std::cout<<"Reading mask image : "<<inputMask<<std::endl;
        ImageReaderType::Pointer maskReader = ImageReaderType::New();
        maskReader -> SetFileName( inputMask );
        maskReader -> Update();

        ImageType::Pointer imageMask = maskReader  -> GetOutput();


        std::cout << "==========================================================================" << std::endl << std::endl;

            
        typedef itk::BinaryBallStructuringElement<ImageType::PixelType, ImageType::ImageDimension> StructuringElementType;
        unsigned int radius = 2;
        StructuringElementType structuringElement;
        structuringElement.SetRadius(radius);
        structuringElement.CreateStructuringElement();
        
        //Dilates the brain mask
        typedef itk::BinaryDilateImageFilter <ImageType, ImageType, StructuringElementType> BinaryDilateImageFilterType;
        BinaryDilateImageFilterType::Pointer dilateFilter = BinaryDilateImageFilterType::New();
        dilateFilter->SetInput(imageMask.GetPointer());
        dilateFilter->SetKernel(structuringElement);
        dilateFilter->SetForegroundValue( 1.0 );
        dilateFilter->Update();
        
        //Creates the halo
        SubtractImageFilterType::Pointer subFilter = SubtractImageFilterType::New();
        subFilter->SetInput1(dilateFilter->GetOutput());
        subFilter->SetInput2(imageMask.GetPointer());
        subFilter->Update();
        
        //Creates intensities equaled to 255 in the halo
        MultiplyImageFilterType::Pointer multFilter = MultiplyImageFilterType::New();
        multFilter->SetInput(subFilter->GetOutput());
        multFilter->SetConstant(255.0);
        multFilter->Update();
        
        //Add the halo to the original input image
        AddImageFilterType::Pointer addFilter = AddImageFilterType::New();
        addFilter->SetInput1(image.GetPointer());
        addFilter->SetInput2(multFilter->GetOutput());
        addFilter->Update();

        //Saves the white matter volumic image
        ImageWriterType::Pointer writer =  ImageWriterType::New();
        writer -> SetFileName( output );
        writer -> SetInput( addFilter -> GetOutput() );

        if ( strcmp(output,"") != 0)
        {
            std::cout << "Writing " << output << " ... ";
            writer->Update();
            std::cout << "done." << std::endl;
        }


    } catch (TCLAP::ArgException &e)  // catch any exceptions
    { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

    return EXIT_SUCCESS;
}
