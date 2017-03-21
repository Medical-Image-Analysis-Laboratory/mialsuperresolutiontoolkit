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

        const char *inputLabels = NULL;
        const char *inputMask = NULL;
        const char *output = NULL;

        const char *test = "undefined";

        std::vector< int > x1, y1, z1, x2, y2, z2;
   
        double start_time_unix, end_time_unix, diff_time_unix;

        // Parse arguments
        TCLAP::CmdLine cmd("Creates  the white matter volume from segmentation labels to allow postprocessing with freesurfer (Cortical complexity).", ' ', "Unversioned");
        
        // Input image with tissue labels
        TCLAP::ValueArg<std::string> inputArg  ("i","input-labels","Input labels",true,"","string",cmd);
        
        // Input brain mask
        TCLAP::ValueArg<std::string> maskArg  ("m","input-mask","Input brain mask",true,"","string",cmd);

        // Input reconstructed image for initialization
        TCLAP::ValueArg<std::string> outputArg  ("o","output-wm","Output white matter volume. "
                                              "Typically it is further used in freesurfer." ,true,"","string",cmd);
      
        //TCLAP::ValueArg<std::string>debugDirArg("","debug","Directory where  SR reconstructed image at each outer loop of the reconstruction optimization is saved",false,"","string",cmd);

        // Parse the argv array.
        cmd.parse( argc, argv );
    
        inputLabels = inputArg.getValue().c_str();
        inputMask = maskArg.getValue().c_str();
        output = outputArg.getValue().c_str();
        

        // typedefs
        const   unsigned int    Dimension = 3;
        typedef float  PixelType;

        typedef itk::Image< PixelType, Dimension >  ImageType;
        typedef ImageType::Pointer                  ImagePointer;
        typedef std::vector<ImagePointer>           ImagePointerArray;

        typedef itk::Image< unsigned char, Dimension >  ImageMaskType;
        typedef itk::Image< unsigned char, 2 >  SliceMaskType;
        
        typedef itk::ImageFileReader< ImageMaskType > MaskReaderType;
        typedef itk::ImageMaskSpatialObject< Dimension > MaskType;

        typedef ImageType::RegionType               RegionType;
        typedef std::vector< RegionType >           RegionArrayType;
        
        //typedef btk::SliceBySliceTransformBase< double, Dimension > TransformBaseType;
        //typedef btk::SliceBySliceTransform< double, Dimension > TransformType;
        //typedef TransformType::Pointer                          TransformPointer;

        typedef itk::ImageFileReader< ImageType >   ImageReaderType;
        typedef itk::ImageFileWriter< ImageMaskType >   MaskWriterType;

        //typedef itk::TransformFileReader     TransformReaderType;
        //typedef TransformReaderType::TransformListType * TransformListType;

        // Rigid 3D transform definition (typically for reconstructions in adults)
        //typedef btk::Euler3DTransform< double > Rigid3DTransformType;
        //typedef Rigid3DTransformType::Pointer   Rigid3DTransformPointer;

        //typedef itk::Euler3DTransform< double > EulerTransformType;

        //typedef btk::ImageIntersectionCalculator<ImageType> IntersectionCalculatorType;
        //IntersectionCalculatorType::Pointer intersectionCalculator = IntersectionCalculatorType::New();

        // Interpolator used to compute the error metric between 2 registration iterations
        typedef itk::NearestNeighborInterpolateImageFunction<ImageMaskType,double>     NNMaskInterpolatorType;
        //typedef itk::LinearInterpolateImageFunction<ImageType,double>     LinearInterpolatorType;
        //typedef itk::BSplineInterpolateImageFunction<ImageType,double>     BSplineInterpolatorType;

        typedef itk::ResampleImageFilter<ImageMaskType, ImageMaskType> ResamplerImageMaskFilterType;

        typedef itk::ExtractImageFilter<ImageMaskType, ImageMaskType> ExtractImageMaskFilterType;
        
        typedef itk::ExtractImageFilter<ImageMaskType, SliceMaskType> ExtractSliceMaskFilterType;
        
        typedef itk::ExtractImageFilter<ImageType, ImageType> ExtractImageFilterType;

        //typedef itk::CastImageFilter<ImageType,ImageMaskType> CasterType;

        // A helper class which creates an image which is perfect copy of the input image
        typedef itk::ImageDuplicator<ImageType> DuplicatorType;

        typedef itk::OrientImageFilter<ImageType,ImageType> OrientImageFilterType;
        typedef itk::OrientImageFilter<ImageMaskType,ImageMaskType> OrientImageMaskFilterType;

        typedef itk::ImageRegionIterator< ImageMaskType >  MaskIteratorType;
        typedef itk::ImageRegionIterator< SliceMaskType >  SliceMaskIteratorType;
        typedef itk::ImageRegionIterator< ImageType >  IteratorType;
        typedef itk::ImageRegionIteratorWithIndex< ImageMaskType >  MaskIteratorTypeWithIndex;

        //typedef itk::AddImageFilter< ImageMaskType, ImageMaskType, ImageMaskType > AddImageMaskFilter;
        //typedef itk::MultiplyImageFilter< ImageMaskType, ImageMaskType, ImageMaskType > MultiplyImageMaskFilterType;

        //std::vector<OrientImageFilterType::Pointer> orientImageFilter(numberOfImages);
        //std::vector<OrientImageMaskFilterType::Pointer> orientMaskImageFilter(numberOfImages);

        ImageMaskType::Pointer      imageMask;
        MaskType::Pointer           mask;
        RegionType                  roi;

        ImageType::IndexType  roiIndex;
        ImageType::SizeType   roiSize;

        // Filter setup
        std::cout<<"Reading label image: "<<inputLabels<<std::endl;
        ImageReaderType::Pointer imageReader = ImageReaderType::New();
        imageReader -> SetFileName( inputLabels );
        imageReader -> Update();
        
        ImageType::Pointer labelIm = imageReader -> GetOutput();

        std::cout<<"Reading mask image : "<<mask<<std::endl;
        MaskReaderType::Pointer maskReader = MaskReaderType::New();
        maskReader -> SetFileName( inputMask );
        maskReader -> Update();

        imageMask = maskReader  -> GetOutput();

        /*
        orientMaskImageFilter[i] = OrientImageMaskFilterType::New();
        orientMaskImageFilter[i] -> UseImageDirectionOn();
        orientMaskImageFilter[i] -> SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP);
        orientMaskImageFilter[i] -> SetInput(maskReader -> GetOutput());
        orientMaskImageFilter[i] -> Update();

        imageMasks[i] = orientMaskImageFilter[i]  -> GetOutput();
        */

        //MaskType::Pointer mask = MaskType::New();
        mask = MaskType::New();
        mask -> SetImage( imageMask );

        roi = mask -> GetAxisAlignedBoundingBoxRegion();
        //std::cout << "roi : "<<roi<<std::endl;
        

        std::cout << "==========================================================================" << std::endl << std::endl;

        ImageMaskType::Pointer outWMVolume = ImageMaskType::New();
        outWMVolume->SetRegions(labelIm->GetLargestPossibleRegion());
        outWMVolume->Allocate();
        outWMVolume->FillBuffer(0.0);

        outWMVolume->SetOrigin(labelIm->GetOrigin());
        outWMVolume->SetSpacing(labelIm->GetSpacing());
        outWMVolume->SetDirection(labelIm->GetDirection());

        IteratorType itLabelIm(labelIm,roi);
        MaskIteratorType itOutWMVol(outWMVolume,roi);
        
        
        //Creates the white matter volume composed of white matter label and all labels contained inside the white matter
        for (itLabelIm.GoToBegin(), itOutWMVol.GoToBegin(); !itLabelIm.IsAtEnd(); ++itLabelIm, ++itOutWMVol)
        {
            //Combines the labels 3, 6 and 7
            if( (itLabelIm.Get() == 3) || (itLabelIm.Get() == 4) || (itLabelIm.Get() == 6) || (itLabelIm.Get() == 7) )
            {
                itOutWMVol.Set( 1.0 );
            }
        }
        
        //TODO: fill holes created by the ventricules, maybe we can used the brain mask
        /*
        unsigned int radius = 4;
        
        typedef itk::BinaryBallStructuringElement<ImageMaskType::PixelType, ImageMaskType::ImageDimension>
        StructuringElementType;
        StructuringElementType structuringElement;
        structuringElement.SetRadius(radius);
        structuringElement.CreateStructuringElement();
        
        typedef itk::BinaryMorphologicalClosingImageFilter <ImageMaskType, ImageMaskType, StructuringElementType>
        BinaryMorphologicalClosingImageFilterType;
        BinaryMorphologicalClosingImageFilterType::Pointer closingFilter = BinaryMorphologicalClosingImageFilterType::New();
        closingFilter->SetInput(outWMVolume.GetPointer());
        closingFilter->SetKernel(structuringElement);
        closingFilter->SetForegroundValue(1.0);
        closingFilter->Update();
        */
        

        ImageMaskType::IndexType inputIndex = roi.GetIndex();
        ImageMaskType::SizeType  inputSize  = roi.GetSize();

        unsigned int i=inputIndex[1] + inputSize[1];

        //Create slice by slice (coronal direction) a white matter area composed of white matter label and all labels contained inside the white matter
        
        unsigned int sliceDirection = 1;
        for ( unsigned int i=inputIndex[sliceDirection]; i < inputIndex[sliceDirection] + inputSize[sliceDirection]; i++ )
        {
            std::cout << "process slice #" << i << std::endl;
            
            ImageMaskType::RegionType wholeSliceRegion;
            wholeSliceRegion = roi;
            
            ImageMaskType::IndexType  wholeSliceRegionIndex = wholeSliceRegion.GetIndex();
            ImageMaskType::SizeType   wholeSliceRegionSize  = wholeSliceRegion.GetSize();
            
            wholeSliceRegionIndex[sliceDirection]= i;
            wholeSliceRegionSize[sliceDirection] = 0;
            
            wholeSliceRegion.SetIndex(wholeSliceRegionIndex);
            wholeSliceRegion.SetSize(wholeSliceRegionSize);
            
            //Extract slice
            ExtractSliceMaskFilterType::Pointer sliceExtractor = ExtractSliceMaskFilterType::New();
            sliceExtractor->SetExtractionRegion(wholeSliceRegion);
            sliceExtractor->SetInput(outWMVolume.GetPointer());
#if ITK_VERSION_MAJOR >= 4
            sliceExtractor->SetDirectionCollapseToIdentity(); // This is required.
#endif
            sliceExtractor->Update();
            
            //Create a white matter area composed of white matter label and all labels contained inside the white matter
            
            /*
             typedef itk::BinaryMorphologicalClosingImageFilter <SliceMaskType, SliceMaskType, StructuringElementSliceType> BinaryMorphologicalClosingSliceFilterType;
             BinaryMorphologicalClosingSliceFilterType::Pointer closingSliceFilter = BinaryMorphologicalClosingSliceFilterType::New();
             closingSliceFilter->SetInput(sliceExtractor->GetOutput());
             closingSliceFilter->SetKernel(structuringSliceElement);
             closingSliceFilter->SetForegroundValue(1.0);
             closingSliceFilter->Update();
             */
            
            typedef itk::BinaryFillholeImageFilter< SliceMaskType > BinaryFillholeSliceMaskFilterType;
            BinaryFillholeSliceMaskFilterType::Pointer fillingHoleFilter = BinaryFillholeSliceMaskFilterType::New();
            fillingHoleFilter->SetInput( sliceExtractor->GetOutput() );
            fillingHoleFilter->SetFullyConnected( true );
            fillingHoleFilter->SetForegroundValue( 1.0 );
            fillingHoleFilter->Update();
            
            
            typedef itk::BinaryBallStructuringElement<SliceMaskType::PixelType, SliceMaskType::ImageDimension> StructuringElementSliceType;
            unsigned int radius = 0;
            StructuringElementSliceType structuring2SliceElement;
            structuring2SliceElement.SetRadius(radius);
            structuring2SliceElement.CreateStructuringElement();
            
            //Remove small component
            typedef itk::BinaryErodeImageFilter <SliceMaskType, SliceMaskType, StructuringElementSliceType> BinaryErodeSliceFilterType;
            BinaryErodeSliceFilterType::Pointer erodeFilter = BinaryErodeSliceFilterType::New();
            erodeFilter->SetInput(fillingHoleFilter->GetOutput());
            erodeFilter->SetKernel(structuring2SliceElement);
            erodeFilter->SetForegroundValue( 1.0 );
            erodeFilter->Update();
            
            //Dilate by 1 the slice mask where small components have been removed by the erode operation
            radius = 0;
            StructuringElementSliceType structuringSliceElement;
            structuringSliceElement.SetRadius(radius);
            structuringSliceElement.CreateStructuringElement();
            
            typedef itk::BinaryDilateImageFilter <SliceMaskType, SliceMaskType, StructuringElementSliceType> BinaryDilateSliceFilterType;
            BinaryDilateSliceFilterType::Pointer dilateFilter = BinaryDilateSliceFilterType::New();
            dilateFilter->SetInput(erodeFilter->GetOutput());
            dilateFilter->SetKernel(structuringSliceElement);
            dilateFilter->SetForegroundValue( 1.0 );
            dilateFilter->Update();
            
            //SliceMaskType::Pointer outWMslice = fillingHoleFilter->GetOutput();
            SliceMaskType::Pointer outWMslice = dilateFilter->GetOutput();
            
            SliceMaskIteratorType itOutWMSlice(outWMslice,outWMslice->GetLargestPossibleRegion());
            
            wholeSliceRegionIndex[sliceDirection]= i;
            wholeSliceRegionSize[sliceDirection] = 1;
            
            wholeSliceRegion.SetIndex(wholeSliceRegionIndex);
            wholeSliceRegion.SetSize(wholeSliceRegionSize);
            
            MaskIteratorType itOutWMVolBySlice(outWMVolume,wholeSliceRegion);
            
            for (itOutWMVolBySlice.GoToBegin(), itOutWMSlice.GoToBegin(); !itOutWMVolBySlice.IsAtEnd(); ++itOutWMVolBySlice, ++itOutWMSlice)
            {
                itOutWMVolBySlice.Set(itOutWMSlice.Get());
            }
            
        }

        
        sliceDirection = 2;
        for ( unsigned int i=inputIndex[sliceDirection]; i < inputIndex[sliceDirection] + inputSize[sliceDirection]; i++ )
        {
            std::cout << "process slice #" << i << std::endl;
            
            ImageMaskType::RegionType wholeSliceRegion;
            wholeSliceRegion = roi;
            
            ImageMaskType::IndexType  wholeSliceRegionIndex = wholeSliceRegion.GetIndex();
            ImageMaskType::SizeType   wholeSliceRegionSize  = wholeSliceRegion.GetSize();
            
            wholeSliceRegionIndex[sliceDirection]= i;
            wholeSliceRegionSize[sliceDirection] = 0;
            
            wholeSliceRegion.SetIndex(wholeSliceRegionIndex);
            wholeSliceRegion.SetSize(wholeSliceRegionSize);
            
            //Extract slice
            ExtractSliceMaskFilterType::Pointer sliceExtractor = ExtractSliceMaskFilterType::New();
            sliceExtractor->SetExtractionRegion(wholeSliceRegion);
            sliceExtractor->SetInput(outWMVolume.GetPointer());
#if ITK_VERSION_MAJOR >= 4
            sliceExtractor->SetDirectionCollapseToIdentity(); // This is required.
#endif
            sliceExtractor->Update();
            
            //Create a white matter area composed of white matter label and all labels contained inside the white matter
            unsigned int radius = 2;
            
            typedef itk::BinaryBallStructuringElement<SliceMaskType::PixelType, SliceMaskType::ImageDimension>
            StructuringElementSliceType;
            StructuringElementSliceType structuringSliceElement;
            structuringSliceElement.SetRadius(radius);
            structuringSliceElement.CreateStructuringElement();
            
            /*
             typedef itk::BinaryMorphologicalClosingImageFilter <SliceMaskType, SliceMaskType, StructuringElementSliceType> BinaryMorphologicalClosingSliceFilterType;
             BinaryMorphologicalClosingSliceFilterType::Pointer closingSliceFilter = BinaryMorphologicalClosingSliceFilterType::New();
             closingSliceFilter->SetInput(sliceExtractor->GetOutput());
             closingSliceFilter->SetKernel(structuringSliceElement);
             closingSliceFilter->SetForegroundValue(1.0);
             closingSliceFilter->Update();
             */
            
            typedef itk::BinaryFillholeImageFilter< SliceMaskType > BinaryFillholeSliceMaskFilterType;
            BinaryFillholeSliceMaskFilterType::Pointer fillingHoleFilter = BinaryFillholeSliceMaskFilterType::New();
            fillingHoleFilter->SetInput( sliceExtractor->GetOutput() );
            fillingHoleFilter->SetFullyConnected( true );
            fillingHoleFilter->SetForegroundValue( 1.0 );
            fillingHoleFilter->Update();
            
            SliceMaskType::Pointer outWMslice = fillingHoleFilter->GetOutput();
            
            SliceMaskIteratorType itOutWMSlice(outWMslice,outWMslice->GetLargestPossibleRegion());
            
            wholeSliceRegionIndex[sliceDirection]= i;
            wholeSliceRegionSize[sliceDirection] = 1;
            
            wholeSliceRegion.SetIndex(wholeSliceRegionIndex);
            wholeSliceRegion.SetSize(wholeSliceRegionSize);
            
            MaskIteratorType itOutWMVolBySlice(outWMVolume,wholeSliceRegion);
            
            for (itOutWMVolBySlice.GoToBegin(), itOutWMSlice.GoToBegin(); !itOutWMVolBySlice.IsAtEnd(); ++itOutWMVolBySlice, ++itOutWMSlice)
            {
                itOutWMVolBySlice.Set(itOutWMSlice.Get());
            }
            
        }

        
        
        /*
        sliceDirection = 0;
        for ( unsigned int i=inputIndex[sliceDirection]; i < inputIndex[sliceDirection] + inputSize[sliceDirection]; i++ )
        {
            std::cout << "process slice #" << i << std::endl;
            
            ImageMaskType::RegionType wholeSliceRegion;
            wholeSliceRegion = roi;
            
            ImageMaskType::IndexType  wholeSliceRegionIndex = wholeSliceRegion.GetIndex();
            ImageMaskType::SizeType   wholeSliceRegionSize  = wholeSliceRegion.GetSize();
            
            wholeSliceRegionIndex[sliceDirection]= i;
            wholeSliceRegionSize[sliceDirection] = 0;
            
            wholeSliceRegion.SetIndex(wholeSliceRegionIndex);
            wholeSliceRegion.SetSize(wholeSliceRegionSize);
            
            //Extract slice
            ExtractSliceMaskFilterType::Pointer sliceExtractor = ExtractSliceMaskFilterType::New();
            sliceExtractor->SetExtractionRegion(wholeSliceRegion);
            sliceExtractor->SetInput(outWMVolume.GetPointer());
#if ITK_VERSION_MAJOR >= 4
            sliceExtractor->SetDirectionCollapseToIdentity(); // This is required.
#endif
            sliceExtractor->Update();
            
            //Create a white matter area composed of white matter label and all labels contained inside the white matter
            unsigned int radius = 2;
            
            typedef itk::BinaryBallStructuringElement<SliceMaskType::PixelType, SliceMaskType::ImageDimension>
            StructuringElementSliceType;
            StructuringElementSliceType structuringSliceElement;
            structuringSliceElement.SetRadius(radius);
            structuringSliceElement.CreateStructuringElement();
            
            typedef itk::BinaryFillholeImageFilter< SliceMaskType > BinaryFillholeSliceMaskFilterType;
            BinaryFillholeSliceMaskFilterType::Pointer fillingHoleFilter = BinaryFillholeSliceMaskFilterType::New();
            fillingHoleFilter->SetInput( sliceExtractor->GetOutput() );
            fillingHoleFilter->SetFullyConnected( true );
            fillingHoleFilter->SetForegroundValue( 1.0 );
            fillingHoleFilter->Update();
            
            SliceMaskType::Pointer outWMslice = fillingHoleFilter->GetOutput();
            
            SliceMaskIteratorType itOutWMSlice(outWMslice,outWMslice->GetLargestPossibleRegion());
            
            wholeSliceRegionIndex[sliceDirection]= i;
            wholeSliceRegionSize[sliceDirection] = 1;
            
            wholeSliceRegion.SetIndex(wholeSliceRegionIndex);
            wholeSliceRegion.SetSize(wholeSliceRegionSize);
            
            MaskIteratorType itOutWMVolBySlice(outWMVolume,wholeSliceRegion);
            
            for (itOutWMVolBySlice.GoToBegin(), itOutWMSlice.GoToBegin(); !itOutWMVolBySlice.IsAtEnd(); ++itOutWMVolBySlice, ++itOutWMSlice)
            {
                itOutWMVolBySlice.Set(itOutWMSlice.Get());
            }
            
        }
        */
        
        


        //Saves the white matter volumic image
        MaskWriterType::Pointer maskWriter =  MaskWriterType::New();
        maskWriter -> SetFileName( output );
        //writer -> SetInput( resampler -> GetOutput() );
        maskWriter -> SetInput( outWMVolume.GetPointer() );

        if ( strcmp(output,"") != 0)
        {
            std::cout << "Writing " << output << " ... ";
            maskWriter->Update();
            std::cout << "done." << std::endl;
        }


    } catch (TCLAP::ArgException &e)  // catch any exceptions
    { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

    return EXIT_SUCCESS;
}
