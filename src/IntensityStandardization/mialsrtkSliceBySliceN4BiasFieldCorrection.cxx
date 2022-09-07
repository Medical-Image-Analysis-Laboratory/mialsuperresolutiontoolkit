/*=========================================================================

Program: Independent slice by slice N4 MRI Bias Field Correction
Language: C++
Date: $Date: 2012-28-12 14:00:00 +0100 (Fri, 28 Dec 2012) $
Version: $Revision: 1 $
Author: $Sebastien Tourbier$

Copyright (c) 2017 Medical Image Analysis Laboratory (MIAL), Lausanne
  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

==========================================================================*/

#include "itkImage.h"
#include "itkImageMaskSpatialObject.h"
#include "itkExtractImageFilter.h"
#include "itkShrinkImageFilter.h"
#include <itkN4BiasFieldCorrectionImageFilter.h>

#include "itkStatisticsImageFilter.h"
#include "itkAddImageFilter.h"

#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageLinearConstIteratorWithIndex.h"

#include <iostream>
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "float.h"

//#include "limits.h"

int main(int argc, char *argv[])
{
    unsigned int ShrinkFactor = 4;

    if( argc < 5 )
    {
        std::cerr << "Usage: " << std::endl;
        std::cerr << argv[0] << " inputImageFile inputMaskFile outputImageFile outputBiasFile" << std::endl;
        return EXIT_FAILURE;
    }

    // TODO: check if content of last argument is indeed verbose
    bool verbose = false;
    if (argc == 6){
        if (argv[5] == std::string("verbose")){
            verbose = true;
        }
        else{
            throw std::invalid_argument("ERROR: Last parameter ought to be verbose \nUsage: " +
                                        std::string(argv[0]) +
                                        " inputImageFile inputMaskFile outputImageFile outputBiasFile *verbose* ");
        }
    }
 
    const unsigned int dimension3D = 3;
    const unsigned int dimension2D = 2;

    typedef float InputPixelType;
    typedef float OutputPixelType;
    typedef unsigned char MaskPixelType;

    typedef itk::Image<InputPixelType, dimension3D> InputImageType;
    typedef itk::Image<OutputPixelType, dimension3D> OutputImageType;
    typedef itk::Image<MaskPixelType, dimension3D> MaskType;

    typedef itk::ImageMaskSpatialObject< dimension3D > MaskSpatialType;

    typedef InputImageType::RegionType InputRegionType;

    typedef itk::Image<InputPixelType, dimension2D> SliceImageType;
    typedef itk::Image<MaskPixelType, dimension2D> SliceImageMaskType;

    typedef itk::ImageFileReader<InputImageType> ReaderType;
    typedef itk::ImageFileWriter<OutputImageType> WriterType;
    typedef itk::ImageFileReader<MaskType> MaskReaderType;

    typedef itk::ExtractImageFilter<InputImageType, SliceImageType> ExtractImageFilterType;
    typedef itk::ExtractImageFilter<MaskType, SliceImageMaskType> ExtractImageMaskFilterType;

    typedef itk::StatisticsImageFilter<SliceImageType> StatisticsImageFilterType;
    typedef itk::AddImageFilter<SliceImageType, SliceImageType, SliceImageType> AddFilterType;


    //
    if (verbose){
      std::cerr << "Read input image... " << std::endl;
    }
    
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(argv[1]);
    reader->Update();

    InputImageType::Pointer inputImage = reader->GetOutput();

    if (verbose){
      std::cerr << "Read input mask... " << std::endl;
    }
    MaskReaderType::Pointer maskReader = MaskReaderType::New();
    maskReader->SetFileName( argv[2] );
    maskReader->Update();

    MaskType::Pointer maskImage = maskReader->GetOutput();

    maskImage->SetOrigin(inputImage->GetOrigin());
    maskImage->SetSpacing(inputImage->GetSpacing());

    MaskSpatialType::Pointer mask = MaskSpatialType::New();
    mask -> SetImage( maskImage );

    InputRegionType roi = mask -> GetAxisAlignedBoundingBoxRegion();

    OutputImageType::Pointer outImage = OutputImageType::New();
    outImage->SetRegions(inputImage->GetLargestPossibleRegion());
    outImage->Allocate();
    outImage->FillBuffer(0.0);

    outImage->SetOrigin(inputImage->GetOrigin());
    outImage->SetSpacing(inputImage->GetSpacing());
    outImage->SetDirection(inputImage->GetDirection());

    OutputImageType::Pointer logField = OutputImageType::New();
    logField->SetRegions(inputImage->GetLargestPossibleRegion());
    logField->Allocate();
    logField->FillBuffer(0.0);

    logField->SetOrigin(inputImage->GetOrigin());
    logField->SetSpacing(inputImage->GetSpacing());
    logField->SetDirection(inputImage->GetDirection());

    // Extract each slice in the input image and input mask and correct bias field using N4 bias field correction filter

    InputImageType::IndexType inputIndex = roi.GetIndex();
    InputImageType::SizeType  inputSize  = roi.GetSize();

    //TODO: Can we parallelize this ?
    //Iteration over the slices of the LR images

    unsigned int i=inputIndex[2] + inputSize[2];

    //Loop over slices of the current stack
    for ( unsigned int i=inputIndex[2]; i < inputIndex[2] + inputSize[2]; i++ )
    {   
        
        if (verbose){
          std::cout << "Process slice #" << i << "..." << std::endl;
        }

        InputImageType::RegionType wholeSliceRegion;
        wholeSliceRegion = roi;

        InputImageType::IndexType  wholeSliceRegionIndex = wholeSliceRegion.GetIndex();
        InputImageType::SizeType   wholeSliceRegionSize  = wholeSliceRegion.GetSize();

        wholeSliceRegionIndex[2]= i;
        wholeSliceRegionSize[2] = 0;

        wholeSliceRegion.SetIndex(wholeSliceRegionIndex);
        wholeSliceRegion.SetSize(wholeSliceRegionSize);

        //Extract slice in input mask
        ExtractImageMaskFilterType::Pointer sliceMaskExtractor = ExtractImageMaskFilterType::New();
        sliceMaskExtractor->SetExtractionRegion(wholeSliceRegion);
        sliceMaskExtractor->SetInput(maskImage);
#if ITK_VERSION_MAJOR >= 4
        sliceMaskExtractor->SetDirectionCollapseToIdentity(); // This is required.
#endif
        sliceMaskExtractor->Update();

        //Extract slice in input image
        ExtractImageFilterType::Pointer sliceExtractor = ExtractImageFilterType::New();
        sliceExtractor->SetExtractionRegion(wholeSliceRegion);
        sliceExtractor->SetInput(inputImage);
#if ITK_VERSION_MAJOR >= 4
        sliceExtractor->SetDirectionCollapseToIdentity(); // This is required.
#endif
        sliceExtractor->Update();

        if (verbose){
          std::cerr << "Shrink input image... " << std::endl;
        }
        typedef itk::ShrinkImageFilter<SliceImageType, SliceImageType> ShrinkerType;
        ShrinkerType::Pointer shrinker = ShrinkerType::New();

        shrinker->SetInput(sliceExtractor->GetOutput());
        shrinker->SetShrinkFactors(ShrinkFactor);
        shrinker->Update();
        shrinker->UpdateLargestPossibleRegion();

        if (verbose){
          std::cerr << "Shrink input mask... " << std::endl;
        }
        typedef itk::ShrinkImageFilter<SliceImageMaskType, SliceImageMaskType> MaskShrinkerType;
        MaskShrinkerType::Pointer maskShrinker = MaskShrinkerType::New();

        maskShrinker->SetInput(sliceMaskExtractor->GetOutput());
        maskShrinker->SetShrinkFactors(ShrinkFactor);
        maskShrinker->Update();
        maskShrinker->UpdateLargestPossibleRegion();

        //
        if (verbose){
          std::cerr << "Run N4 Bias Field Correction... " << std::endl;
        }
        typedef itk::N4BiasFieldCorrectionImageFilter<SliceImageType,SliceImageMaskType,SliceImageType> CorrecterType;
        CorrecterType::Pointer correcter = CorrecterType::New();

        correcter->SetInput1(shrinker->GetOutput());

        correcter->SetMaskImage(maskShrinker->GetOutput());
        correcter->SetMaskLabel(1);

        unsigned int NumberOfFittingLevels = 3;
        unsigned int NumberOfHistogramBins = 100;

        //With B-spline grid res. = [1, 1, 1]
        CorrecterType::ArrayType NumberOfControlPoints(NumberOfFittingLevels);
        NumberOfControlPoints[0] = NumberOfFittingLevels+1;
        NumberOfControlPoints[1] = NumberOfFittingLevels+1;
        //NumberOfControlPoints[2] = NumberOfFittingLevels+1;

        CorrecterType::VariableSizeArrayType maximumNumberOfIterations(NumberOfFittingLevels);
        maximumNumberOfIterations[0] = 50;
        maximumNumberOfIterations[1] = 40;
        //maximumNumberOfIterations[2] = 30;

        float WienerFilterNoise = 0.01;
        float FullWidthAtHalfMaximum = 0.15;
        float ConvergenceThreshold = 0.0001;

        correcter->SetMaximumNumberOfIterations(maximumNumberOfIterations);
        correcter->SetNumberOfFittingLevels(NumberOfFittingLevels);
        correcter->SetNumberOfControlPoints(NumberOfControlPoints);
        correcter->SetWienerFilterNoise(WienerFilterNoise);
        correcter->SetBiasFieldFullWidthAtHalfMaximum(FullWidthAtHalfMaximum);
        correcter->SetConvergenceThreshold(ConvergenceThreshold);
        correcter->SetNumberOfHistogramBins(NumberOfHistogramBins);
        correcter->Update();

        //
        if (verbose){
          std::cerr << "Extract log field estimated... " << std::endl;
        }
        typedef CorrecterType::BiasFieldControlPointLatticeType PointType;
        typedef CorrecterType::ScalarImageType ScalarImageType;

        typedef itk::BSplineControlPointImageFilter<PointType, ScalarImageType> BSplinerType;
        BSplinerType::Pointer bspliner = BSplinerType::New();
        bspliner->SetInput( correcter->GetLogBiasFieldControlPointLattice() );
        bspliner->SetSplineOrder( correcter->GetSplineOrder() );
        bspliner->SetSize( sliceExtractor->GetOutput()->GetLargestPossibleRegion().GetSize() );
        bspliner->SetOrigin( sliceExtractor->GetOutput()->GetOrigin() );
        bspliner->SetDirection( sliceExtractor->GetOutput()->GetDirection() );
        bspliner->SetSpacing( sliceExtractor->GetOutput()->GetSpacing() );
        bspliner->Update();

        /*
        SliceImageType::Pointer logField = SliceImageType::New();
        logField->SetOrigin(bspliner->GetOutput()->GetOrigin());
        logField->SetSpacing(bspliner->GetOutput()->GetSpacing());
        logField->SetRegions(bspliner->GetOutput()->GetLargestPossibleRegion().GetSize());
        logField->SetDirection(bspliner->GetOutput()->GetDirection());
        logField->Allocate();
        */

        OutputImageType::RegionType wholeSliceRegion3D;
        wholeSliceRegion3D = roi;

        InputImageType::IndexType  wholeSliceRegionIndex3D = wholeSliceRegion3D.GetIndex();
        InputImageType::SizeType   wholeSliceRegionSize3D  = wholeSliceRegion3D.GetSize();

        wholeSliceRegionIndex3D[2]= i;
        wholeSliceRegionSize3D[2] = 1;

        wholeSliceRegion3D.SetIndex(wholeSliceRegionIndex3D);
        wholeSliceRegion3D.SetSize(wholeSliceRegionSize3D);

        itk::ImageRegionIterator<ScalarImageType> ItB(bspliner->GetOutput(),bspliner->GetOutput()->GetLargestPossibleRegion());
        itk::ImageRegionIterator<OutputImageType> ItF(logField,wholeSliceRegion3D);
        //itk::ImageRegionIterator<MaskType> ItM(maskImage,wholeSliceRegion3D);

        //std::cout << bspliner->GetOutput()->GetLargestPossibleRegion() << std::endl;
        //std::cout << wholeSliceRegion3D << std::endl;

        for(ItB.GoToBegin(), ItF.GoToBegin(); !ItB.IsAtEnd(); ++ItB, ++ItF)
        {
            ItF.Set( ItB.Get()[0] );
        }

        //

    }
    if (verbose){
      std::cerr << "Compute the slice corrected... " << std::endl;
    }
    typedef itk::ExpImageFilter<OutputImageType, OutputImageType> ExpFilterType;
    ExpFilterType::Pointer expFilter = ExpFilterType::New();
    expFilter->SetInput( logField );
    expFilter->Update();

    typedef itk::DivideImageFilter<OutputImageType, OutputImageType, OutputImageType> DividerType;
    DividerType::Pointer divider = DividerType::New();
    divider->SetInput1( inputImage );
    divider->SetInput2( expFilter->GetOutput() );
    divider->SetConstant2(1e-6);
    divider->Update();

    // Remove the min value slice by slice
    //Loop over slices in the brain mask
    for ( unsigned int i=inputIndex[2]; i < inputIndex[2] + inputSize[2]; i++ )
    {
        
        if (verbose){
          std::cout << "Process slice #" << i << "..." << std::endl;
        }
        

        InputImageType::RegionType wholeSliceRegion;
        wholeSliceRegion = roi;

        InputImageType::IndexType  wholeSliceRegionIndex = wholeSliceRegion.GetIndex();
        InputImageType::SizeType   wholeSliceRegionSize  = wholeSliceRegion.GetSize();

        wholeSliceRegionIndex[2]= i;
        wholeSliceRegionSize[2] = 0;

        wholeSliceRegion.SetIndex(wholeSliceRegionIndex);
        wholeSliceRegion.SetSize(wholeSliceRegionSize);

        //Extract slice in input mask
        ExtractImageMaskFilterType::Pointer sliceMaskExtractor = ExtractImageMaskFilterType::New();
        sliceMaskExtractor->SetExtractionRegion(wholeSliceRegion);
        sliceMaskExtractor->SetInput(maskImage);
#if ITK_VERSION_MAJOR >= 4
        sliceMaskExtractor->SetDirectionCollapseToIdentity(); // This is required.
#endif
        sliceMaskExtractor->Update();

        //Extract slice in input image
        ExtractImageFilterType::Pointer sliceExtractor = ExtractImageFilterType::New();
        sliceExtractor->SetExtractionRegion(wholeSliceRegion);
        sliceExtractor->SetInput(divider->GetOutput());
#if ITK_VERSION_MAJOR >= 4
        sliceExtractor->SetDirectionCollapseToIdentity(); // This is required.
#endif
        sliceExtractor->Update();

        //Extract min value in the slice contained in the brain mask

        itk::ImageRegionIterator<SliceImageType> ItS(sliceExtractor->GetOutput(),sliceExtractor->GetOutput()->GetLargestPossibleRegion());
        itk::ImageRegionIterator<SliceImageMaskType> ItSM(sliceMaskExtractor->GetOutput(),sliceMaskExtractor->GetOutput()->GetLargestPossibleRegion());

        float minValue = FLT_MAX;

        for( ItS.GoToBegin(), ItSM.GoToBegin(); !ItS.IsAtEnd(); ++ItS, ++ItSM)
        {
            if(ItSM.Get() > 1e-2 && ItS.Get() > 1e-1)
            {
                if(ItS.Get() <= minValue) minValue = ItS.Get();
            }
        }
        if (verbose){
          std::cout << "min : " << minValue << std::endl;
        }
        //REMOVE THE MIN VALUE and set the new value in the output image (if contained in the brain mask, otherwise value set to zero)

        OutputImageType::RegionType wholeSliceRegion3D;
        wholeSliceRegion3D = roi;

        InputImageType::IndexType  wholeSliceRegionIndex3D = wholeSliceRegion3D.GetIndex();
        InputImageType::SizeType   wholeSliceRegionSize3D  = wholeSliceRegion3D.GetSize();

        wholeSliceRegionIndex3D[2]= i;
        wholeSliceRegionSize3D[2] = 1;

        wholeSliceRegion3D.SetIndex(wholeSliceRegionIndex3D);
        wholeSliceRegion3D.SetSize(wholeSliceRegionSize3D);

        itk::ImageRegionIterator<OutputImageType> ItO(outImage,wholeSliceRegion3D);

        for( ItS.GoToBegin(), ItSM.GoToBegin(), ItO.GoToBegin(); !ItS.IsAtEnd(); ++ItS, ++ItSM, ++ItO)
        {
            if((ItS.Get() - minValue) > 0.0)
            {
                ItO.Set( (ItS.Get() - minValue) );
                //std::cout << " substract.. " << std::endl;
            }
            else
            {
                ItO.Set(0);
            }
        }



    }

    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(argv[3]);
    writer->SetInput(outImage);
    writer->Update();

    WriterType::Pointer fieldWriter = WriterType::New();
    fieldWriter->SetFileName(argv[4]);
    fieldWriter->SetInput(logField);
    fieldWriter->Update();

    /*
     *
    MaskIteratorTypeWithIndex itOutStackMask(outImageMasks[s],outImageMasks[s]->GetLargestPossibleRegion());
    for(itOutStackMask.GoToBegin(); !itOutStackMask.IsAtEnd(); ++itOutStackMask)
    {
        if(itOutStackMask.Get()>0.0) itOutStackMask.Set(1.0);
    }

    //std::stringstream ssFile;
    //ssFile << "/home/tourbier/Desktop/DbgMasks/hrResMask_" << s << ".nii.gz";

    //

    std::cerr << "Write ouput images ... " << std::endl;

    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(argv[3]);
    writer->SetInput(divider->GetOutput());
    writer->Update();

    WriterType::Pointer fieldWriter = WriterType::New();
    fieldWriter->SetFileName(argv[4]);
    fieldWriter->SetInput(logField);
    fieldWriter->Update();

    */
    if (verbose){
      std::cerr << "Done! " << std::endl;
    }
    return 0;
}



