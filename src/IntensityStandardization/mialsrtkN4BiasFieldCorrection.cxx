/*=========================================================================

Program: N4 MRI Bias Field Correction
Language: C++
Date: $Date: 2012-28-12 14:00:00 +0100 (Fri, 28 Dec 2012) $
Version: $Revision: 1 $
Author: $Sebastien Tourbier$

Copyright (c) 2017 Medical Image Analysis Laboratory (MIAL), Lausanne
  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

==========================================================================*/

#include "itkImage.h"
#include "itkShrinkImageFilter.h"
#include <itkN4BiasFieldCorrectionImageFilter.h>

#include <iostream>
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkEuler3DTransform.h"

#include "itkResampleImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"

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
    verbose = true;
  }
 
  const unsigned int dimension = 3; 
  
  typedef float InputPixelType;
  typedef float OutputPixelType;
  typedef float MaskPixelType;

  typedef itk::Image<InputPixelType, dimension> InputImageType;
  typedef itk::Image<OutputPixelType, dimension> OutputImageType;
  typedef itk::Image<MaskPixelType, dimension> MaskType;
    
  typedef itk::ImageFileReader<InputImageType> ReaderType;
  typedef itk::ImageFileWriter<OutputImageType> WriterType;
  typedef itk::ImageFileReader<MaskType> MaskReaderType;


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

  typedef itk::ResampleImageFilter<MaskType,MaskType> ResampleImageMaskFilterType;
  typedef itk::NearestNeighborInterpolateImageFunction<MaskType> NNInterpolatorType;

  ResampleImageMaskFilterType::Pointer maskUpsampler = ResampleImageMaskFilterType::New();
  NNInterpolatorType::Pointer nnInterpolator = NNInterpolatorType::New();

  typedef  itk::Euler3DTransform<double> Euler3DTransformType;
  Euler3DTransformType::Pointer idTransform = Euler3DTransformType::New();
  idTransform->SetIdentity();

  maskUpsampler -> SetInterpolator(nnInterpolator);
  maskUpsampler -> SetInput(maskReader->GetOutput());
  maskUpsampler -> SetTransform(idTransform);
  maskUpsampler -> SetOutputParametersFromImage(inputImage.GetPointer());
  maskUpsampler -> SetOutputSpacing(inputImage->GetSpacing());
  maskUpsampler -> SetSize(inputImage->GetLargestPossibleRegion().GetSize());

  maskUpsampler -> Update();

  MaskType::Pointer maskImage = maskUpsampler->GetOutput();

	//
  if (verbose){
    std::cerr << "Shrink input image... " << std::endl;
  }
  typedef itk::ShrinkImageFilter<InputImageType, InputImageType> ShrinkerType;
  ShrinkerType::Pointer shrinker = ShrinkerType::New();
  
  shrinker->SetInput(inputImage);
  shrinker->SetShrinkFactors(ShrinkFactor);
  shrinker->Update();
  shrinker->UpdateLargestPossibleRegion();

  if (verbose){
    std::cerr << "Shrink input mask... " << std::endl;
  }
  typedef itk::ShrinkImageFilter<MaskType, MaskType> MaskShrinkerType;
  MaskShrinkerType::Pointer maskShrinker = MaskShrinkerType::New();
  
  maskShrinker->SetInput(maskImage);
  maskShrinker->SetShrinkFactors(ShrinkFactor);
  maskShrinker->Update();
  maskShrinker->UpdateLargestPossibleRegion();



	//
  if (verbose){
	  std::cerr << "Run N4 Bias Field Correction... " << std::endl;
  }
  typedef itk::N4BiasFieldCorrectionImageFilter<InputImageType,MaskType,InputImageType> CorrecterType;
  CorrecterType::Pointer correcter = CorrecterType::New();
  
  correcter->SetInput1(shrinker->GetOutput());
 
  correcter->SetMaskImage(maskShrinker->GetOutput());
  correcter->SetMaskLabel(1);

  unsigned int NumberOfFittingLevels = 3;
  unsigned int NumberOfHistogramBins = 200;

  //With B-spline grid res. = [1, 1, 1]
  CorrecterType::ArrayType NumberOfControlPoints(NumberOfFittingLevels);
  NumberOfControlPoints[0] = NumberOfFittingLevels+1;
  NumberOfControlPoints[1] = NumberOfFittingLevels+1;
  NumberOfControlPoints[2] = NumberOfFittingLevels+1;

  CorrecterType::VariableSizeArrayType maximumNumberOfIterations(NumberOfFittingLevels); 
  maximumNumberOfIterations[0] = 50;
  maximumNumberOfIterations[1] = 40;
  maximumNumberOfIterations[2] = 30;

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
  bspliner->SetSize( inputImage->GetLargestPossibleRegion().GetSize() );
  bspliner->SetOrigin( inputImage->GetOrigin() );
  bspliner->SetDirection( inputImage->GetDirection() );
  bspliner->SetSpacing( inputImage->GetSpacing() );
  bspliner->Update();

  InputImageType::Pointer logField = InputImageType::New();
  logField->SetOrigin(bspliner->GetOutput()->GetOrigin());
  logField->SetSpacing(bspliner->GetOutput()->GetSpacing());
  logField->SetRegions(bspliner->GetOutput()->GetLargestPossibleRegion().GetSize());
  logField->SetDirection(bspliner->GetOutput()->GetDirection());
  logField->Allocate();

  itk::ImageRegionIterator<ScalarImageType> ItB(bspliner->GetOutput(),bspliner->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionIterator<InputImageType> ItF(logField,logField->GetLargestPossibleRegion());
  
  for(ItB.GoToBegin(), ItF.GoToBegin(); !ItB.IsAtEnd(); ++ItB, ++ItF)
  {
    ItF.Set( ItB.Get()[0] );
  }



	//
  if (verbose){
    std::cerr << "Compute the image corrected... " << std::endl;
  }
  typedef itk::ExpImageFilter<InputImageType, InputImageType>
  ExpFilterType;
  ExpFilterType::Pointer expFilter = ExpFilterType::New();
  expFilter->SetInput( logField );
  expFilter->Update();

  typedef itk::DivideImageFilter<InputImageType, InputImageType, InputImageType> DividerType;
  DividerType::Pointer divider = DividerType::New();
  divider->SetInput1( inputImage );
  divider->SetInput2( expFilter->GetOutput() );
  divider->Update();



	//
  if (verbose){
    std::cerr << "Write ouput images ... " << std::endl;
  }
	WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(argv[3]);
  writer->SetInput(divider->GetOutput());
  writer->Update();

	WriterType::Pointer fieldWriter = WriterType::New();
  fieldWriter->SetFileName(argv[4]);
  fieldWriter->SetInput(logField);
	fieldWriter->Update();
  if (verbose){
    std::cerr << "Done! " << std::endl;
  }
  return 0;
}



