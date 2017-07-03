/*=========================================================================

Program: MRI bias field correction using N4
Language: C++
Date: $Date: 2012-28-12 $
Version: $Revision: 1 $
Author: $Sebastien Tourbier$

Copyright (c) 2017 Medical Image Analysis Laboratory (MIAL), Lausanne
  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
     

==========================================================================*/


#include "itkImage.h"
#include "itkPoint.h"

#include "itkAddImageFilter.h"
#include "itkDivideImageFilter.h"
#include "itkStatisticsImageFilter.h"

#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkImageRegionIterator.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

int main( int argc, char * argv [] )
{

  if ( argc < 3 )
    {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " inputImageFile inputBiasFieldImageFile outputCorrectedImageFile  ";
    return EXIT_FAILURE;
    }

  const unsigned int dimension = 3; 
  
  typedef float InputPixelType;
  typedef float OutputPixelType;

  typedef itk::Image<InputPixelType, dimension> InputImageType;
  typedef itk::Image<OutputPixelType, dimension> OutputImageType;
  
  typedef itk::DivideImageFilter<InputImageType, InputImageType, InputImageType> itkDivideFilter;
  typedef itk::StatisticsImageFilter<InputImageType> itkStatisticsImageFilter;
  typedef itk::AddImageFilter<InputImageType, InputImageType, InputImageType> itkAddFilter;
      
  typedef itk::ImageFileReader<InputImageType> ReaderType;
  typedef itk::ImageFileWriter<OutputImageType> WriterType;
  
	//

  std::cerr << "Read input image... " << std::endl;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(argv[1]);
  try
  {
    reader->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
  }
  
  InputImageType::Pointer inputImage = reader->GetOutput();

  std::cout << "Input image: " << inputImage << std::endl;


  std::cerr << "Read bias estimated... " << std::endl;
  ReaderType::Pointer biasReader = ReaderType::New();
  biasReader->SetFileName( argv[2] );
  
  try
  {
    biasReader->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
  }


  InputImageType::Pointer biasImage = biasReader->GetOutput();
  
  biasImage->SetOrigin(inputImage->GetOrigin());
  biasImage->SetSpacing(inputImage->GetSpacing());
  biasImage->SetDirection(inputImage->GetDirection());
  biasImage->Update();

	std::cout<<"Compute the bias-corrected volume \n";
  itkDivideFilter::Pointer divide = itkDivideFilter::New();
  divide->SetInput1(inputImage);   
  divide->SetInput2(biasImage);
  divide->Update();
  OutputImageType::Pointer outputImage = divide->GetOutput();
  
//  WriterType::Pointer dbgWriter = WriterType::New();
//  dbgWriter->SetFileName("/home/tourbier/Desktop/F022_DiffBiasCorr/F022_orig_masked_crop_bcorr_interm.nii");
//  dbgWriter->SetInput(outputImage);
//  dbgWriter->Update();
  
  //Extract the min value of the inputImage and remove it
  itkStatisticsImageFilter::Pointer statisticsImageFilter = itkStatisticsImageFilter::New ();
  statisticsImageFilter->SetInput(inputImage);
  statisticsImageFilter->Update();
  
  InputPixelType minValue = statisticsImageFilter->GetMinimum();
  
  std::cerr << "Remove the min value : " << minValue << std::endl;
      
  itkAddFilter::Pointer add = itkAddFilter::New();
  add->SetInput1(outputImage);
  add->SetConstant2(-minValue);
  add->Update();
  outputImage = add->GetOutput(); 
  
  //

  std::cerr << "Write the bias-corrected volume... " << std::endl;

	WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(argv[3]);
  writer->SetInput(outputImage);
  writer->Update();

  std::cerr << "Done! " << std::endl;
  
  return EXIT_SUCCESS;
}
