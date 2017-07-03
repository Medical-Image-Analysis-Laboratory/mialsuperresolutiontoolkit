/*=========================================================================

Program: Compute Haussdorf distance between two segmentations
Language: C++
Date: $Date$
Version: 1.0
Author: Sebastien Tourbier

Copyright (c) 2017 Medical Image Analysis Laboratory (MIAL), Lausanne
  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
     
=========================================================================*/
#include <iostream>
#include <string>
#include <stdlib.h>

#include "itkImageFileReader.h"
#include "itkImage.h"

#include "itkChangeInformationImageFilter.h"

#include "itkStatisticsImageFilter.h"
#include <itkDirectedHausdorffDistanceImageFilter.h>

#include "itkMultiThreader.h"

#include "vcl_algorithm.h"



int main(int argc, char *argv[])
{

  if( argc < 2 )
   {
      std::cerr << "Usage: " << std::endl;
      std::cerr << argv[0] << " image1.nii image2.nii" << std::endl;
      return EXIT_FAILURE;
   }
 
  const unsigned int dimension = 3; 
  
  typedef unsigned char InputPixelType;

  typedef itk::Image<InputPixelType, dimension> InputImageType;
  typedef itk::ImageFileReader<InputImageType> ReaderType;


  //Read the images 
  //
  //
  //std::cout << "Load first image... " << std::endl;
  ReaderType::Pointer imReader = ReaderType::New();
  imReader->SetFileName(argv[1]);
  try
  {
    imReader->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }
  InputImageType::Pointer image = imReader->GetOutput();
   
  //std::cout << "Load second image... " << std::endl;
  ReaderType::Pointer imReader2 = ReaderType::New();
  imReader2->SetFileName(argv[2]);
  try
  {
     imReader2->Update();
  }
  catch( itk::ExceptionObject & err )
  {
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
  }
  InputImageType::Pointer image2 = imReader2->GetOutput();
  
  typedef itk::ChangeInformationImageFilter< InputImageType >  ChangeInfoFilterType;
  ChangeInfoFilterType::Pointer changeFilter2 = ChangeInfoFilterType::New();
  
  changeFilter2->SetOutputDirection(imReader->GetOutput()->GetDirection());
  changeFilter2->ChangeDirectionOn();
  changeFilter2->SetOutputOrigin(imReader->GetOutput()->GetOrigin());
  changeFilter2->ChangeOriginOn();
  changeFilter2->SetOutputSpacing(imReader->GetOutput()->GetSpacing());
  changeFilter2->ChangeSpacingOn();
  
  changeFilter2->SetInput(imReader2->GetOutput());
  
  try
  {
     changeFilter2->Update();
  }
  catch( itk::ExceptionObject & err )
  {
     std::cerr << err << std::endl;
     return EXIT_FAILURE;
  }

  typedef itk::DirectedHausdorffDistanceImageFilter<InputImageType,InputImageType> DirectedHausdorffDistanceImageFilterType;
  DirectedHausdorffDistanceImageFilterType::Pointer distanceFilter = DirectedHausdorffDistanceImageFilterType::New();

  distanceFilter->SetInput1(imReader->GetOutput());
  distanceFilter->SetInput2(changeFilter2->GetOutput());
  
  //observe(diceCoefFilter.GetPointer());
    
  try
  {
    distanceFilter->Update();

    std::cout << distanceFilter->GetDirectedHausdorffDistance() << std::endl;

  }
  catch (itk::ExceptionObject & err)
  {
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }
  
  //std::cout << "Done! " << std::endl;
  return 0;
}

