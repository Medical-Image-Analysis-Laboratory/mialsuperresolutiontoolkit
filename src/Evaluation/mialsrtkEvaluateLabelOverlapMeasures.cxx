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
#include "itkImage.h"

#include "itkChangeInformationImageFilter.h"

#include "itkStatisticsImageFilter.h"
#include <itkLabelOverlapMeasuresImageFilter.h>

#include "itkImageRegionConstIterator.h"

#include "itkMultiThreader.h"

#include "vcl_algorithm.h"



int main(int argc, char *argv[])
{

  const char *inputFileName1 = NULL;
  const char *inputFileName2 = NULL;
  const char *patientName = NULL;
  const char *stackName = NULL;
  const char *atlasName = NULL;
  const char *outputFileName = NULL;

  // Parse arguments

  TCLAP::CmdLine cmd("Computes overlap metrics to evaluate segmentation quality (input) with respect to ground-truth segmentation (ref) - Metrics can be saved in a CSV file (--output-csv) - Assumes a binary image (Does not consider multi labels for the moment)", ' ', "Unversioned");

  TCLAP::ValueArg<std::string> input1Arg  ("r","ref","Input image file 1 (Reference)",true,"","string",cmd);
  TCLAP::ValueArg<std::string> input2Arg  ("i","input","Input image file 2",true,"","string",cmd);
  TCLAP::ValueArg<std::string> patientArg  ("p","patient-name","Patient name",false,"undefined","string",cmd);
  TCLAP::ValueArg<std::string> stackArg  ("s","stack-name","Stack name",false,"undefined","string",cmd);
  TCLAP::ValueArg<std::string> atlasArg  ("a","atlas-name","Atlas name",false,"undefined","string",cmd);
  TCLAP::ValueArg<std::string> outArg  ("o","output-csv","Output csv file",false,"undefined","string",cmd);

  // Parse the argv array.
  cmd.parse( argc, argv );

  inputFileName1 = input1Arg.getValue().c_str();
  inputFileName2 = input2Arg.getValue().c_str();
  patientName = patientArg.getValue().c_str();
  stackName = stackArg.getValue().c_str();
  atlasName = atlasArg.getValue().c_str();
  outputFileName = outArg.getValue().c_str();

 
  const unsigned int dimension = 3; 
  
  typedef unsigned char InputPixelType;

  typedef itk::Image<InputPixelType, dimension> InputImageType;
  typedef itk::ImageFileReader<InputImageType> ReaderType;

  //Read the images 
  //
  //
  std::cout << "Load first image... " << std::endl;
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
  InputImageType::Pointer image = imReader->GetOutput();
   
  std::cout << "Load second image... " << std::endl;
  ReaderType::Pointer imReader2 = ReaderType::New();
  imReader2->SetFileName(inputFileName2);
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

  //Computes TP, FP, TN, FN by iterating over the mask voxels
  typedef itk::ImageRegionConstIterator<InputImageType> IteratorType;
  
  IteratorType refIt(imReader->GetOutput(),imReader->GetOutput()->GetLargestPossibleRegion());
  IteratorType targetIt(changeFilter2->GetOutput(),changeFilter2->GetOutput()->GetLargestPossibleRegion());

  double TP = 0.0; //True positive
  double FP = 0.0; //False positive
  double TN = 0.0; //True negative
  double FN = 0.0; //False negative

  for(refIt.GoToBegin(), targetIt.GoToBegin(); !refIt.IsAtEnd(); ++refIt, ++targetIt)
  {
    if( (refIt.Get() == 1.0) && (targetIt.Get() == 1.0) )
    {
      TP++;
    }
    else if( (refIt.Get() == 1.0) && (targetIt.Get() != 1.0) )
    {
      FN++;
    }
    else if( (refIt.Get() != 1.0) && (targetIt.Get() == 1.0) )
    {
      FP++;
    }
    else if( (refIt.Get() != 1.0) && (targetIt.Get() != 1.0) )
    {
      TN++;
    }
  }

  std::cout << "True positive: " << TP << std::endl;
  std::cout << "False positive: " << FP << std::endl;
  std::cout << "True negative: " << TN << std::endl;
  std::cout << "False negative: " << FN << std::endl;


  double dice = (2 * TP) / (2 * TP + FN + FP);// Dice coefficient
  double sensitivity = TP / (TP + FN);// SEN =TPR=TP/P=TP/(TP+FN)
  double specificity = TN / (FP + TN);//    SPC = TN / N = TN / (FP +TN)
  double precision = TP / (TP + FP);//     \mathit{PPV} = \mathit{TP} / (\mathit{TP} + \mathit{FP}) precision as positive predictive value
  double recall = TP / (TP + FN);//     \mathit{RECALL} = \mathit{TP} / (\mathit{TP} + \mathit{FN})
  double accuracy = (TP + TN) / (TP+FP+FN+TN);//     \mathit{ACC} = (\mathit{TP} + \mathit{TN}) / (P + N)

  double dice2 = 2 * ((precision*recall)/(precision+recall));// Dice coefficient = 2 * ((precision*recall)/(precision+recall))

  //Computes proportion of under/over segmentation in complement to common sensitivity and specificity
  float extraFraction1 = FP / (TP + FN);// Normalized by the number of voxels in GT
  float extraFraction2 = FP / (TP + FP);// Normalized by the number of voxels in the segmentation
  float missFraction = FN / (TP + FN);
  
  std::cout << "Quality measures: " << std::endl;
  std::cout << " " << std::endl;
  std::cout << "True positive: " << TP << std::endl;
  std::cout << "True negativer: " << TN << std::endl;
  std::cout << "False positive: " << FP << std::endl;
  std::cout << "False negative: " << FN << std::endl;
  std::cout << "Dice coefficient: " << dice << std::endl;
  std::cout << "Dice coefficient 2: " << dice2 << std::endl;
  std::cout << "Sensitivity: " << sensitivity << std::endl;
  std::cout << "Specificity: " << specificity << std::endl;
  std::cout << "ExtraFraction1 (GT): " << extraFraction1 << std::endl;
  std::cout << "ExtraFraction2 (SEG): " << extraFraction2 << std::endl;
  std::cout << "MissFraction (GT): " << missFraction << std::endl;
  std::cout << "Precision: " << precision << std::endl;
  std::cout << "Recall: " << recall << std::endl;
  std::cout << "Accuracy: " << accuracy << std::endl;
  std::cout << " " << std::endl;

  if( strncmp( outputFileName , "undefined" , sizeof(outputFileName) - 1) )
  {
    std::cout << "Write evaluation to file" << outputFileName << std::endl;
    std::ofstream fout(outputFileName, std::ios_base::out | std::ios_base::app);
    fout << patientName << ',' << stackName << ',' << atlasName << ',';
    fout << TP << ','<< TN << ','<< FP << ','<< FN << ',';
    fout << dice << ',' << sensitivity << ',' << specificity << ',';
    fout << extraFraction1 <<',' << extraFraction2 <<',' << missFraction <<',' << precision << ',' << recall << ',' << accuracy <<std::endl;
    fout.close();
  }


  
  std::cout << "Done! " << std::endl;
  return 0;
}

