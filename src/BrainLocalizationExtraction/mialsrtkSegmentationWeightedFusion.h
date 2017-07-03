/*==========================================================================
  © 

  Program: 2D Brain Segmentation with Geodesic Active Contours (header)
  Language: C++
  Date: $Date: 2013-08-22 $
  Version: $Revision: 1.4 $
  Author: $Sebastien Tourbier$

==========================================================================*/
#ifndef STKSEGMENTATIONWEIGHTEDFUSION_H
  #define STKSEGMENTATIONWEIGHTEDFUSION_H
#endif

/* Standard includes */
#include <tclap/CmdLine.h>
#include <iostream>     // std::cout
#include <sstream>
#include <limits>       // std::numeric_limits
#include <vector>
#include <cmath> // std::abs
#include <time.h>

#include "mialsrtkMaths.h"
#include "itkImage.h"
#include "itkPoint.h"

#include "itkBinaryThresholdImageFilter.h"
#include "itkOtsuThresholdImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkNormalizedCorrelationImageToImageMetric.h"

#include "itkAffineTransform.h"
#include "itkImageDuplicator.h"
#include "itkCastImageFilter.h"
#include "itkMaskImageFilter.h"

#include "itkImageRegionIterator.h"
#include "itkConstNeighborhoodIterator.h"

#include "itkNormalizedCorrelationImageFilter.h"
#include "ComputeNormalizedCrossCorrelationImageFilter.h"

#include "itkConnectedComponentImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkMinimumMaximumImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "itkVotingBinaryHoleFillingImageFilter.h"


/************************************************************************/
#define MAX_CHAR_LENGTH  100

  const bool debug = false;
  const std::string intermediateSaveDirectory = "/home/tourbier/Desktop/Debug/";

  const unsigned int dim2D = 2;
  const unsigned int dim3D = 3;

  //Thresholder parameters
  const double lower = 128.0;
  const double upper = 255.0;
  const double insideValue = 0.0;
  const double outsideValue = 255.0;
  const unsigned int numberOfBins = 128;
  const unsigned int minimumObjectSize = 40;

  const unsigned int radiusX = 15;
  const unsigned int radiusY = 15;
  const unsigned int radiusZ = 1;
 
  //////////////////////////////////////////////////////////
    // typedef
    // Pixel Type
    typedef float InputPixelType;
    typedef float OutputPixelType;
    typedef unsigned long InternalPixelType;
    typedef unsigned char MaskPixelType;

    // Mask Type
    typedef itk::Image<MaskPixelType, dim3D> InputMaskType;
    typedef itk::Image<MaskPixelType, dim2D> InternalMaskType;
    typedef itk::Image<MaskPixelType, dim3D> OutputMaskType;

    //Image Type
    typedef itk::Image<InputPixelType, dim3D> InputImageType;
    //typedef itk::Image<InputPixelType, dim2D> InternalImageType;
    typedef itk::Image<InternalPixelType, dim3D> InternalImageType;
    typedef itk::Image<OutputPixelType, dim3D> OutputImageType;

    typedef itk::CastImageFilter< InputMaskType,InputImageType > CastFilterType;
    typedef itk::ImageDuplicator< InputImageType > ImageDuplicatorType;
    typedef itk::ImageDuplicator< InputMaskType > MaskDuplicatorType;
        
    // Reader and Writer Type
    typedef itk::ImageFileReader<InputImageType> ReaderType;
    typedef itk::ImageFileReader<InputMaskType> MaskReaderType;

    typedef itk::ImageFileWriter<OutputImageType> WriterType;
    typedef itk::ImageFileWriter<OutputMaskType> MaskWriterType;

    typedef itk::NormalizedCorrelationImageToImageMetric<InputImageType,InputImageType> SimilarityMetricType;
    typedef SimilarityMetricType::TransformType TransformBaseType;
    typedef TransformBaseType::ParametersType ParametersType;

    typedef itk::ComputeNormalizedCrossCorrelationImageFilter< InputImageType,InputImageType > ComputeNormalizedCrossCorrelationFilterType;

    typedef itk::AffineTransform <double,dim3D> TransformType;

    //typedef itk::LinearInterpolateImageFunction<InputImageType, double> InterpolatorType;
    typedef itk::BSplineInterpolateImageFunction<InputImageType, double> InterpolatorType;
    

    typedef itk::MaskImageFilter<InputImageType,InputMaskType,InputImageType> MaskFilterType;

    //ITK filter type
    typedef itk::BinaryThresholdImageFilter<InputImageType,OutputMaskType> ThresholdingFilterType;
    typedef itk::OtsuThresholdImageFilter<InputImageType,InputImageType> OtsuThresholdingFilterType;
    
    // basic iterator type
    typedef itk::ImageRegionIterator<InputImageType> InputImageIteratorType;
    typedef itk::ImageRegionIterator<InputMaskType> InputMaskIteratorType;

    // Patch iterator type
    typedef itk::ConstNeighborhoodIterator<InputImageType> InputImageNeighborhoodIteratorType;

    typedef itk::ConnectedComponentImageFilter<InputImageType, InternalImageType >  CCFilterType;
    typedef itk::RelabelComponentImageFilter<InternalImageType, OutputMaskType > RelabelType;

    typedef itk::VotingBinaryHoleFillingImageFilter<OutputMaskType, OutputMaskType > FillHolesFilterType;
    //typedef itk::NormalizedCorrelationImageToImageMetric<InputImageNeighborhoodIteratorType::NeighborhoodType, InputImageNeighborhoodIteratorType::NeighborhoodType> NeighborhoodSimilarityMetricType;

    //typedef itk::LinearInterpolateImageFunction<InputImageNeighborhoodIteratorType::NeighborhoodType, double> NeighborhoodInterpolatorType;

//////////////////////////////////////////////////////////
// Function prototypes
int main( int argc, char * argv [] );

void prompt_start(std::vector< std::string > & inputFileNames, std::vector< std::string > & inputRegFileNames, std::vector< std::string > & maskFileNames, const char* outputFileName, unsigned int & method);

void majorityVoting(std::vector<unsigned char> & values, unsigned char & outputValue);
void majorityVoting(std::vector<float> & values, double & outputValue);

void localWeightedVoting(int patchRadius, int numberOfImages,InputImageNeighborhoodIteratorType &targetImageIt, InputImageIteratorType &outputMaskIt, InputMaskIteratorType* &templateRegMasksIts, InputImageNeighborhoodIteratorType* &templateRegImagesIts);
