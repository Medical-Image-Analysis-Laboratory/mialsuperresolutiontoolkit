
/*=========================================================================

Program: Performs Scattered Data Interpolation from a set of LR images, applying transforms
that are computed in mialsrtkImageReconstruction.
This program is a stripped down version of mialsrtkImageReconstruction
Language: C++
Date: $Date: 2022-07-12 14:00:00 +0100 (12 Jul 2022) $
Version: $Revision: 1 $
Author: $ $Priscille de Dumast$

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
#include "itkNormalizedCorrelationImageToImageMetric.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkEuler3DTransform.h"
#include "itkTransformFileWriter.h"
#include "itkImage.h"
#include "itkImageMaskSpatialObject.h"
#include "itkCastImageFilter.h"
#include "itkImageDuplicator.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"


#include "itkCurvatureAnisotropicDiffusionImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "itkVersorRigid3DTransform.h"

/*Btk includes*/
//#include "btkSliceBySliceTransform.h"
//#include "btkEulerSliceBySliceTransform.h"
//#include "btkSliceBySliceTransformBase.h"
//#include "btkLowToHighImageResolutionMethod.h"


//#include "btkImageIntersectionCalculator.h"

/* mialsrtk includes */
//#include "mialsrtkLowToHighImageResolutionMethod.h"

#include "mialsrtkVersorSliceBySliceTransform.h"
#include "mialsrtkSliceBySliceTransformBase.h"
#include "mialsrtkLowToHighImageResolutionMethod.h"

#include "mialsrtkImageIntersectionCalculator.h"

#include "mialsrtkResampleImageByInjectionFilter.h"
#include "mialsrtkSliceBySliceRigidRegistration.h"
#include "mialsrtkBSplineInterpolateImageFunction.h"



int main( int argc, char *argv[] )
{

  try {

  std::vector< std::string > input;
  std::vector< std::string > mask;
  std::vector< std::string > transform;
  std::vector< std::string > roi;
  std::vector< std::string > resampled;
  unsigned int itMax;
  double epsilon;
  double margin;

  const char *outImage = NULL;
  const char *combinedMask = NULL;

  std::string refImage;

  // Parse arguments

  TCLAP::CmdLine cmd("Creates a high resolution image from a set of low "
      "resolution images", ' ', "Unversioned");

  TCLAP::MultiArg<std::string> inputArg("i","input","Image file",true,"string",cmd);
  TCLAP::MultiArg<std::string> maskArg("m","","Mask file",false,"string",cmd);
  TCLAP::MultiArg<std::string> transformArg("t","transform","Transform output "
      "file",false,"string",cmd);
  //  TCLAP::MultiArg<std::string> roiArg("","roi","roi file (written as mask)",false,
  //     "string",cmd);
  TCLAP::ValueArg<std::string> outArg("o","output","High resolution image",true,
      "","string",cmd);
  TCLAP::ValueArg<std::string> refArg("r","reference","Reference Image",false, "","string",cmd);
  TCLAP::SwitchArg  verboseArg("v","verbose","Verbose output (False by default)",cmd, false);

  /*TCLAP::SwitchArg  boxSwitchArg("","box","Use intersections for roi calculation",false);
  TCLAP::SwitchArg  maskSwitchArg("","mask","Use masks for roi calculation",false);
  TCLAP::SwitchArg  allSwitchArg("","all","Use the whole image FOV",false);

  std::vector<TCLAP::Arg*>  xorlist;
  xorlist.push_back(&boxSwitchArg);
  xorlist.push_back(&maskSwitchArg);
  xorlist.push_back(&allSwitchArg);

  cmd.xorAdd( xorlist );
  */

  // Parse the argv array.
  cmd.parse( argc, argv );

  input = inputArg.getValue();
  mask = maskArg.getValue();
  transform = transformArg.getValue();

  refImage = refArg.getValue();
  outImage = outArg.getValue().c_str();
  bool verbose = verboseArg.getValue();
  //roi = roiArg.getValue();

    // typedefs

  const    unsigned int    Dimension = 3;
  typedef  float           PixelType;

  typedef itk::Image< PixelType, Dimension >  ImageType;
  typedef ImageType::Pointer                  ImagePointer;

  typedef ImageType::RegionType               RegionType;
  typedef std::vector< RegionType >           RegionArrayType;

  typedef itk::ImageFileReader< ImageType  >  ImageReaderType;



  typedef mialsrtk::SliceBySliceTransformBase< double, Dimension > TransformBaseType;
  typedef mialsrtk::SliceBySliceTransform< double, Dimension > TransformType;
  typedef TransformType::Pointer                          TransformPointer;

  // Register the SliceBySlice transform (a non-default ITK transform) with the TransformFactory of ITK
  itk::TransformFactory<TransformType>::RegisterTransform();

  typedef itk::TransformFileReader     TransformReaderType;
  typedef TransformReaderType::TransformListType * TransformListType;


  typedef itk::Image< unsigned char, Dimension >    ImageMaskType;
  typedef ImageMaskType::Pointer                    ImageMaskPointer;

  typedef itk::ImageFileReader< ImageMaskType >     MaskReaderType;
  typedef itk::ImageMaskSpatialObject< Dimension >  MaskType;
  typedef MaskType::Pointer  MaskPointer;



  /* Registration type required in case of slice by slice transformations
  A rigid transformation is employed because there is not distortions like
  in diffusion imaging. We have performed a comparison of accuracy between
  both types of transformations. */
  typedef mialsrtk::SliceBySliceRigidRegistration<ImageType> RegistrationType;
  typedef RegistrationType::Pointer RegistrationPointer;

  // Registration type required in case of 3D affine trasforms
  typedef mialsrtk::RigidRegistration<ImageType> Rigid3DRegistrationType;
  typedef Rigid3DRegistrationType::Pointer Rigid3DRegistrationPointer;

  // Slice by slice transform definition (typically for in utero reconstructions)
  typedef mialsrtk::SliceBySliceTransformBase< double, Dimension > TransformBaseType;
  typedef mialsrtk::SliceBySliceTransform< double, Dimension > TransformType;
  typedef TransformType::Pointer                          TransformPointer;



  // Rigid 3D transform definition (typically for reconstructions in adults)
  //typedef itk::Euler3DTransform< double > Rigid3DTransformType;
  typedef itk::VersorRigid3DTransform< double > Rigid3DTransformType;
  typedef Rigid3DTransformType::Pointer   Rigid3DTransformPointer;

  // This filter does a rigid registration over all the LR images and compute the average image in HR space
  //typedef mialsrtk::LowToHighImageResolutionMethod<ImageType,Rigid3DTransformType > LowToHighResFilterType;
  typedef mialsrtk::LowToHighImageResolutionMethod<ImageType,Rigid3DTransformType > LowToHighResFilterType;
  LowToHighResFilterType::Pointer lowToHighResFilter = LowToHighResFilterType::New();

  // Resampler type required in case of a slice by slice transform
  typedef mialsrtk::ResampleImageByInjectionFilter< ImageType, ImageType
                                               >  ResamplerType;

  typedef itk::NormalizedCorrelationImageToImageMetric< ImageType,
                                                        ImageType > NCMetricType;

  typedef mialsrtk::ImageIntersectionCalculator<ImageType> IntersectionCalculatorType;
  IntersectionCalculatorType::Pointer intersectionCalculator = IntersectionCalculatorType::New();

  typedef itk::CastImageFilter<ImageType,ImageMaskType> CasterType;
  typedef itk::ImageDuplicator<ImageType> DuplicatorType;

  // Filter setup
  unsigned int numberOfImages = input.size();
  std::vector< ImagePointer >         images(numberOfImages);
  std::vector< ImageMaskPointer >     imageMasks(numberOfImages);

  std::vector< TransformPointer >     transforms(numberOfImages);
  std::vector< Rigid3DTransformPointer >     rigid3DTransforms(numberOfImages);

  std::vector< RegistrationPointer >  registration(numberOfImages);
  std::vector< Rigid3DRegistrationPointer >  rigid3DRegistration(numberOfImages);

  std::vector< MaskPointer >          masks(numberOfImages);
  std::vector< RegionType >           rois(numberOfImages);

  ImagePointer hrImage;
  ImagePointer hrRefImage;
  if (verbose){
    std::cout<<"Reading the reference image : "<<refImage<<"\n";
  }
  ImageReaderType::Pointer imageReader = ImageReaderType::New();
  imageReader -> SetFileName( refImage );
  imageReader -> Update();
  hrRefImage = imageReader -> GetOutput();


  for (unsigned int i=0; i<numberOfImages; i++)
  {
    if (verbose){
      std::cout<<"Reading image : "<<input[i].c_str()<<"\n";
    }
    ImageReaderType::Pointer imageReader = ImageReaderType::New();
    imageReader -> SetFileName( input[i].c_str() );
    imageReader -> Update();
    images[i] = imageReader -> GetOutput();


    if (verbose){
      std::cout<<"Reading masks:"<<mask[i]<<std::endl;
    }
    MaskReaderType::Pointer maskReader = MaskReaderType::New();
    maskReader -> SetFileName( mask[i].c_str() );
    maskReader -> Update();
    imageMasks[i] = maskReader -> GetOutput();

    masks[i] = MaskType::New();
    masks[i] -> SetImage( imageMasks[i] );
    rois[i] = masks[i] -> GetAxisAlignedBoundingBoxRegion();

    if (verbose){
      std::cout<<"Reading transform:"<<transform[i]<<std::endl<<std::endl;
    }
    TransformReaderType::Pointer transformReader = TransformReaderType::New();
    transformReader -> SetFileName( transform[i] );
    transformReader -> Update();

    TransformListType transformsList = transformReader->GetTransformList();
    TransformReaderType::TransformListType::const_iterator titr = transformsList->begin();
    TransformPointer trans = static_cast< TransformType * >( titr->GetPointer() );

    //transforms[i]= TransformType::New();
    transforms[i] = TransformType::New();
    transforms[i]=static_cast< TransformType * >( titr->GetPointer() );
    //transforms[i] -> SetImage( preImages[i] );
    //transforms[i] -> SetImage( orientImageFilter[i] -> GetOutput());
    transforms[i] -> SetImage( imageReader -> GetOutput());

  }




  if (verbose){
    std::cout << std::endl; std::cout.flush();

    // Inject images
    std::cout << "Injecting images ... "; std::cout.flush();
  }

  ResamplerType::Pointer resampler = ResamplerType::New();

  for (unsigned int i=0; i<numberOfImages; i++)
  {
    resampler -> AddInput( images[i] );
    resampler -> AddMask( imageMasks[i] );
    resampler -> AddRegion( rois[i] );

    resampler -> SetTransform(i, transforms[i]) ;
  }

  resampler -> UseReferenceImageOn();
  resampler -> SetReferenceImage( hrRefImage );
  // resampler -> SetReferenceImageMask(lowToHighResFilter -> GetImageMaskCombination());

  // resampler -> SetUseDebluringKernel( debluringArg.isSet() );

  resampler -> Update();


  hrImage = resampler -> GetOutput();
  if (verbose){
    std::cout << "done. " << std::endl; std::cout.flush();
  }


  // Write HR image

  typedef itk::ImageFileWriter< ImageType >  WriterType;

  WriterType::Pointer writer =  WriterType::New();
  writer-> SetFileName( outImage );
  writer-> SetInput( hrImage );
  writer-> Update();


  return EXIT_SUCCESS;

  } catch (TCLAP::ArgException &e)  // catch any exceptions
  { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}
