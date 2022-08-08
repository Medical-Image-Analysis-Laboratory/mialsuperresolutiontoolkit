
/*=========================================================================

Program: Performs slice to volume registration and Scattered Data Interpolation
Language: C++
Date: $Date: 2012-28-12 14:00:00 +0100 (Fri, 28 Dec 2012) $
Version: $Revision: 1 $
Author: $Sebastien Tourbier$

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
  TCLAP::MultiArg<std::string> roiArg("","roi","roi file (written as mask)",false,
      "string",cmd);
  TCLAP::MultiArg<std::string> resampledArg("","ir","Resampled image with initial "
      "transform (this is an output to check initial transform consistency)",false,"string",cmd);
  TCLAP::ValueArg<std::string> outArg("o","output","High resolution image",true,
      "","string",cmd);
  TCLAP::ValueArg<std::string> refArg("r","reference","Reference Image",false, "","string",cmd);

  TCLAP::ValueArg<std::string> combinedMaskArg("","combinedMasks","All image "
      "masks combined in a single one",false,"","string",cmd);
  TCLAP::ValueArg<unsigned int> iterArg("n","iter","Maximum number of iterations"
      " (default 10)",false, 10,"unsigned int",cmd);
  TCLAP::ValueArg<double> epsilonArg("e","epsilon","Minimal percent change between "
      "two iterations considered as convergence. (default 0.0001)",false, 1e-4,
      "double",cmd);

  TCLAP::ValueArg<double> marginArg("","margin","Adds a margin to the reconstructed "
      "images to compensate for a small FOV in the reference. The value must be "
      "provided in millimeters (default 0).",false, 0.0,"double",cmd);

  TCLAP::SwitchArg  boxSwitchArg("","box","Use intersections for roi calculation",false);
  TCLAP::SwitchArg  maskSwitchArg("","mask","Use masks for roi calculation",false);
  TCLAP::SwitchArg  allSwitchArg("","all","Use the whole image FOV",false);

  TCLAP::SwitchArg  rigid3DSwitchArg("","3D","Use of 3D rigid transforms."
      " Recommended for adult subjects.", cmd, false);

  TCLAP::SwitchArg  noregSwitchArg("","noreg","No registration is performed, the "
      "image is reconstructed with the identity transform. This option is important "
      "to have a reference for performance assessment. ", cmd, false);

  // Flag that set deblurring Kernel for SDI
  TCLAP::SwitchArg  debluringArg("","debluring","Flag to set deblurring kernel for SDI (double the neighborhood)"
                                          " (by default it is disable.).",cmd,false);

  //TCLAP::SwitchArg  atlasInitSwitchArg("","atlasInit","Atlas image is used as as initial HR image for "
  //    "slice-to-volume registration , Assumed initial transform is result of global rigid stack registration", cmd, false);

  // xor arguments for roi assessment

  std::vector<TCLAP::Arg*>  xorlist;
  xorlist.push_back(&boxSwitchArg);
  xorlist.push_back(&maskSwitchArg);
  xorlist.push_back(&allSwitchArg);

  cmd.xorAdd( xorlist );

  // Parse the argv array.
  cmd.parse( argc, argv );

  input = inputArg.getValue();
  mask = maskArg.getValue();
  outImage = outArg.getValue().c_str();
  refImage = refArg.getValue();
  combinedMask = combinedMaskArg.getValue().c_str();
  transform = transformArg.getValue();
  roi = roiArg.getValue();
  resampled = resampledArg.getValue();
  itMax = iterArg.getValue();
  epsilon = epsilonArg.getValue();
  margin = marginArg.getValue();

  bool rigid3D = rigid3DSwitchArg.getValue();
  bool noreg   = noregSwitchArg.getValue();
  //bool useatlas = atlasInitSwitchArg.getValue();

  // typedefs
  const bool verbose = false;
  const    unsigned int    Dimension = 3;
  typedef  float           PixelType;

  typedef itk::Image< PixelType, Dimension >  ImageType;
  typedef ImageType::Pointer                  ImagePointer;

  typedef ImageType::RegionType               RegionType;
  typedef std::vector< RegionType >           RegionArrayType;

  typedef itk::ImageFileReader< ImageType  >  ImageReaderType;

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
  ImagePointer hrImageOld;
  ImagePointer hrImageIni;
  ImagePointer hrRefImage;

  lowToHighResFilter -> SetNumberOfImages(numberOfImages);
  lowToHighResFilter -> SetTargetImage( 0 );
  lowToHighResFilter -> SetMargin( margin );

  if (noreg)
  {
    lowToHighResFilter -> SetIterations( 0 );
    itMax = 1;
  }
  bool computeRefImage = true;

  if(refImage != "")
  {
      std::cout<<"Reading the reference image : "<<refImage<<"\n";
      ImageReaderType::Pointer imageReader = ImageReaderType::New();
      imageReader -> SetFileName( refImage );
      imageReader -> Update();
      hrImageIni = imageReader -> GetOutput();
      hrRefImage = imageReader -> GetOutput();
      computeRefImage = false;
  }

  for (unsigned int i=0; i<numberOfImages; i++)
  {
    std::cout<<"Reading image : "<<input[i].c_str()<<"\n";
    ImageReaderType::Pointer imageReader = ImageReaderType::New();
    imageReader -> SetFileName( input[i].c_str() );
    imageReader -> Update();
    images[i] = imageReader -> GetOutput();

    lowToHighResFilter -> SetImageArray(i, images[i] );

    if ( boxSwitchArg.isSet() )
    {
      intersectionCalculator -> AddImage( images[i] );
    }

  }

  if ( boxSwitchArg.isSet() )
  {
    intersectionCalculator -> Update();
  }

  // Set roi according to the provided arguments

  for (unsigned int i=0; i<numberOfImages; i++)
  {
    // if a mask of the LR image is provided ...
    if ( maskSwitchArg.isSet() )
    {
      MaskReaderType::Pointer maskReader = MaskReaderType::New();
      maskReader -> SetFileName( mask[i].c_str() );
      maskReader -> Update();
      imageMasks[i] = maskReader -> GetOutput();
      lowToHighResFilter -> SetImageMaskArray( i, imageMasks[i] );

      masks[i] = MaskType::New();
      masks[i] -> SetImage( imageMasks[i] );
      rois[i] = masks[i] -> GetAxisAlignedBoundingBoxRegion();

      lowToHighResFilter -> SetRegionArray( i, rois[i] );


    }
    // estimate the intersection of the ROIs (-> brain can be cropped !)
    else if ( boxSwitchArg.isSet() )
      {
        imageMasks[i] = intersectionCalculator -> GetImageMask(i);
        lowToHighResFilter -> SetImageMaskArray( i, imageMasks[i] );

        masks[i] = MaskType::New();
        masks[i] -> SetImage( imageMasks[i] );
        rois[i] = masks[i] -> GetAxisAlignedBoundingBoxRegion();
        lowToHighResFilter -> SetRegionArray( i, rois[i] );


      }
      // use the entire image (longer computation)
      else if ( allSwitchArg.isSet() )
        {
          DuplicatorType::Pointer duplicator = DuplicatorType::New();
          duplicator -> SetInputImage( images[i] );
          duplicator -> Update();

          CasterType::Pointer caster = CasterType::New();
          caster -> SetInput( duplicator -> GetOutput() );
          caster -> Update();

          imageMasks[i] = caster -> GetOutput();
          imageMasks[i] -> FillBuffer(1);
          lowToHighResFilter -> SetImageMaskArray( i, imageMasks[i] );

          masks[i] = MaskType::New();
          masks[i] -> SetImage( imageMasks[i] );
          rois[i] = masks[i] -> GetAxisAlignedBoundingBoxRegion();
          lowToHighResFilter -> SetRegionArray( i, rois[i] );

        }
  }

  std::cout<<"Start rigid registration on the desired target image (#0 by default)\n";
  try
    {

       lowToHighResFilter->StartRegistration();

    }
  catch( itk::ExceptionObject & err )
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }

  // Write combined image mask

  typedef itk::ImageFileWriter< ImageMaskType >  MaskWriterType;

  if ( strcmp(combinedMask,"") != 0 )
  {
    try
    {
      std::cout << "Write image mask combination..." <<std::endl << lowToHighResFilter -> GetImageMaskCombination() << std::endl;
      MaskWriterType::Pointer maskWriter =  MaskWriterType::New();
      maskWriter -> SetFileName( combinedMask );
      maskWriter -> SetInput( lowToHighResFilter -> GetImageMaskCombination() );
      maskWriter -> Update();
    }
    catch( itk::ExceptionObject & err )
    {
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
    }
  }

  // Write resampled LR images in HR space
  if ( resampled.size() > 0 )
  {
    for (unsigned int i=0; i<numberOfImages; i++)
    {
      lowToHighResFilter -> WriteResampledImages( i, resampled[i].c_str() );
    }
  }

  // Image registration performed slice by slice or affine 3D according to
  // the user selection

  if(computeRefImage)
  {
    std::cout << "Get global rigid HR image" << std::endl;
    hrImageIni = lowToHighResFilter->GetHighResolutionImage();
    hrRefImage = lowToHighResFilter->GetHighResolutionImage();
  }

  for (unsigned int i=0; i<numberOfImages; i++)
  {
    if (rigid3D)
    {
      rigid3DTransforms[i] = lowToHighResFilter -> GetTransformArray(i);
    }else
    {
      transforms[i] = TransformType::New();
      transforms[i] -> SetImage( images[i] );
      transforms[i] -> Initialize( lowToHighResFilter -> GetInverseTransformArray(i) );
    }
  }

  unsigned int im = numberOfImages;

  //Identify and exclude bad slices based on SNR
  /** Check all image slices and compute and store the SNR metric for all slices */

  std::cout << "Checking image slice quality... " << std::endl;
  /** Stores the starting slice index of jth image in the image slice index vector */
  //m_ImageSliceIndex.push_back( m_sliceSNRmetric.size() );

  /** Anisotropic diffusion Gaussian filtering for computing the SNR measure */
  // typedef itk::CurvatureAnisotropicDiffusionImageFilter< ImageType, ImageType >  ADGaussianFilterType;

  // typedef itk::SubtractImageFilter< InputImageType, ImageType, ImageType > SubtractFilterType;
  // typedef itk::ResampleImageFilter< ImageType, ImageType > ResampleFilterType;

  // ADGaussianFilter = ADGaussianFilterType::New();
  // ADGaussianFilter->SetInput( images[i] );
  // int          numberOfIterations = 3;
  // double       conductance = 3.0;
  // double       timeStep = 0.0125;
  // ADGaussianFilter->SetNumberOfIterations( numberOfIterations );
  // ADGaussianFilter->SetTimeStep( timeStep );
  // ADGaussianFilter->SetConductanceParameter( conductance );
  // ADGaussianFilter->Update();


  // Slice-to-volume registration

  float previousMetric = 0.0;
  float currentMetric = 0.0;

  // Default registration step lengths
  float min_step = 0.0001;
  float max_step = 0.2;

  // Reduce default registration step lengths
  min_step = 0.5 * min_step;
  max_step = 0.5 * max_step;

  for(unsigned int it=1; it <= itMax; it++)
  {
    std::cout << "Iteration " << it << std::endl; std::cout.flush();

    // Start registration

    #pragma omp parallel for private(im) schedule(dynamic)

    for (im=0; im<numberOfImages; im++)
    {
      if (verbose){
        std::cout << "Registering image " << im << " ... "; std::cout.flush();
      }
      if (rigid3D)
      {
        rigid3DRegistration[im] = Rigid3DRegistrationType::New();
        rigid3DRegistration[im] -> SetFixedImage( images[im] );
        rigid3DRegistration[im] -> SetMovingImage( hrRefImage );
        rigid3DRegistration[im] -> SetFixedImageMask( imageMasks[im] );
        rigid3DRegistration[im] -> SetTransform( rigid3DTransforms[im] );

        if (noreg)
          rigid3DRegistration[im] -> SetIterations( 0 );

        try
          {
          rigid3DRegistration[im]->Update();
          }
        catch( itk::ExceptionObject & err )
          {
          std::cerr << "ExceptionObject caught !" << std::endl;
          std::cerr << err << std::endl;
  //        return EXIT_FAILURE;
          }

        rigid3DTransforms[im] = rigid3DRegistration[im] -> GetTransform();
      } else
        {
          registration[im] = RegistrationType::New();
          registration[im] -> SetFixedImage( images[im] );
          registration[im] -> SetMovingImage( hrRefImage );

          if (it == 1)
          {
            std::cout << "Iteration " << it << ":" << std::endl;
            std::cout << "  - Use the initial reference image for slice-to-volume registration. " << std::endl;
            registration[im] -> SetMovingImage( hrRefImage );
          }
          else
          {
            std::cout << "Iteration " << it << " > 1 :" << std::endl;
            std::cout << "  - Update the reference image for slice-to-volume registration with reconstructed HR image of previous iteration. " << std::endl;
            registration[im] -> SetMovingImage( hrImage );
          }
          std::cout << "  - Use constant registration step lengths: [" << min_step << ";"<< max_step << "]" << std::endl << std::cout.flush();
          registration[im] -> SetMinStepLength(min_step);
          registration[im] -> SetMaxStepLength(max_step);
          registration[im] -> SetImageMask( imageMasks[im] );
          registration[im] -> SetTransform( transforms[im] );

          if (noreg)
            registration[im] -> SetIterations( 0 );

          try
            {
                registration[im] -> StartRegistration();
            }
          catch( itk::ExceptionObject & err )
            {
            std::cerr << "ExceptionObject caught !" << std::endl;
            std::cerr << err << std::endl;
    //        return EXIT_FAILURE;
            }

          transforms[im] = static_cast< TransformType* >(registration[im] -> GetTransform());
        }

      std::cout << "done. "; std::cout.flush();

    }

    std::cout << std::endl; std::cout.flush();

    // Inject images

    std::cout << "Injecting images ... "; std::cout.flush();


    ResamplerType::Pointer resampler = ResamplerType::New();

    for (unsigned int i=0; i<numberOfImages; i++)
    {
      resampler -> AddInput( images[i] );
      resampler -> AddMask( imageMasks[i] );
      resampler -> AddRegion( rois[i] );

      if (rigid3D)
      {
        transforms[i] = TransformType::New();
        transforms[i] -> SetImage( images[i] );
        transforms[i] -> Initialize( rigid3DTransforms[i] );
      }

      resampler -> SetTransform(i, transforms[i]) ;
    }

    resampler -> UseReferenceImageOn();
    resampler -> SetReferenceImage( hrRefImage );
    resampler -> SetReferenceImageMask(lowToHighResFilter -> GetImageMaskCombination());

    resampler -> SetUseDebluringKernel( debluringArg.isSet() );

    resampler -> Update();

    if (it == 1)
      hrImageOld = hrImageIni;
    else
      hrImageOld = hrImage;

    hrImage = resampler -> GetOutput();

    std::cout << "done. " << std::endl; std::cout.flush();

    // compute error

    typedef itk::Euler3DTransform< double > EulerTransformType;
    EulerTransformType::Pointer identity = EulerTransformType::New();
    identity -> SetIdentity();

    //typedef itk::LinearInterpolateImageFunction<
    //                                  ImageType,
    //                                  double>     InterpolatorType;
    typedef itk::BSplineInterpolateImageFunction<
                                      ImageType,
                                      double>     InterpolatorType;

    InterpolatorType::Pointer interpolator = InterpolatorType::New();

    NCMetricType::Pointer nc = NCMetricType::New();
    nc -> SetFixedImage(  hrImageOld );
    nc -> SetMovingImage( hrImage );
    nc -> SetFixedImageRegion( hrImageOld -> GetLargestPossibleRegion() );
    nc -> SetTransform( identity );
    nc -> SetInterpolator( interpolator );
    nc -> Initialize();

    previousMetric = currentMetric;
    currentMetric = - nc -> GetValue( identity -> GetParameters() );
    std::cout<<"previousMetric: "<<previousMetric<<", currentMetric: "<<currentMetric<<"\n";
    double delta = 0.0;

    if (it >= 2)
      delta = (currentMetric - previousMetric) / previousMetric;
    else
      delta = 1;

    if (delta < epsilon) break;

  }

  // Write HR image

  typedef itk::ImageFileWriter< ImageType >  WriterType;

  WriterType::Pointer writer =  WriterType::New();
  writer-> SetFileName( outImage );
  writer-> SetInput( hrImage );
  writer-> Update();


  // Write transforms

  typedef itk::TransformFileWriter TransformWriterType;

  if ( transform.size() > 0 )
  {
    for (unsigned int i=0; i<numberOfImages; i++)
    {
      TransformWriterType::Pointer transformWriter = TransformWriterType::New();

      if (rigid3D)
      {
        transformWriter -> SetInput( rigid3DTransforms[i] );
      } else
        {
          transformWriter -> SetInput( transforms[i] );
        }

      transformWriter -> SetFileName ( transform[i].c_str() );


      try
      {
        std::cout << "Writing " << transform[i].c_str() << " ... " ; std::cout.flush();
        transformWriter -> Update();
        std::cout << " done! " << std::endl;
      }
      catch ( itk::ExceptionObject & excp )
      {
        std::cerr << "Error while saving transform" << std::endl;
        std::cerr << excp << std::endl;
        std::cout << "[FAILED]" << std::endl;
        throw excp;
      }

    }
  }

  return EXIT_SUCCESS;

  } catch (TCLAP::ArgException &e)  // catch any exceptions
  { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}
