
/*=========================================================================

Program: Performs robust, segmentation-driven, Total-Variation-regularized super-resolution image reconstruction.
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
#include <sstream>

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>

/* Itk includes */
#include "itkEuler3DTransform.h"
#include "itkVersorRigid3DTransform.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageMaskSpatialObject.h"
#include "itkTransformFileReader.h"
#include "itkTransformFactory.h"
#include "itkCastImageFilter.h"

#include "itkResampleImageFilter.h"

#include "itkPermuteAxesImageFilter.h"
#include "itkFlipImageFilter.h"
#include "itkOrientImageFilter.h"

/*Btk includes*/
#include "mialsrtkSliceBySliceTransform.h"
//#include "btkSuperResolutionImageFilter.h"
#include "mialsrtkRobustSuperResolutionRigidImageFilterWithGMM.h"
#include "mialsrtkImageRegistrationFilter.h"

#include "mialsrtkLowToHighImageResolutionMethod.h"

#include "itkImageDuplicator.h"

#include "mialsrtkMaths.h"


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

        std::vector< std::string > input;
        std::vector< std::string > pre_input;

        std::vector< std::string > mask;
        std::vector< std::string > transform;
        std::vector< std::string > outTransform;

        double gap = 0.0;

        const char *debugDir = NULL;
        const char *debugfilename = "SR_igd_debug_loop_";

        const char *outImage = NULL;
        const char *refImage = NULL;

        const char *refMask = NULL;

        std::string wList,meanList,varList;
        std::vector< std::string > wFile,meanFile,varFile;

        const char *test = "undefined";

        std::vector< int > x1, y1, z1, x2, y2, z2;

        unsigned int iter;

        float beta;
        float lambda;
        float deltat = 1.0;
        float normD = 12.0;
        float theta_init = 1.0;
        float gamma = 1.0;

        float stepScale = 1.0;
        float tau_init = 1 / sqrt (12.0);
        float sigma_init = 1 / sqrt(12.0);

        float huberCriterion = 5.0;

        double innerConvThreshold;
        double outerConvThreshold;

        int numberOfLoops;
        int numberOfBregmanLoops;

        double start_time_unix, end_time_unix, diff_time_unix;

        // Parse arguments
        TCLAP::CmdLine cmd("Apply super-resolution algorithm using one or multiple input images.", ' ', "Unversioned");

        // Input LR images
        TCLAP::MultiArg<std::string> inputArg("i","input","Low-resolution image file",true,"string",cmd);
        // Input LR masks
        TCLAP::MultiArg<std::string> maskArg("m","mask","low-resolution image mask file",false,"string",cmd);
        // Ouput HR image
        TCLAP::ValueArg<std::string> outArg  ("o","output","Super resolution output image",true,"","string",cmd);

        // Input reconstructed image for initialization
        TCLAP::ValueArg<std::string> refArg  ("r","reconstructed","Reconstructed image for initialization. "
                                              "Typically the output of btkImageReconstruction is used." ,true,"","string",cmd);
        // Input motion parameters - Used only if initHR is disable, meaning that motion parameters were previously estimated
        TCLAP::MultiArg<std::string> transArg("t","in-transform","transform file",false,"string",cmd);

        //Optimization parameters
        TCLAP::ValueArg<int> iterArg  ("","iter","Number of inner iterations (default = 50)",false, 50,"int",cmd);
        TCLAP::ValueArg<float> betaArg  ("","beta","GMM-based Regularization factor (default = 0.1)",false, 0.1,"float",cmd);
        TCLAP::ValueArg<float> lambdaArg  ("","lambda","TV Regularization factor (default = 0.1)",false, 0.1,"float",cmd);
        TCLAP::ValueArg<float> deltatArg  ("","deltat","Parameter deltat (default = 1.0)",false, 1.0,"float",cmd);
        TCLAP::ValueArg<float> gammaArg  ("","gamma","Parameter gamma (default = 1.0)",false, 1.0,"float",cmd);
        TCLAP::ValueArg<float> stepScaleArg  ("","step-scale","Parameter step scale (default = 1.0)",false, 1.0,"float",cmd);
        TCLAP::ValueArg<double> innerLoopThresholdArg  ("","inner-thresh","Inner loop convergence threshold (default = 1e-4)",false, 1e-4,"double",cmd);
        TCLAP::ValueArg<double> outerLoopThresholdArg  ("","outer-thresh","Outer loop convergence threshold (default = 1e-4)",false, 1e-4,"double",cmd);
        TCLAP::ValueArg<int> loopArg  ("","loop","Number of loops (SR/denoising) (default = 5)",false, 5,"int",cmd);
        TCLAP::ValueArg<int> bregmanLoopArg  ("","bregman-loop","Number of Bregman loops (default = 10)",false, 10,"int",cmd);

        TCLAP::SwitchArg  boxcarSwitchArg("","boxcar","A boxcar-shaped PSF is assumed as imaging model"
                                          " (by default a Gaussian-shaped PSF is employed.).",cmd,false);

        // Flag that enables the update of motion parameters during SR
        TCLAP::SwitchArg  updateMotionSwitchArg("","update-motion","Flag to enable the update of motion parameters during SR"
                                                " (by default it is disable.).",cmd,false);

        // Arguments only used when motion estimation is updated during SR
        // Mask of the reconstructed image for initialization
        TCLAP::ValueArg<std::string> refMaskArg("","mask-reconstructed","Mask of the reconstructed image for initialization - Only used when motion estimation is updated during SR ",false,"","string",cmd);
        //Input preprocessed images for the first reconstruction -
        TCLAP::MultiArg<std::string> preInputArg("","pre-input","Low-resolution pre-processed image file - Only used when motion estimation is updated during SR",false,"string",cmd); // Used only if initHR is enable
        // Output motion parameters -  Only used when motion estimation is updated during SR
        TCLAP::MultiArg<std::string> outTransArg("","out-transform","output transform file - Only used when motion estimation is updated during SR",false,"string",cmd);

        TCLAP::ValueArg<std::string>debugDirArg("","debug","Directory where  SR reconstructed image at each outer loop of the reconstruction optimization is saved",false,"","string",cmd);

        //GMM parameters for feedback in the image restoration/reconstruction (SR) algorithm
        //TCLAP::ValueArg<std::string> wArg("","weight-image","Vector image (dim: number of tissue segmented) of the weights (GMM)",true,"","vector image",cmd);
        //TCLAP::ValueArg<std::string> meanArg("","mean-image","Vector image (dim: number of tissue segmented) of the means (GMM)",true,"","vector image",cmd);
        //TCLAP::ValueArg<std::string> varArg("","var-image","Vector image (dim: number of tissue segmented) of the variances (GMM)",true,"","vector image",cmd);

        //V1: take multiple inputs for each segmentation
        //TCLAP::MultiArg<std::string> wArg("","weight-image","Vector image (dim: number of tissue segmented) of the weights (GMM)",true,"string",cmd);
        //TCLAP::MultiArg<std::string> meanArg("","mean-image","Vector image (dim: number of tissue segmented) of the means (GMM)",true,"string",cmd);
        //TCLAP::MultiArg<std::string> varArg("","var-image","Vector image (dim: number of tissue segmented) of the variances (GMM)",true,"string",cmd);

        //V2: take one input that is a list of segmentation
        TCLAP::ValueArg<std::string> wListArg("W","gmm-weight-list","File containing a list of GMM segmentation weights",false,"undef","gmm weights list",cmd);
        TCLAP::ValueArg<std::string> meanListArg("M","gmm-mean-list","File containing a list of GMM segmentation means",false,"undef","gmm means list",cmd);
        TCLAP::ValueArg<std::string> varListArg("V","gmm-var-list","File containing a list of GMM segmentation variances",false,"undef","gmm variances list",cmd);

        TCLAP::SwitchArg  useRobustSRArg("","use-robust","Flag to enable the use of robust energy in SR"
                                         " (by default it is disable.). If enable, rho1 and rho2 tuning parameters can be set with flags --rho1 and --rho2. (rho1,rho2 =1e-1 by default)",cmd,false);

        TCLAP::ValueArg<float> rho1Arg  ("","rho1","Parameter rho1 (default = 0.1), used only in RobustSR flag is enabled.",false, 0.1,"float",cmd);
        TCLAP::ValueArg<float> rho2Arg  ("","rho2","Parameter rho2 (default = 0.1), used only in RobustSR flag is enabled.",false, 0.1,"float",cmd);
        TCLAP::ValueArg<float> huberArg  ("","huber-mode","Control how many median absolute deviations (MADs)  from the median error must  a point be before it becomes an outlier (Huber norm threshold),  (default = 5.0), used only in RobustSR flag is enabled.",false, 2.0,"float",cmd);

        TCLAP::ValueArg<float> upscalingArg("","upscaling-factor","Upscaling factor used in the super resolution (default = 1)",false, 2,"float",cmd);
        TCLAP::SwitchArg  verboseArg("v","verbose","Verbose output (False by default)",cmd, false);
        
        // Parse the argv array.
        cmd.parse( argc, argv );

        input = inputArg.getValue();
        mask = maskArg.getValue();
        outImage = outArg.getValue().c_str();

        refImage = refArg.getValue().c_str();
        transform = transArg.getValue();

        refMask = refMaskArg.getValue().c_str();
        pre_input = preInputArg.getValue();
        outTransform = outTransArg.getValue();
        
        //V1
        //wFile=wArg.getValue();
        //meanFile=meanArg.getValue();
        //varFile=varArg.getValue();

        //V2
        wList = wListArg.getValue();
        meanList = meanListArg.getValue();
        varList = varListArg.getValue();

        iter = iterArg.getValue();

        beta = betaArg.getValue();
        lambda = lambdaArg.getValue();

        deltat = deltatArg.getValue();
        gamma = gammaArg.getValue();
        innerConvThreshold = innerLoopThresholdArg.getValue();
        outerConvThreshold = outerLoopThresholdArg.getValue();
        numberOfLoops = loopArg.getValue();
        numberOfBregmanLoops = bregmanLoopArg.getValue();
        stepScale = stepScaleArg.getValue();

        tau_init =  stepScale * tau_init;
        sigma_init = ( 1 / stepScale ) * sigma_init;

        debugDir = debugDirArg.getValue().c_str();
        bool verbose = verboseArg.getValue();
        
        if ( ( strcmp(refMask,"") == 0 || pre_input.size() == 0 ) && updateMotionSwitchArg.isSet())
        {
            std::cout << "Execution abandonned - Motion Update during SR is enable but some required input are missing" << std::endl;
            return EXIT_FAILURE;
        }


        // typedefs
        const   unsigned int    Dimension = 3;
        typedef mialsrtk::SliceBySliceTransformBase< double, Dimension > TransformBaseType;
        typedef mialsrtk::SliceBySliceTransform< double, Dimension > TransformType;
        typedef TransformType::Pointer                          TransformPointer;

        // Register the SliceBySlice transform (a non-default ITK transform) with the TransformFactory of ITK
        itk::TransformFactory<TransformType>::RegisterTransform();

        typedef float  PixelType;

        typedef itk::Image< PixelType, Dimension >  ImageType;
        typedef ImageType::Pointer                  ImagePointer;
        typedef std::vector<ImagePointer>           ImagePointerArray;

        typedef itk::Image< unsigned char, Dimension >  ImageMaskType;
        typedef itk::ImageFileReader< ImageMaskType > MaskReaderType;
        typedef itk::ImageMaskSpatialObject< Dimension > MaskType;

        typedef ImageType::RegionType               RegionType;
        typedef std::vector< RegionType >           RegionArrayType;

        typedef itk::ImageFileReader< ImageType >   ImageReaderType;
        typedef itk::ImageFileWriter< ImageType >   WriterType;

        typedef itk::TransformFileReader     TransformReaderType;
        typedef TransformReaderType::TransformListType * TransformListType;

        //Vector image type
        typedef itk::VectorImage<PixelType,Dimension> VectorImageType;
        typedef VectorImageType::Pointer VectorImagePointer;
        typedef itk::ImageFileReader <VectorImageType> VectorImageReaderType;

        /* Registration type required in case of slice by slice transformations
  A rigid transformation is employed because there is not distortions like
  in diffusion imaging. We have performed a comparison of accuracy between
  both types of transformations. */
        typedef mialsrtk::SliceBySliceRigidRegistration<ImageType> RegistrationType;
        typedef RegistrationType::Pointer RegistrationPointer;

        // Rigid 3D transform definition (typically for reconstructions in adults)
        typedef itk::VersorRigid3DTransform< double > Rigid3DTransformType;
        typedef Rigid3DTransformType::Pointer   Rigid3DTransformPointer;

        typedef itk::Euler3DTransform< double > EulerTransformType;

        // Resampler type required in case of a slice by slice transform
        typedef mialsrtk::ResampleImageByInjectionFilter< ImageType, ImageType >  ResamplerByInjectionType;

        // Registration Metric
        typedef itk::NormalizedCorrelationImageToImageMetric< ImageType,ImageType > NCMetricType;

        //typedef btk::ImageIntersectionCalculator<ImageType> IntersectionCalculatorType;
        //IntersectionCalculatorType::Pointer intersectionCalculator = IntersectionCalculatorType::New();

        // Interpolator used to compute the error metric between 2 registration iterations
        //typedef itk::LinearInterpolateImageFunction<ImageType,double>     InterpolatorType;
        typedef itk::BSplineInterpolateImageFunction<ImageType,double>     InterpolatorType;

        //typedef itk::CastImageFilter<ImageType,ImageMaskType> CasterType;

        // A helper class which creates an image which is perfect copy of the input image
        typedef itk::ImageDuplicator<ImageType> DuplicatorType;

        // Super resolution filter that solves the inverse problem
        typedef mialsrtk::RobustSuperResolutionRigidImageFilterWithGMM< ImageType, ImageType >  ResamplerType;
        ResamplerType::Pointer resampler = ResamplerType::New();
        resampler -> SetVerbose(verbose);

        typedef itk::OrientImageFilter<ImageType,ImageType> OrientImageFilterType;
        typedef itk::OrientImageFilter<ImageMaskType,ImageMaskType> OrientImageMaskFilterType;

        typedef itk::LinearInterpolateImageFunction<ImageType,double>     LinearInterpolatorType;
        typedef itk::ResampleImageFilter<ImageType,ImageType>             ResampleImageFilterType;

        typedef itk::LinearInterpolateImageFunction<VectorImageType,double>     VectorLinearInterpolatorType;
        typedef itk::NearestNeighborInterpolateImageFunction<VectorImageType,double>     VectorNNInterpolatorType;
        typedef itk::ResampleImageFilter<VectorImageType,VectorImageType>             ResampleVectorImageFilterType;

        typedef itk::NearestNeighborInterpolateImageFunction<ImageMaskType,double>     NNInterpolatorType;
        typedef itk::ResampleImageFilter<ImageMaskType,ImageMaskType>             ResampleImageMaskFilterType;




        unsigned int numberOfImages = input.size();

        //V1
        //unsigned int numberOfSegmentations = wFile.size();

        std::vector<OrientImageFilterType::Pointer> orientImageFilter(numberOfImages);
        std::vector<OrientImageMaskFilterType::Pointer> orientMaskImageFilter(numberOfImages);

        std::vector< ImagePointer >         preImages(numberOfImages);
        std::vector< ImageMaskType::Pointer >     imageMasks(numberOfImages);
        std::vector< TransformPointer >     transforms(numberOfImages);
        std::vector< RegistrationPointer >  registration(numberOfImages);

        std::vector<MaskType::Pointer> masks(numberOfImages);


        std::vector< RegionType >           rois(numberOfImages);

        ImageType::IndexType  roiIndex;
        ImageType::SizeType   roiSize;

        // Filter setup
        for (unsigned int i=0; i<numberOfImages; i++)
        {
            // add image
            std::cout<<"Reading image : "<<input[i].c_str()<<std::endl;
            ImageReaderType::Pointer imageReader = ImageReaderType::New();
            imageReader -> SetFileName( input[i].c_str() );
            imageReader -> Update();

            resampler -> AddInput(  imageReader -> GetOutput() );

            /*
            orientImageFilter[i] = OrientImageFilterType::New();
            orientImageFilter[i] -> UseImageDirectionOn();
            orientImageFilter[i] -> SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP);
            orientImageFilter[i] -> SetInput(imageReader -> GetOutput());
            orientImageFilter[i] -> Update();

            resampler -> AddInput( orientImageFilter[i] -> GetOutput() );
            */

            //registrationFilter -> AddImage(imageReader -> GetOutput());

            // add region
            if ( mask.size() > 0 )
            {
                std::cout<<"Reading mask image : "<<mask[i].c_str()<<std::endl;
                MaskReaderType::Pointer maskReader = MaskReaderType::New();
                maskReader -> SetFileName( mask[i].c_str() );
                maskReader -> Update();

                imageMasks[i] = maskReader  -> GetOutput();

                /*
                orientMaskImageFilter[i] = OrientImageMaskFilterType::New();
                orientMaskImageFilter[i] -> UseImageDirectionOn();
                orientMaskImageFilter[i] -> SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP);
                orientMaskImageFilter[i] -> SetInput(maskReader -> GetOutput());
                orientMaskImageFilter[i] -> Update();

                imageMasks[i] = orientMaskImageFilter[i]  -> GetOutput();
                */

                //MaskType::Pointer mask = MaskType::New();
                masks[i]= MaskType::New();
                masks[i] -> SetImage( imageMasks[i] );

                rois[i] = masks[i] -> GetAxisAlignedBoundingBoxRegion();
                //std::cout << "rois "<< i << " : "<<rois[i]<<std::endl;

                resampler -> AddMask( masks[i] );

                RegionType roi = masks[i] -> GetAxisAlignedBoundingBoxRegion();
                roiIndex = roi.GetIndex();
                roiSize  = roi.GetSize();

            } else
            {
                std::cout<<"Creating a mask image (entire input image)"<<std::endl;
                roiSize  = imageReader -> GetOutput() -> GetLargestPossibleRegion().GetSize();
                roiIndex = imageReader -> GetOutput() -> GetLargestPossibleRegion().GetIndex();
            }

            RegionType imageRegion;
            imageRegion.SetIndex(roiIndex);
            imageRegion.SetSize (roiSize);
            resampler -> AddRegion( imageRegion );

            if ( pre_input.size() > 0)
            {
                // add image
                std::cout<<"Reading pre-processed image : "<<pre_input[i].c_str()<<std::endl;
                ImageReaderType::Pointer preImageReader = ImageReaderType::New();
                preImageReader -> SetFileName( pre_input[i].c_str() );
                preImageReader -> Update();

                preImages[i] = preImageReader -> GetOutput();

            }


            if (transform.size() > 0 )
            {
                std::cout<<"Reading transform:"<<transform[i]<<std::endl;
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


                for(unsigned int j=0; j< trans -> GetNumberOfSlices(); j++)
                    resampler -> SetTransform(i, j, trans -> GetSliceTransform(j) ) ;

            }

        }

        // Set the reference image
        std::cout<<"Reading the reference image : "<<refImage<<std::endl;
        ImageReaderType::Pointer refReader = ImageReaderType::New();
        refReader -> SetFileName( refImage );
        refReader -> Update();

        ImageType::Pointer referenceIm = refReader->GetOutput();

        ImageType::SizeType refSize = referenceIm->GetLargestPossibleRegion().GetSize();
        ImageType::SpacingType refSpacing = referenceIm->GetSpacing();

        ImageType::SpacingType newRefSpacing;
        newRefSpacing[0] = refSpacing[0] / upscalingArg.getValue();
        newRefSpacing[1] = refSpacing[1] / upscalingArg.getValue();
        newRefSpacing[2] = refSpacing[2] / upscalingArg.getValue();

        ImageType::SizeType newRefSize;
        newRefSize[0]= ( refSpacing[0] / newRefSpacing[0] ) * refSize[0];
        newRefSize[1]= ( refSpacing[1] / newRefSpacing[1] ) * refSize[1];
        newRefSize[2]= ( refSpacing[2] / newRefSpacing[2] ) * refSize[2];

        LinearInterpolatorType::Pointer linInterpolator = LinearInterpolatorType::New();

        EulerTransformType::Pointer idTransform = EulerTransformType::New();
        idTransform->SetIdentity();

        ResampleImageFilterType::Pointer upsampler = ResampleImageFilterType::New();
        upsampler -> SetTransform(idTransform);
        upsampler -> SetInterpolator( linInterpolator );
        upsampler -> SetOutputParametersFromImage(referenceIm);
        //upsampler -> SetOutputDirection(referenceIm->GetDirection());
        //upsampler -> SetOutputOrigin(referenceIm->GetOrigin());
        upsampler -> SetOutputSpacing(newRefSpacing);
        upsampler -> SetSize(newRefSize);
        upsampler -> SetInput(referenceIm);

        upsampler -> Update();

        upsampler -> Print( std::cout );

        referenceIm = upsampler->GetOutput();


        referenceIm -> Print( std::cout );

        /*
        OrientImageFilterType::Pointer orientRefImageFilter = OrientImageFilterType::New();
        orientRefImageFilter -> UseImageDirectionOn();
        orientRefImageFilter -> SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP);
        orientRefImageFilter -> SetInput(refReader -> GetOutput());
        orientRefImageFilter -> Update();

        ImageType::Pointer referenceIm = orientRefImageFilter->GetOutput();
        */

        ImageType::RegionType referenceRegion =  referenceIm->GetLargestPossibleRegion();

        //std::cout << "Reference region : " << referenceIm->GetLargestPossibleRegion() << std::endl;
        //std::cout << "Reference image size at loading : " << referenceIm->GetLargestPossibleRegion().GetNumberOfPixels() << std::endl;

        // Set the mask of the reference image if provided
        ImageMaskType::Pointer imageMaskCombination;
        if(strcmp(refMask,"") != 0)
        {
            MaskReaderType::Pointer refMaskReader = MaskReaderType::New();
            refMaskReader -> SetFileName( refMask );
            refMaskReader -> Update();

            imageMaskCombination = refMaskReader -> GetOutput();

            NNInterpolatorType::Pointer nnInterpolator = NNInterpolatorType::New();

            ResampleImageMaskFilterType::Pointer maskUpsampler = ResampleImageMaskFilterType::New();
            maskUpsampler -> SetInterpolator(nnInterpolator);
            maskUpsampler -> SetInput(refMaskReader->GetOutput());

            maskUpsampler -> SetTransform(idTransform);
            maskUpsampler -> SetOutputParametersFromImage(imageMaskCombination);
            maskUpsampler -> SetOutputSpacing(newRefSpacing);
            maskUpsampler -> SetSize(newRefSize);

            maskUpsampler -> Update();

            imageMaskCombination = maskUpsampler->GetOutput();

            /*
            OrientImageMaskFilterType::Pointer orientRefMaskImageFilter = OrientImageMaskFilterType::New();
            orientRefMaskImageFilter -> UseImageDirectionOn();
            orientRefMaskImageFilter -> SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP);
            orientRefMaskImageFilter -> SetInput(refMaskReader -> GetOutput());
            orientRefMaskImageFilter -> Update();

            imageMaskCombination = orientRefMaskImageFilter -> GetOutput()
            */
        }

        std::cout << "==========================================================================" << std::endl << std::endl;


        //V1
        /*
        std::vector< VectorImagePointer > weightVImages(numberOfSegmentations);
        std::vector< VectorImagePointer > meanVImages(numberOfSegmentations);
        std::vector< VectorImagePointer > varVImages(numberOfSegmentations);

        for(unsigned int i=0; i < numberOfSegmentations; i++)
        {
            std::cout<<"Reading GMM parameters of segmentation # " << i << std::endl;
            // Read the vector images with GMM parameters
            std::cout<<"Reading the GMM weights : " << wFile[i] << std::endl;
            VectorImageReaderType::Pointer wReader = VectorImageReaderType::New();
            wReader -> SetFileName( wFile[i] );
            wReader -> Update();

            weightVImages[i] = wReader -> GetOutput();

            std::cout<<"Reading the GMM means : " << meanFile[i] << std::endl;
            VectorImageReaderType::Pointer meanReader = VectorImageReaderType::New();
            meanReader -> SetFileName( meanFile[i] );
            meanReader -> Update();

            meanVImages[i] = meanReader -> GetOutput();

            std::cout<<"Reading the GMM variances : " << varFile[i] << std::endl;
            VectorImageReaderType::Pointer varReader = VectorImageReaderType::New();
            varReader -> SetFileName( varFile[i] );
            varReader -> Update();

            varVImages[i] = varReader -> GetOutput();

            std::cout << std::endl;
            std::cout << "**********************************************************************\n" << std::endl;
        }
        */

        //V2

        unsigned int nbSegsW=0;

        if(strcmp(wList.c_str(),"undef") != 0)
        {
            std::ifstream fileInW(wList.c_str());
            if (!fileInW.is_open() && beta>0)
            {
                std::cerr << "GMM weight file list " << wList.c_str() << " does not exist... Exiting..." << std::endl;
                exit(-1);
            }
            while (!fileInW.eof() && beta>0)
            {
                char tmpStr[2048];
                fileInW.getline(tmpStr,2048);
                if (strcmp(tmpStr,"") == 0)
                    continue;
                wFile.push_back(tmpStr);
                nbSegsW++;
            }

            unsigned int nbSegsM=0;
            std::ifstream fileInM(meanList.c_str());
            if (!fileInM.is_open() && beta>0)
            {
                std::cerr << "GMM mean file list " << meanList.c_str() << " does not exist... Exiting..." << std::endl;
                exit(-1);
            }
            while (!fileInM.eof() && beta>0)
            {
                char tmpStr[2048];
                fileInM.getline(tmpStr,2048);
                if (strcmp(tmpStr,"") == 0)
                    continue;
                meanFile.push_back(tmpStr);
                nbSegsM++;
            }

            unsigned int nbSegsV=0;
            std::ifstream fileInV(varList.c_str());
            if (!fileInV.is_open() && beta>0)
            {
                std::cerr << "GMM variance file list " << varList.c_str() << " does not exist... Exiting..." << std::endl;
                exit(-1);
            }
            while (!fileInV.eof() && beta>0)
            {
                char tmpStr[2048];
                fileInV.getline(tmpStr,2048);
                if (strcmp(tmpStr,"") == 0)
                    continue;
                varFile.push_back(tmpStr);
                nbSegsV++;
            }

            if( nbSegsW!=nbSegsM && nbSegsW!=nbSegsV && beta>0)
            {
                std::cerr << "Input GMM parameter lists do not match!" << std::endl;
                exit(-1);
            }

        }

        unsigned int numberOfSegmentations = nbSegsW;


        std::vector< VectorImagePointer > weightVImages(numberOfSegmentations);
        std::vector< VectorImagePointer > meanVImages(numberOfSegmentations);
        std::vector< VectorImagePointer > varVImages(numberOfSegmentations);
        if(strcmp(wList.c_str(),"undef")!=0)
        {

            for(unsigned int i=0; i < numberOfSegmentations; i++)
            {
                std::cout<<"Reading GMM parameters of segmentation # " << i << std::endl;
                // Read the vector images with GMM parameters
                std::cout<<"Reading the GMM weights : " << wFile[i] << std::endl;
                VectorImageReaderType::Pointer wReader = VectorImageReaderType::New();
                wReader -> SetFileName( wFile[i] );
                wReader -> Update();

                weightVImages[i] = wReader -> GetOutput();

                VectorNNInterpolatorType::Pointer vNNInterpolator = VectorNNInterpolatorType::New();

                ResampleVectorImageFilterType::Pointer wUpsampler = ResampleVectorImageFilterType::New();
                wUpsampler -> SetInterpolator(vNNInterpolator);
                wUpsampler -> SetInput(wReader -> GetOutput());

                wUpsampler -> SetTransform(idTransform);
                wUpsampler -> SetOutputParametersFromImage(weightVImages[i]);
                wUpsampler -> SetOutputSpacing(newRefSpacing);
                wUpsampler -> SetSize(newRefSize);

                wUpsampler -> Update();

                weightVImages[i] = wUpsampler -> GetOutput();
                weightVImages[i] -> Update();

                std::cout<<"Reading the GMM means : " << meanFile[i] << std::endl;
                VectorImageReaderType::Pointer meanReader = VectorImageReaderType::New();
                meanReader -> SetFileName( meanFile[i] );
                meanReader -> Update();

                meanVImages[i] = meanReader -> GetOutput();

                ResampleVectorImageFilterType::Pointer mUpsampler = ResampleVectorImageFilterType::New();
                mUpsampler -> SetInterpolator(vNNInterpolator);
                mUpsampler -> SetInput(meanReader -> GetOutput());

                mUpsampler -> SetTransform(idTransform);
                mUpsampler -> SetOutputParametersFromImage(meanVImages[i]);
                mUpsampler -> SetOutputSpacing(newRefSpacing);
                mUpsampler -> SetSize(newRefSize);

                mUpsampler -> Update();

                meanVImages[i] = mUpsampler -> GetOutput();
                meanVImages[i] -> Update();


                std::cout<<"Reading the GMM variances : " << varFile[i] << std::endl;
                VectorImageReaderType::Pointer varReader = VectorImageReaderType::New();
                varReader -> SetFileName( varFile[i] );
                varReader -> Update();

                varVImages[i] = varReader -> GetOutput();

                ResampleVectorImageFilterType::Pointer vUpsampler = ResampleVectorImageFilterType::New();
                vUpsampler -> SetInterpolator(vNNInterpolator);
                vUpsampler -> SetInput(varReader -> GetOutput());

                vUpsampler -> SetTransform(idTransform);
                vUpsampler -> SetOutputParametersFromImage(varVImages[i]);
                vUpsampler -> SetOutputSpacing(newRefSpacing);
                vUpsampler -> SetSize(newRefSize);

                vUpsampler -> Update();

                varVImages[i] = vUpsampler -> GetOutput();
                varVImages[i] -> Update();


                std::cout << std::endl;
                std::cout << "**********************************************************************\n" << std::endl;
            }
        }
        std::cout << "==========================================================================" << std::endl << std::endl;

        std::cout << "Performing super resolution (TV using IGD) with the following settings: \n" << std::endl;
        std::cout << "# iterations max : " << iter << std::endl;
        std::cout << "# loop max : " << numberOfLoops << std::endl << std::endl;
        std::cout << "Lambda (TV reg.) : " << lambda << std::endl;
        std::cout << "Beta (GMM reg.) : " << beta << std::endl;
        std::cout << "Gamma : " << gamma << std::endl;
        std::cout << "Delta t : " << deltat << std::endl;
        std::cout << "Convergence threshold (inner) : " << innerConvThreshold << std::endl;
        std::cout << "Convergence threshold (outer) : " << outerConvThreshold << std::endl << std::endl;

        //resampler -> Print(std::cout);


        WriterType::Pointer debugwriter =  WriterType::New();

        /*
  debugwriter -> SetFileName( "/Users/sebastientourbier/Desktop/Patient01/SR/Manual/SR_Patient07_3V_noNLM_BCORR_NORM_after_init_loop.nii.gz" );
  debugwriter -> SetInput( resampler -> GetOutput() );
  debugwriter->Update();
  */

        // Initialization of SR optimization parameters
        double criterion = 0.0;

        float theta = 0.0;
        float sigma = 0.0;
        float tau = 0.0;

        // Registration parameters
        unsigned int itMax=2;
        double epsilon=1e-4;


        if( updateMotionSwitchArg.isSet() )
        {
            std::cout << "Motion estimation enabled with the following settings: " << std::endl << std::endl;
            std::cout << "# iterations max : " << itMax << std::endl;
            std::cout << "Convergence threshold : " << epsilon << std::endl;
        }
        else
        {
            std::cout << "Motion estimation disabled" << std::endl;
        }

        std::cout << "==========================================================================" << std::endl << std::endl;


        // Variables used if motion estimation is updated during SR
        ImagePointer hrImageInit;
        ImagePointer hrImageRef;
        ImagePointer hrImage;
        ImagePointer hrImageOld;

        start_time_unix = mialsrtk::getTime();;

        // Bregman loops
        for (int j = 0; j < numberOfBregmanLoops; j++)
        {
            criterion = 1.0;

            std::cout << "Bregman loop init : "<< j << std::endl<<std::endl;
            std::cout<<"Theta : "<<theta_init<<std::endl;
            std::cout<<"Sigma : "<<sigma_init<<std::endl;
            std::cout<<"Tau : "<<tau_init<<std::endl<<std::endl;

            resampler -> SetCurrentBregmanLoop(j);
            resampler -> SetCurrentOuterIteration(0);

            resampler -> SetUseRobustSR( useRobustSRArg.isSet() );

            if( useRobustSRArg.isSet() )
            {
                resampler -> SetRho1( rho1Arg.getValue() );
                resampler -> SetRho2( rho2Arg.getValue() );
                resampler -> SetHuberCriterion(huberArg.getValue());
            }

            resampler -> UseReferenceImageOn();

            if (j == 0)
            {
                std::cout << "Initial HR image set to input image associated with flag -r." << std::endl<<std::endl;;
                resampler -> SetReferenceImage( upsampler -> GetOutput() );
            }
            else
            {
                std::cout << "Initial HR image set from previous iteration." << std::endl<<std::endl;;
                resampler -> SetReferenceImage( resampler -> GetOutput() );
            }

            resampler -> SetIterations(iter);
            resampler -> SetLambda( lambda );
            resampler -> SetGamma( gamma );
            resampler -> SetSigma( sigma_init );
            resampler -> SetTau( tau_init );
            resampler -> SetTheta( theta_init );
            resampler -> SetDeltat( deltat );

            resampler -> SetConvergenceThreshold( innerConvThreshold );

            resampler -> SetSliceGap( gap );

            resampler -> SetBeta( beta );

            for(unsigned int i=0; i<numberOfSegmentations; i++)
            {
                resampler -> AddGMMWeights( weightVImages[i] );
                resampler -> AddGMMMeans( meanVImages[i] );
                resampler -> AddGMMVars( varVImages[i] );
            }


            /*
    std::cout << "Reference image at loop " << 0 << ": " << std::endl;
    std::cout << "Pointer : " << refReader ->GetOutput() << std::endl;
    std::cout << "Region : " << refReader ->GetOutput() ->GetLargestPossibleRegion() << std::endl;
    */

            std::cout << "**************************************************************************" << std::endl << std::endl;

            if ( boxcarSwitchArg.isSet() )
                resampler -> SetPSF( ResamplerType::BOXCAR );
            resampler -> Print(std::cout);
            resampler -> Update();


            std::cout << "**************************************************************************" << std::endl << std::endl;

            theta = theta_init;
            sigma = sigma_init;
            tau = tau_init;

            vnl_vector<float> Z = resampler -> GetZVector();
            //vnl_vector<float> Y = resampler -> GetObservationsY();

            // Outer loops
            for (int i=0; i<numberOfLoops; i++)
            {
                //Motion estimation if enabled
                if(pre_input.size()>0 && updateMotionSwitchArg.isSet() )
                {
                    std::cout << "Update motion parameters (Slice by Slice)" << std::endl << std::endl;

                    DuplicatorType::Pointer duplicator = DuplicatorType::New();
                    duplicator->SetInputImage(resampler->GetOutput());
                    duplicator->Update();
                    hrImageRef = duplicator -> GetOutput();
                    hrImageRef -> DisconnectPipeline();

                    unsigned int im = numberOfImages;
                    float previousMetric = 0.0;
                    float currentMetric = 0.0;

                    //Iterative slice by slice registration
                    for(unsigned int it=1; it <= itMax; it++)
                    {
                        std::cout << "Iteration " << it << std::endl;// <<std::cout.flush();

                        // Start registration
#pragma omp parallel for private(im) schedule(dynamic)
                        for (im=0; im<numberOfImages; im++)
                        {
                            std::cout << "Registering image " << im << " / "<< numberOfImages <<" ... "; //std::cout.flush();

                            registration[im] = RegistrationType::New();
                            registration[im] -> SetFixedImage( preImages[im] );
                            registration[im] -> SetMovingImage( hrImageRef );
                            registration[im] -> SetImageMask( imageMasks[im] );
                            registration[im] -> SetTransform( transforms[im] );

                            try
                            {

                                registration[im] -> StartRegistration();

                            }
                            catch( itk::ExceptionObject & err )
                            {

                                std::cout << "ExceptionObject caught !" << std::endl;
                                std::cout << err << std::endl;
                                // return EXIT_FAILURE;
                            }

                            transforms[im] = static_cast< TransformType* >(registration[im] -> GetTransform());


                            std::cout << "done. "; //std::cout.flush();

                        }

                        //std::cout << std::endl; //std::cout.flush();

                        // Inject images onto a regular HR grid
                        std::cout << "Injecting images ...  "; //std::cout.flush();
                        ResamplerByInjectionType::Pointer resamplerByInj = ResamplerByInjectionType::New();

                        for (unsigned int p=0; p<numberOfImages; p++)
                        {
                            //std::cout << "Add input " << p << " : " << std::endl;
                            resamplerByInj -> AddInput( preImages[p] );
                            resamplerByInj -> AddRegion( rois[p] );
                            resamplerByInj -> SetTransform(p, transforms[p].GetPointer()) ;
                        }

                        resamplerByInj -> UseReferenceImageOn();
                        resamplerByInj -> SetReferenceImage( hrImageRef );
                        resamplerByInj -> SetReferenceImageMask(imageMaskCombination);
                        resamplerByInj -> Update();

                        if(it > 1)
                            hrImageOld = hrImage;

                        hrImage = resamplerByInj -> GetOutput();

                        std::cout << "done. " << std::endl; //std::cout.flush();

                        // compute error
                        double delta = 0.0;
                        if (it > 1)
                        {
                            EulerTransformType::Pointer identity = EulerTransformType::New();
                            identity -> SetIdentity();

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

                            delta = (currentMetric - previousMetric) / previousMetric;

                            std::cout<<"previousMetric: "<<previousMetric<<", currentMetric: "<<currentMetric<< ", delta : " << delta <<std::endl;
                        }
                        else
                        {
                            delta = 1;
                        }

                        if (delta < epsilon) break;

                    }// End of iterative registration

                    // Update transforms
                    for (unsigned int lr=0; lr<numberOfImages; lr++)
                    {
                        for(unsigned int s=0; s< transforms[lr] -> GetNumberOfSlices(); s++)
                            resampler -> SetTransform(lr, s, transforms[lr] -> GetSliceTransform(s) ) ;
                    }

                    std::cout << "**************************************************************************" << std::endl << std::endl;

                }// End of motion estimation

                resampler -> SetCurrentOuterIteration(i+1);
                std::cout << "Bregman loop : "<< j << " / TV Loop : "<< (resampler -> GetCurrentOuterIteration()) <<std::endl<<std::endl;

                //Update optimization parameters
                theta = resampler -> GetTheta();
                std::cout<<"Theta (outer) = "<<theta<<std::endl;
                resampler -> SetGamma( gamma );

                std::cout << "Sigma old / new = " << sigma << " / ";
                sigma = sigma / theta;
                std::cout<<sigma<<std::endl;
                resampler -> SetSigma( sigma  );

                std::cout << "Tau old / new  = " << tau << " / ";
                tau = theta * tau;
                std::cout<<tau<<std::endl;
                resampler -> SetTau( tau );

                resampler -> SetZVector(Z);

                //resampler -> SetTheta( theta );

                //resampler -> UpdateXest();
                //resampler -> SetXold();

                /*
      std::cout << "Reference image at loop " << i+1 << ": " << std::endl;
      std::cout << "Pointer : " << resampler ->GetOutput() << std::endl;
      std::cout << "Region : " << resampler ->GetOutput() ->GetLargestPossibleRegion() << std::endl;
      */

                std::cout << "New reference size : " << resampler -> GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels() << std::endl;


                resampler -> SetReferenceImage( resampler -> GetOutput() );
                resampler -> Update();

                criterion = resampler -> GetCriterionValue();

                std::cout << std::endl << "Outer loop criterion = " << criterion << std::endl << std::endl;

                //resampler -> Print(std::cout);

                if ( strcmp(debugDir,"") != 0 )
                {
                    std::ostringstream ss;
                    ss << debugDir << "/" << debugfilename << (i+1) << ".nii.gz";

                    std::cout << "##################################################################" << std::endl;
                    std::cout << "Debug mode : writing image "<< ss.str() << std::endl;
                    std::cout << "##################################################################" << std::endl << std::endl;

                    debugwriter -> SetFileName( ss.str() );
                    debugwriter -> SetInput( resampler -> GetOutput() );
                    debugwriter->Update();
                }

                bool bCSV=true;
                if(bCSV==true)
                {
                    //Save TVEnergy in CSV file
                    const char * csvFileName="/home/tourbier/Desktop/NewbornWithGapForConvergence/tv_energy_inf.csv";
                    bool writeHeaders = false;

                    std::ifstream fin;
                    fin.open(csvFileName,std::ios_base::out | std::ios_base::app);

                    if(fin.is_open())
                    {
                        //Test if the file is empty. If so, we add an extra line for headers
                        //std::cout << "Test if CSV  is empty. If so, we add an extra line for headers." << std::endl;
                        int csvLength;

                        fin.seekg(0, std::ios::end);
                        csvLength = fin.tellg();

                        fin.close();

                        if(csvLength == 0)
                        {
                            writeHeaders = true;
                            std::cout << "Write headers in CSV" << std::endl;
                        }
                        else
                        {
                            std::cout << "CSV empty ( length : " << int2str(csvLength) << std::endl;
                        }


                        //NOT WORKING ON MAC
                        /*if(fin.peek() == std::std::ifstream::traits_type::eof())
                        {
                            writeHeaders = true;
                            std::cout << "Write headers in CSV" << std::endl;
                        }
                        fin.close();*/
                    }
                    else
                    {
                        std::cout << "CSV file opening failed." << std::endl;
                    }

                    std::ofstream fout(csvFileName, std::ios_base::out | std::ios_base::app);

                    if(writeHeaders)
                    {
                        fout  << "Algo" << "," << "lambda" << "," << "step-scale" << "," << "deltat" << "," << "gamma" << ",";
                        fout << "Innerloops" << "," << "InnerThreshold" << "," << "Outerloops" << "," << "OuterThreshold" << ",";
                        fout << "TVEnergy";
                        fout << std::endl;
                    }

                    fout << "TV" << "," << lambda << "," << stepScale << "," << deltat << "," << gamma << ",";
                    fout << iter << "," << innerConvThreshold << "," << numberOfLoops << "," << outerConvThreshold << ",";

                    //Add value of TV energy to CSV
                    fout << resampler -> GetTVEnergy();

                    fout << std::endl;
                    fout.close();

                    std::cout << "Metrics saved in CSV" << std::endl;


                    std::cout << std::endl << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

                    //
                }

                if(criterion < outerConvThreshold)
                {
                    std::cout << "Outer loop has converged after "<< resampler -> GetCurrentOuterIteration() <<" iterations! ( last value = " << criterion << " )"<< std::endl;
                    break;
                }

                std::cout << "**************************************************************************" << std::endl << std::endl;

            }// End of outer loops

            //Update bregman variable
            //resampler -> UpdateZ();

        }//End of bregmann loops

        //Save TVEnergy in CSV file
        const char * csvFileName="/home/tourbier/Desktop/NewbornWithGapForConvergence/tv_energy_inf.csv";
        bool writeHeaders = false;

        std::ifstream fin;
        fin.open(csvFileName,std::ios_base::out | std::ios_base::app);

        if(fin.is_open())
        {
            //Test if the file is empty. If so, we add an extra line for headers
            //std::cout << "Test if CSV  is empty. If so, we add an extra line for headers." << std::endl;
            int csvLength;

            fin.seekg(0, std::ios::end);
            csvLength = fin.tellg();

            fin.close();

            if(csvLength == 0)
            {
                writeHeaders = true;
                std::cout << "Write headers in CSV" << std::endl;
            }
            else
            {
                std::cout << "CSV empty ( length : " << int2str(csvLength) << std::endl;
            }


            //NOT WORKING ON MAC
            /*if(fin.peek() == std::std::ifstream::traits_type::eof())
            {
                writeHeaders = true;
                std::cout << "Write headers in CSV" << std::endl;
            }
            fin.close();*/
        }
        else
        {
            std::cout << "CSV file opening failed." << std::endl;
        }

        std::ofstream fout(csvFileName, std::ios_base::out | std::ios_base::app);

        if(writeHeaders)
        {
            fout << "Date" << "," << "Algo" << "," << "lambda" << "," << "step-scale" << "," << "deltat" << "," << "gamma" << ",";
            fout << "Innerloops" << "," << "InnerThreshold" << "," << "Outerloops" << "," << "OuterThreshold" << ",";
            fout << "TVEnergy";
            fout << std::endl;
        }

        fout << mialsrtk::getRealCurrentDate() << "," << "TV" << "," << lambda << "," << stepScale << "," << deltat << "," << gamma << ",";
        fout << iter << "," << innerConvThreshold << "," << numberOfLoops << "," << outerConvThreshold << ",";

        //Add value of TV energy to CSV
        fout << resampler -> GetTVEnergy();

        fout << std::endl;
        fout.close();

        std::cout << "Metrics saved in CSV" << std::endl;


        std::cout << std::endl << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

        //

        std::cout << "h1" << std::endl;
        end_time_unix = mialsrtk::getTime();;
        std::cout << "h2" << std::endl;
        diff_time_unix = end_time_unix - start_time_unix;

        mialsrtk::printTime("TV (IGD)",diff_time_unix);

        double innerLoopRunTime = resampler -> GetInnerOptTime();
        double initCostFunctionRunTime = resampler -> GetInitTime();

        mialsrtk::printTime("Initialization for",initCostFunctionRunTime);
        mialsrtk::printTime("Inner loop",innerLoopRunTime);

        // Write image

        WriterType::Pointer writer =  WriterType::New();
        writer -> SetFileName( outImage );
        writer -> SetInput( resampler -> GetOutput() );
        //writer -> SetInput( outputImage );

        if ( strcmp(outImage,"") != 0)
        {
            std::cout << "Writing " << outImage << " ... ";
            writer->Update();
            std::cout << "done." << std::endl;
        }

        // Write transforms

        typedef itk::TransformFileWriter TransformWriterType;

        if ( outTransform.size() > 0 && updateMotionSwitchArg.isSet() )
        {
            for (unsigned int i=0; i<numberOfImages; i++)
            {
                TransformWriterType::Pointer transformWriter = TransformWriterType::New();
                transformWriter -> SetInput( transforms[i] );
                transformWriter -> SetFileName ( outTransform[i].c_str() );

                try
                {
                    std::cout << "Writing " << outTransform[i].c_str() << " ... " ; std::cout.flush();
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

    } catch (TCLAP::ArgException &e)  // catch any exceptions
    { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

    return EXIT_SUCCESS;
}

