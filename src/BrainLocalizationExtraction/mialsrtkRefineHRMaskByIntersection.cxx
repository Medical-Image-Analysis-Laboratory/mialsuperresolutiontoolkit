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
#include "itkVersorRigid3DTransform.h"
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

/*Mialsrtktk includes*/
#include "mialsrtkSliceBySliceTransform.h"

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
#include "itkImageLinearConstIteratorWithIndex.h"

#include "itkBinaryThresholdImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkBinaryMorphologicalClosingImageFilter.h"
#include "itkBinaryMorphologicalOpeningImageFilter.h"

#include "itkBinaryDilateImageFilter.h"

// classes help the MRF/Gibbs filter to segment the image
#include "itkMRFImageFilter.h"
#include "itkImageClassifierBase.h"
#include "itkDistanceToCentroidMembershipFunction.h"
#include "itkMinimumDecisionRule.h"
#include "itkComposeImageFilter.h"

#include "itkImageGaussianModelEstimator.h"
#include "itkMahalanobisDistanceMembershipFunction.h"

// image storage and I/O classes
#include "itkSize.h"
#include "itkVector.h"


#include "crlMSTAPLEImageFilter.h"
//#include "../CRKit/Tools/MSTAPLE.h"
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

        std::vector< std::string > input;
        std::vector< std::string > mask;
        std::vector< std::string > outLRMask;
        std::vector< std::string > transform;

        const char *outMask = NULL;
        const char *refImage = NULL;

        const char *test = "undefined";

        std::vector< int > x1, y1, z1, x2, y2, z2;

        double start_time_unix, end_time_unix, diff_time_unix;

        // Parse arguments
        TCLAP::CmdLine cmd("Refine The super resolution mask based on the intersection of the mask.", ' ', "Unversioned");

        // Input LR images
        TCLAP::MultiArg<std::string> inputArg("i","input","Low-resolution image file",true,"string",cmd);
        
        // Input LR masks
        TCLAP::MultiArg<std::string> maskArg("m","mask","low-resolution image mask file",true,"string",cmd);

        // Input LR masks
        TCLAP::MultiArg<std::string> outLRMaskArg("O","output-lrmask","output low-resolution image mask file",true,"string",cmd);
        
        // Input motion parameters
        TCLAP::MultiArg<std::string> transArg("t","transform","transform file",true,"string",cmd);
        
        // Ouput HR image
        TCLAP::ValueArg<std::string> outArg  ("o","output-mask","Super resolution output mask",true,"","string",cmd);

        // Input reconstructed image for initialization
        TCLAP::ValueArg<std::string> refArg  ("r","reference","Reconstructed image for reference. "
                                              "Typically the output of SR program is used." ,true,"","string",cmd);

        // Radius of the structuring element (ball) used for dilation of each slice
        TCLAP::ValueArg<int> radiusDilationArg  ("","radius-dilation","Radius of the structuring element (ball) used for binary morphological dilation.",true,0,"int",cmd);

        TCLAP::SwitchArg stapleArg ("","use-staple","Use STAPLE for voting (Majority voting is used by default)", cmd, false);
        TCLAP::SwitchArg verboseArg("v","verbose","Verbose output (False by default)",cmd, false);
        //TCLAP::ValueArg<std::string>debugDirArg("","debug","Directory where  SR reconstructed image at each outer loop of the reconstruction optimization is saved",false,"","string",cmd);

        // Parse the argv array.
        cmd.parse( argc, argv );

        input = inputArg.getValue();
        mask = maskArg.getValue();
        outLRMask = outLRMaskArg.getValue();

        outMask = outArg.getValue().c_str();

        refImage = refArg.getValue().c_str();
        transform = transArg.getValue();
        bool verbose = verboseArg.getValue();
        //TODO: check if mask.size() == transform.size()

        // typedefs
        const   unsigned int    Dimension = 3;
        typedef float  PixelType;

        typedef itk::Image< PixelType, Dimension >  ImageType;
        typedef ImageType::Pointer                  ImagePointer;
        typedef std::vector<ImagePointer>           ImagePointerArray;

        typedef itk::Image< unsigned char, Dimension >  ImageMaskType;
        typedef itk::ImageFileReader< ImageMaskType > MaskReaderType;
        typedef itk::ImageMaskSpatialObject< Dimension > MaskType;

        typedef ImageType::RegionType               RegionType;
        typedef std::vector< RegionType >           RegionArrayType;
        
        typedef mialsrtk::SliceBySliceTransformBase< double, Dimension > TransformBaseType;
        typedef mialsrtk::SliceBySliceTransform< double, Dimension > TransformType;
        typedef TransformType::Pointer                          TransformPointer;

        // Register the SliceBySlice transform (a non-default ITK transform) with the TransformFactory of ITK
        itk::TransformFactory<TransformType>::RegisterTransform();

        typedef itk::ImageFileReader< ImageType >   ImageReaderType;
        typedef itk::ImageFileWriter< ImageType >   ImageWriterType;
        typedef itk::ImageFileWriter< ImageMaskType >   MaskWriterType;

        typedef itk::TransformFileReader     TransformReaderType;
        typedef TransformReaderType::TransformListType * TransformListType;

        // Rigid 3D transform definition (typically for reconstructions in adults)

        typedef itk::VersorRigid3DTransform< double > EulerTransformType;

        //typedef mialsrtk::ImageIntersectionCalculator<ImageType> IntersectionCalculatorType;
        //IntersectionCalculatorType::Pointer intersectionCalculator = IntersectionCalculatorType::New();

        // Interpolator used to compute the error metric between 2 registration iterations
        typedef itk::NearestNeighborInterpolateImageFunction<ImageMaskType,double>     NNMaskInterpolatorType;
        typedef itk::LinearInterpolateImageFunction<ImageType,double>     LinearImageInterpolatorType;
        //typedef itk::BSplineInterpolateImageFunction<ImageType,double>     BSplineInterpolatorType;

        typedef itk::ResampleImageFilter<ImageMaskType, ImageMaskType> ResamplerImageMaskFilterType;
        typedef itk::ResampleImageFilter<ImageType, ImageType> ResamplerImageFilterType;

        typedef itk::ExtractImageFilter<ImageMaskType, ImageMaskType> ExtractImageMaskFilterType;

        //typedef itk::CastImageFilter<ImageType,ImageMaskType> CasterType;

        // A helper class which creates an image which is perfect copy of the input image
        typedef itk::ImageDuplicator<ImageType> DuplicatorType;

        typedef itk::OrientImageFilter<ImageType,ImageType> OrientImageFilterType;
        typedef itk::OrientImageFilter<ImageMaskType,ImageMaskType> OrientImageMaskFilterType;

        typedef itk::ImageRegionIterator< ImageMaskType >  MaskIteratorType;
        typedef itk::ImageRegionIterator< ImageType >  IteratorType;
        typedef itk::ImageRegionIteratorWithIndex< ImageMaskType >  MaskIteratorTypeWithIndex;
        typedef itk::ImageRegionIteratorWithIndex< ImageType >  IteratorTypeWithIndex;

        typedef itk::AddImageFilter< ImageMaskType, ImageMaskType, ImageMaskType > AddImageMaskFilter;
        typedef itk::MultiplyImageFilter< ImageMaskType, ImageMaskType, ImageMaskType > MultiplyImageMaskFilterType;

        typedef itk::BinaryBallStructuringElement<ImageMaskType::PixelType, ImageMaskType::ImageDimension> StructuringElementType;
        typedef itk::BinaryDilateImageFilter <ImageMaskType, ImageMaskType, StructuringElementType> BinaryDilateImageFilterType;

        BinaryDilateImageFilterType::RadiusType radiusDilation2D;
        radiusDilation2D[0] = radiusDilationArg.getValue();
        radiusDilation2D[1] = radiusDilationArg.getValue();
        radiusDilation2D[2] = 0; // Dilate only in the plane of the slice

        BinaryDilateImageFilterType::RadiusType radiusDilation3D;
        radiusDilation3D[0] = radiusDilationArg.getValue();
        radiusDilation3D[1] = radiusDilationArg.getValue();
        radiusDilation3D[2] = radiusDilationArg.getValue();

        unsigned int numberOfImages = mask.size();

        std::vector<OrientImageFilterType::Pointer> orientImageFilter(numberOfImages);
        std::vector<OrientImageMaskFilterType::Pointer> orientMaskImageFilter(numberOfImages);

        std::vector< ImageMaskType::Pointer >     imageMasks(numberOfImages);
        std::vector< TransformPointer >     transforms(numberOfImages);

        std::vector<MaskType::Pointer> masks(numberOfImages);
        std::vector< RegionType >           rois(numberOfImages);

        ImageType::IndexType  roiIndex;
        ImageType::SizeType   roiSize;

        // Filter setup
        for (unsigned int i=0; i<numberOfImages; i++)
        {

            std::cout<<"Reading image : "<<input[i].c_str()<<std::endl;
            ImageReaderType::Pointer imageReader = ImageReaderType::New();
            imageReader -> SetFileName( input[i].c_str() );
            imageReader -> Update();

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
        }

        // Set the reference image
        std::cout<<"Reading the reference image : "<<refImage<<std::endl;
        ImageReaderType::Pointer refReader = ImageReaderType::New();
        refReader -> SetFileName( refImage );
        refReader -> Update();

        ImageType::Pointer referenceIm = refReader->GetOutput();

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



        std::cout << "==========================================================================" << std::endl << std::endl;

        //start_time_unix = mialtk::getTime();;
        
        //Refine the output SR mask by the intersection of all LR mask
        //Should create an output HR mask
        //Should resample each LR mask by applying the slice transform, then copy the given slice in the HR mask...
        // Think about how to design the intersection...

        ImageMaskType::Pointer outImageMask = ImageMaskType::New();
        outImageMask->SetRegions(referenceIm->GetLargestPossibleRegion());
        outImageMask->Allocate();
        outImageMask->FillBuffer(0.0);

        outImageMask->SetOrigin(referenceIm->GetOrigin());
        outImageMask->SetSpacing(referenceIm->GetSpacing());
        outImageMask->SetDirection(referenceIm->GetDirection());

        IteratorType itRefIm(referenceIm,referenceIm->GetLargestPossibleRegion());


        //Extract each slices of the stack for subsequent faster injection

        std::vector< std::vector< ImageMaskType::Pointer > > StacksOfMaskSlices(numberOfImages);
        std::vector< std::vector< itk::VersorRigid3DTransform<double>::Pointer > > transformsArray(numberOfImages);
        std::vector< std::vector< itk::Transform<double>::Pointer > > invTransformsArray(numberOfImages);
        if (verbose){
            std::cout << "Extract the slices... ";
        }
        for(unsigned s=0; s<numberOfImages ; s++)
        {
            ImageMaskType::IndexType inputIndex = rois[s].GetIndex();
            ImageMaskType::SizeType  inputSize  = rois[s].GetSize();

            unsigned int i=inputIndex[2] + inputSize[2];

            //StacksOfSlices[s].set_size(inputSize[2]);

            //Loop over images of the current stack
            for ( unsigned int i=inputIndex[2]; i < inputIndex[2] + inputSize[2]; i++ )
            {
                //std::cout << "process image # " << s << " slice #" << i << std::endl;

                ImageMaskType::RegionType wholeSliceRegion;
                wholeSliceRegion = rois[s];

                ImageMaskType::IndexType  wholeSliceRegionIndex = wholeSliceRegion.GetIndex();
                ImageMaskType::SizeType   wholeSliceRegionSize  = wholeSliceRegion.GetSize();

                wholeSliceRegionIndex[2]= i;
                wholeSliceRegionSize[2] = 1;

                wholeSliceRegion.SetIndex(wholeSliceRegionIndex);
                wholeSliceRegion.SetSize(wholeSliceRegionSize);

                //Extract slice
                ExtractImageMaskFilterType::Pointer sliceExtractor = ExtractImageMaskFilterType::New();
                sliceExtractor->SetExtractionRegion(wholeSliceRegion);
                sliceExtractor->SetInput(imageMasks[s]);
#if ITK_VERSION_MAJOR >= 4
                sliceExtractor->SetDirectionCollapseToIdentity(); // This is required.
#endif
                sliceExtractor->Update();

                StacksOfMaskSlices[s].push_back(sliceExtractor->GetOutput());

                itk::VersorRigid3DTransform<double>::Pointer sliceTransformPtr = itk::VersorRigid3DTransform<double>::New();
                sliceTransformPtr = static_cast< itk::VersorRigid3DTransform<double>::Pointer >(transforms[s] -> GetSliceTransform(i));

                itk::VersorRigid3DTransform<double>::Pointer sliceInvTransformPtr = itk::VersorRigid3DTransform<double>::New();
                sliceInvTransformPtr -> SetCenter(sliceTransformPtr->GetCenter());
                sliceInvTransformPtr -> SetMatrix(sliceTransformPtr->GetMatrix());
                sliceInvTransformPtr -> SetOffset(sliceTransformPtr->GetOffset());

                //itk::VersorRigid3DTransform<double> * sliceTransform = static_cast< itk::VersorRigid3DTransform<double> * >(transforms[s] -> GetSliceTransform(i));

                transformsArray[s].push_back(sliceTransformPtr);

                //itk::VersorRigid3DTransform<double> * inverseSliceTransform = static_cast< itk::VersorRigid3DTransform<double> * >(transforms[s] -> GetSliceTransform(i) -> Clone());
                //itk::VersorRigid3DTransform<double>::Pointer inverseSliceTransform = itk::VersorRigid3DTransform<double>::New();
                //transforms[s] -> GetSliceTransform(i) -> GetInverse(inverseSliceTransform);

                invTransformsArray[s].push_back(sliceInvTransformPtr->GetInverseTransform());
            }
        }
        if (verbose){
            std::cout << std::endl;
        }
        //Strictly speaking, this is not an injection process, but it's faster to do it this way
        ImageMaskType::PointType outputPoint;      //physical point in HR output image
        ImageMaskType::IndexType outputIndex;      //index in HR output image
        ImageMaskType::PointType transformedPoint; //Physical point location after applying affine transform
        itk::ContinuousIndex<double,3> inputContIndex;   //continuous index in LR image
        itk::ContinuousIndex<double,3>   interpolationContIndex;   //continuous index in LR image for interpolation (i.e. z = 0)
        //interpolationContIndex[2] = 0;

        int counter = 0;

        //Define a threshold for z coordinate based on FWHM = 2sqrt(2ln2)sigma = 2.3548 sigma
        float cst = sqrt(8*log(2.0));
        float deltaz = (2.0 / cst) * imageMasks[0]->GetSpacing()[2];

        int numberOfVoxels = outImageMask->GetLargestPossibleRegion().GetSize()[0]*outImageMask->GetLargestPossibleRegion().GetSize()[1]*outImageMask->GetLargestPossibleRegion().GetSize()[2];
        if (verbose){
        std::cout << "Number of voxels: " << numberOfVoxels << std::endl;
        }
        NNMaskInterpolatorType::Pointer nnInterpolator = NNMaskInterpolatorType::New();

        std::vector< ImageMaskType::Pointer > outImageMasks(numberOfImages);

        for(unsigned int s=0; s<numberOfImages; s++)
        {

            outImageMasks[s] = ImageMaskType::New();
            outImageMasks[s]->SetRegions(referenceIm->GetLargestPossibleRegion());
            outImageMasks[s]->Allocate();
            outImageMasks[s]->FillBuffer(0.0);

            outImageMasks[s]->SetOrigin(referenceIm->GetOrigin());
            outImageMasks[s]->SetSpacing(referenceIm->GetSpacing());
            outImageMasks[s]->SetDirection(referenceIm->GetDirection());
            
            //unsigned int sizeX = m_ImageArray[0]->GetLargestPossibleRegion().GetSize()[0];
            //unsigned int sizeY = m_ImageArray[0]->GetLargestPossibleRegion().GetSize()[1];

            ImageMaskType::IndexType inputIndex = rois[s].GetIndex();
            ImageMaskType::SizeType  inputSize  = rois[s].GetSize();

            //TODO: Can we parallelize this ?
            //Iteration over the slices of the LR images

            unsigned int i=inputIndex[2] + inputSize[2];

            //Loop over slices of the current stack
            for ( unsigned int i=inputIndex[2]; i < inputIndex[2] + inputSize[2]; i++ )
            {
                //std::cout << "process image # " << s << " slice #" << i << std::endl;

                ImageMaskType::RegionType wholeSliceRegion;
                wholeSliceRegion = rois[s];

                ImageMaskType::IndexType  wholeSliceRegionIndex = wholeSliceRegion.GetIndex();
                ImageMaskType::SizeType   wholeSliceRegionSize  = wholeSliceRegion.GetSize();

                wholeSliceRegionIndex[2]= i;
                wholeSliceRegionSize[2] = 1;

                wholeSliceRegion.SetIndex(wholeSliceRegionIndex);
                wholeSliceRegion.SetSize(wholeSliceRegionSize);

                //Extract slice
                ExtractImageMaskFilterType::Pointer sliceExtractor = ExtractImageMaskFilterType::New();
                sliceExtractor->SetExtractionRegion(wholeSliceRegion);
                sliceExtractor->SetInput(imageMasks[s]);
#if ITK_VERSION_MAJOR >= 4
                sliceExtractor->SetDirectionCollapseToIdentity(); // This is required.
#endif
                sliceExtractor->Update();

                ResamplerImageMaskFilterType::Pointer sliceResampler = ResamplerImageMaskFilterType::New();
                sliceResampler -> SetInput(sliceExtractor->GetOutput());
                sliceResampler -> SetTransform(invTransformsArray[s][i-inputIndex[2]]);
                sliceResampler -> SetInterpolator(nnInterpolator);
                sliceResampler -> SetOutputParametersFromImage(outImageMasks[s]);
                sliceResampler -> Update();

                AddImageMaskFilter::Pointer addMaskFilter = AddImageMaskFilter::New();
                addMaskFilter -> SetInput1(outImageMasks[s]);
                addMaskFilter -> SetInput2(sliceResampler->GetOutput());
                addMaskFilter -> Update();

                outImageMasks[s] = addMaskFilter->GetOutput();

                //MultiplyImageMaskFilterType::Pointer multMaskFilter = MultiplyImageMaskFilterType::New();
                //multMaskFilter -> SetInput1(outImageMasks[s]);
                //multMaskFilter -> SetInput2(sliceResampler->GetOutput());
                //multMaskFilter -> Update();

                //outImageMasks[s] = multMaskFilter->GetOutput();
            }

            MaskIteratorTypeWithIndex itOutStackMask(outImageMasks[s],outImageMasks[s]->GetLargestPossibleRegion());
            for(itOutStackMask.GoToBegin(); !itOutStackMask.IsAtEnd(); ++itOutStackMask)
            {
                if(itOutStackMask.Get()>0.0) itOutStackMask.Set(1.0);
            }

            std::stringstream ssFile;
            ssFile << "/home/tourbier/Desktop/DbgMasks/hrResMask_" << s << ".nii.gz";

            MaskWriterType::Pointer writer = MaskWriterType::New();
            writer -> SetInput(outImageMasks[s]);
            writer -> SetFileName(ssFile.str().c_str());
            writer -> Update();

            if(!stapleArg.isSet())
            {
                AddImageMaskFilter::Pointer addMaskFilter2 = AddImageMaskFilter::New();
                addMaskFilter2 -> SetInput1(outImageMask);
                addMaskFilter2 -> SetInput2(outImageMasks[s]);
                addMaskFilter2 -> Update();

                outImageMask = addMaskFilter2->GetOutput();
            }
            
        }



        std::stringstream ssFile2;
        ssFile2 << "/home/ch176971/Desktop/DbgMasks/hrResMask_all.nii.gz";

        MaskWriterType::Pointer writer2 = MaskWriterType::New();
        writer2 -> SetInput(outImageMask);
        writer2 -> SetFileName(ssFile2.str().c_str());
        writer2 -> Update();

        //Perform STAPLE to refine the HR brain mask
        typedef itk::Image< double , 4 > OutputSTAPLEImageType;
        typedef crl::MSTAPLEImageFilter<ImageMaskType, OutputSTAPLEImageType> MSTAPLEFilterType;
        MSTAPLEFilterType::Pointer staple = MSTAPLEFilterType::New();
        staple->SetVerbose(verbose);

        ImageType::Pointer outStapleImage = ImageType::New();
        outStapleImage->SetRegions(referenceIm->GetLargestPossibleRegion());
        outStapleImage->Allocate();
        outStapleImage->FillBuffer(0.0);
        outStapleImage->SetOrigin(referenceIm->GetOrigin());
        outStapleImage->SetSpacing(referenceIm->GetSpacing());
        outStapleImage->SetDirection(referenceIm->GetDirection());

        if(stapleArg.isSet())
        {
            if (verbose){
                std::cout << "Perform STAPLE..." << std::endl;
            }
            //typedef itk::Image< double , 4 > OutputSTAPLEImageType;
            //typedef crl::MSTAPLEImageFilter<ImageMaskType, OutputSTAPLEImageType> MSTAPLEFilterType;

            bool stapleMAP = false;
            bool stationaryPriorSet = false;
            bool initialExpertPerformanceSet = false;

            int underflowProtection = 0; //Underflow protection : 0 none, 1 strong, 2 extreme. Controls computation with none, some, or extremely extended precision. Useful with large numbers of input segmentations.
            bool useCompression = true; //Write out the reference standard using compression.
            bool assignConsensusVoxels = true; //Determines if voxels with label estimates that are the same amongst all inputs are assigned directly or used in the computation.
            bool startAtEStep = false; //Start at the E Step by estimating the reference standard (if true). Start at the M Step by estimating the parameters from the initial reference standard (if false).
            //bool startAtMStep = true; //Start at the M Step by estimating the parameters from the initial reference standard.

            double alpha = 0.0; //Alpha parameter of the beta distribution for MAP.
            double beta = 0.0; //Beta parameter of the beta distribution for MAP.

            int maxiterations = -1; //Maximum number of iterations. The E-M algorithm is terminated after this number of iterations has been computed.

            double relativeConvergence = 5e-07; //Relative convergence threshold, used to terminate the E-M algorithm when it has converged. Convergence is defined by relative changes in the mean trace of expert performance below this level.
            double stationaryWeight = 0.01; //Stationary prior weight, used to set the weight of stationary prior for each tissue class label with respect to spatially varying prior

            //MSTAPLEFilterType::Pointer staple = MSTAPLEFilterType::New();
            //staple->SetUnderflowProtection( underflowProtection );
            //staple->SetUseWriteCompression( useCompression );
            staple->SetAssignConsensusVoxels( assignConsensusVoxels );
            //staple->SetStartAtEStep( startAtEStep );

            //staple->SetMAPStaple(stapleMAP);
            
            if (stapleMAP)
            {
                staple->SetMAPAlpha(alpha);
                staple->SetMAPBeta(beta);
            }

            for(int s=0; s<numberOfImages; s++)
            {
                staple->SetInput(s,const_cast<ImageMaskType*>(outImageMasks[s].GetPointer()));
            }

            // Now apply the optional arguments to the object.
            if (maxiterations != -1) {
                staple->SetMaximumIterations(maxiterations);
            }

            //staple->SetRelativeConvergenceThreshold( relativeConvergence );
            //staple->SetStationaryPriorWeight( stationaryWeight );
            if ((stationaryWeight>1) || (stationaryWeight<0))
            {
                std::cerr << "Weight is not between 0 and 1" << std::endl;
                return EXIT_FAILURE;
            }
            staple->SetNumberOfThreads(8);
            // Execute STAPLE
            try{
                staple->Update();
            }
            catch( itk::ExceptionObject & err )
            {
                std::cerr << "ExceptionObject caught !" << std::endl;
                std::cerr << err << std::endl;
            }

            //staple->Print(std::cout);
            std::stringstream ssFileS;
            ssFileS << "/home/ch176971/Desktop/DbgMasks/staple2.nii.gz";

            itk::ImageFileWriter< OutputSTAPLEImageType >::Pointer writerS = itk::ImageFileWriter< OutputSTAPLEImageType >::New();
            writerS -> SetInput(staple->GetOutput());
            writerS -> SetFileName(ssFileS.str().c_str());
            writerS -> Update();


            //ImageMaskType::RegionType outRegion = outSTAPLE -> GetLargestPossibleRegion();

            //ImageMaskType::IndexType outIndex = outRegion.GetIndex();
            //ImageMaskType::SizeType  outSize  = outRegion.GetSize();

            OutputSTAPLEImageType::RegionType outStapleRegion = staple->GetOutput()->GetBufferedRegion();
            const unsigned int numberOfLabels = outStapleRegion.GetSize()[3];

            typedef itk::ImageLinearConstIteratorWithIndex< OutputSTAPLEImageType >  OutputSTAPLEIteratorTypeWithIndex;

            OutputSTAPLEIteratorTypeWithIndex itOutputSTAPLE(staple->GetOutput(), outStapleRegion);
            itOutputSTAPLE.SetDirection(3); //Walk along label dimension
            itOutputSTAPLE.GoToBegin();

            ImageMaskType::IndexType index3D;
            OutputSTAPLEImageType::IndexType index4D;



            //IteratorTypeWithIndex  itOutStapleIm(outStapleImage,outStapleImage->GetLargestPossibleRegion());
            //itOutStapleIm.GoToBegin();

            while(!itOutputSTAPLE.IsAtEnd())
            {
                itOutputSTAPLE.GoToBeginOfLine();

                while(!itOutputSTAPLE.IsAtEndOfLine())
                {
                    index4D = itOutputSTAPLE.GetIndex();

                    /*
                    if(index4D[3]==1 && itOutputSTAPLE.Get() >= 0.5)//Inside brain label
                    {
                        index3D[0] = index4D[0];
                        index3D[1] = index4D[1];
                        index3D[2] = index4D[2];

                        outImageMask->SetPixel( index3D, 1.0);
                    }
                    */

                    if(index4D[3]==1 && itOutputSTAPLE.Get() > 0.0)//Inside brain label
                    {
                        index3D[0] = index4D[0];
                        index3D[1] = index4D[1];
                        index3D[2] = index4D[2];

                        outImageMask->SetPixel( index3D, 1.0);
                        outStapleImage->SetPixel(index3D, itOutputSTAPLE.Get());

                    }

                    ++itOutputSTAPLE;
                }
                
                itOutputSTAPLE.NextLine();
            }
        }
        else //Perform majority voting to refine the HR brain mask
        {
            int mvThreshold = ceil(0.5 * numberOfImages);
            if (verbose){
                std::cout << "Perform Majority Voting ,  thresh = " << mvThreshold << " ..." << std::endl;
            }
            MaskIteratorTypeWithIndex itOutImageMask(outImageMask,outImageMask->GetLargestPossibleRegion());

            for(itOutImageMask.GoToBegin(); !itOutImageMask.IsAtEnd(); ++itOutImageMask)
            {
                if((numberOfImages % 2 == 0) && (itOutImageMask.Get() > mvThreshold))
                {
                    itOutImageMask.Set(1.0);
                }
                else if((numberOfImages % 2 != 0) && (itOutImageMask.Get() >= mvThreshold))
                {
                    itOutImageMask.Set(1.0);
                }
                else
                {
                    itOutImageMask.Set(0.0);
                }
            }
        }


        std::stringstream ssFile3;
        ssFile3 << "/home/tourbier/Desktop/DbgMasks/hrResMask_beforeMRF.nii.gz";

        MaskWriterType::Pointer writer3 = MaskWriterType::New();
        writer3 -> SetInput(outImageMask);
        writer3 -> SetFileName(ssFile3.str().c_str());
        writer3 -> Update();

        std::stringstream ssFile32;
        ssFile32 << "/home/tourbier/Desktop/DbgMasks/hrResMask_staple_beforeMRF.nii.gz";

        ImageWriterType::Pointer writer32 = ImageWriterType::New();
        writer32 -> SetInput(outStapleImage);
        writer32 -> SetFileName(ssFile32.str().c_str());
        writer32 -> Update();

        typedef itk::FixedArray< unsigned char,  1> FixedArrayMaskType;
        typedef itk::Image< FixedArrayMaskType, Dimension> ArrayImageMaskType;

        typedef itk::Image<itk::Vector<unsigned char,1>, Dimension> VectorImageMaskType;
        typedef itk::Image<itk::Vector<float,1>, Dimension> VectorImageType;

        // We convert the input into vector images
        //

        typedef VectorImageType::PixelType VectorImagePixelType;

        VectorImageType::Pointer vecImage = VectorImageType::New();
        vecImage->SetLargestPossibleRegion( outStapleImage->GetLargestPossibleRegion() );
        vecImage->SetBufferedRegion( outStapleImage->GetBufferedRegion() );
        vecImage->SetOrigin(outStapleImage->GetOrigin());
        vecImage->SetSpacing(outStapleImage->GetSpacing());
        vecImage->SetDirection(outStapleImage->GetDirection());
        vecImage->Allocate();

        //enum { VecImageDimension = VectorImageMaskType::ImageDimension };

        typedef itk::ImageRegionIterator< VectorImageType > VectorImageIterator;
        VectorImageIterator vecIt( vecImage, vecImage->GetBufferedRegion() );
        vecIt.GoToBegin();

        typedef itk::ImageRegionIterator< ImageType > ImageIterator;
        ImageIterator outIt( outStapleImage, outStapleImage->GetBufferedRegion() );
        outIt.GoToBegin();

         typedef itk::ImageRegionIterator< ImageMaskType > ImageMaskIterator;
        ImageMaskIterator outItMV(outImageMask,outImageMask->GetLargestPossibleRegion());
        outItMV.GoToBegin();

        //Set up the vector to store the image  data
        typedef VectorImageType::PixelType     DataVector;
        DataVector   dblVec;
        if( stapleArg.isSet())
        {
            while ( !vecIt.IsAtEnd() )
            {

                dblVec[0] = outIt.Get();
                vecIt.Set(dblVec);
                ++vecIt;
                ++outIt;
            }
        }
        else
        {
            while ( !vecIt.IsAtEnd() )
            {
                dblVec[0] = (float) outItMV.Get();
                vecIt.Set(dblVec);
                ++vecIt;
                ++outItMV;
            }
        }

        //----------------------------------------------------------------------
        //Set membership function (Using the statistics objects)
        //----------------------------------------------------------------------
        typedef VectorImageType::PixelType         VectorImagePixelType;

        //typedef itk::Statistics::MahalanobisDistanceMembershipFunction< VectorImageMaskPixelType >  MembershipFunctionType;
        typedef itk::Statistics::DistanceToCentroidMembershipFunction< VectorImagePixelType > MembershipFunctionType;

        typedef MembershipFunctionType::Pointer MembershipFunctionPointer;
        typedef std::vector< MembershipFunctionPointer >    MembershipFunctionPointerVector;

        //----------------------------------------------------------------------
        //Set the decision rule
        //----------------------------------------------------------------------
        typedef itk::Statistics::DecisionRule::Pointer DecisionRuleBasePointer;
        typedef itk::Statistics::MinimumDecisionRule DecisionRuleType;
        DecisionRuleType::Pointer  myDecisionRule = DecisionRuleType::New();
        if (verbose){
            std::cout << " site 3 " << std::endl;
        }
        //----------------------------------------------------------------------
        // Set the classifier to be used and assigne the parameters for the
        // supervised classifier algorithm except the input image which is
        // grabbed from the Gibbs application pipeline.
        //----------------------------------------------------------------------
        //---------------------------------------------------------------------
        //  Software Guide : BeginLatex
        //
        //  Then we define the classifier that is needed
        //  for the Gibbs prior model to make correct segmenting decisions.
        //
        //  Software Guide : EndLatex
        // Software Guide : BeginCodeSnippet
        typedef itk::ImageClassifierBase< VectorImageType,ImageMaskType > ClassifierType;
        typedef ClassifierType::Pointer                    ClassifierPointer;
        ClassifierPointer myClassifier = ClassifierType::New();
        // Software Guide : EndCodeSnippet
        // Set the Classifier parameters
        myClassifier->SetNumberOfClasses(2);
        // Set the decison rule
        myClassifier->SetDecisionRule((DecisionRuleBasePointer) myDecisionRule );
        //Add the membership functions
        //for( unsigned int i=0; i<2; i++ )
        //{
        //    myClassifier->AddMembershipFunction( membershipFunctions[i] );
        //}

        double meanDistance = 0.5;
        MembershipFunctionType::CentroidType centroid(1);

        for(unsigned int k = 0; k < 2; k++)
        {
            MembershipFunctionType::Pointer membershipFunction = MembershipFunctionType::New();
            centroid[0] = k;
            membershipFunction->SetCentroid( centroid );
            myClassifier -> AddMembershipFunction( membershipFunction );
        }

        typedef itk::MRFImageFilter<  VectorImageType,  ImageMaskType> MRFImageFilterType;

        //Set the neighborhood radius.
        //For example, a neighborhood radius of 2 in a 3D image will result in a clique of size 5x5x5.
        //A neighborhood radius of 1 will result in a clique of size 3x3x3.
        int radius = 3;
        MRFImageFilterType::NeighborhoodRadiusType radiusMRF;
        radiusMRF[0] = radius;
        radiusMRF[1] = radius;
        radiusMRF[2] = 1;

        int numberOfNeighbors = (radius*2+1) * (radius*2+1) * (1*2+1);

        MRFImageFilterType::Pointer mrfFilter = MRFImageFilterType::New();

        mrfFilter -> SetInput(vecImage);

        mrfFilter -> SetNumberOfClasses(2);
        mrfFilter -> SetMaximumNumberOfIterations(50);
        mrfFilter -> SetErrorTolerance(1e-7);

        mrfFilter -> SetSmoothingFactor(30);

        mrfFilter -> SetNeighborhoodRadius(radiusMRF);

        mrfFilter->SetNumberOfThreads(8);

        std::vector< double > weights(numberOfNeighbors,1.0);
        /*
        weights.push_back(1.5);
        weights.push_back(2.0);
        weights.push_back(1.5);
        weights.push_back(2.0);
        weights.push_back(0.0); // This is the central pixel
        weights.push_back(2.0);
        weights.push_back(1.5);
        weights.push_back(2.0);
        weights.push_back(1.5);
        weights.push_back(1.5);
        weights.push_back(2.0);
        weights.push_back(1.5);
        weights.push_back(2.0);
        weights.push_back(0.0); // This is the central pixel
        weights.push_back(2.0);
        weights.push_back(1.5);
        weights.push_back(2.0);
        weights.push_back(1.5);
        weights.push_back(1.5);
        weights.push_back(2.0);
        weights.push_back(1.5);
        weights.push_back(2.0);
        weights.push_back(0.0); // This is the central pixel
        weights.push_back(2.0);
        weights.push_back(1.5);
        weights.push_back(2.0);
        weights.push_back(1.5);
        */

        double totalWeight = 0;
        for(std::vector< double >::const_iterator wcIt = weights.begin();
            wcIt != weights.end(); ++wcIt )
          {
          totalWeight += *wcIt;
          }
        for(std::vector< double >::iterator wIt = weights.begin();
            wIt != weights.end(); ++wIt )
          {
          *wIt = static_cast< double > ( (*wIt) * meanDistance / (2 * totalWeight));
          }
        mrfFilter->SetMRFNeighborhoodWeight( weights );

        mrfFilter -> SetClassifier(myClassifier);
        if (verbose){
            std::cout << "Run Markov Random Field Filtering... "; std::cout.flush();
        }
        mrfFilter -> Update();
        if (verbose){
            std::cout << "Number of Iterations : ";
            std::cout << mrfFilter->GetNumberOfIterations() << std::endl;
            std::cout << "Stop condition: " << std::endl;
            std::cout << "  (1) Maximum number of iterations " << std::endl;
            std::cout << "  (2) Error tolerance:  "  << std::endl;
            std::cout << mrfFilter->GetStopCondition() << std::endl;
        }
        //Set up the vector to store the image  data

        typedef itk::ImageRegionIterator< ImageMaskType > ImageMaskIterator;

        ImageMaskIterator outMRFIt( mrfFilter->GetOutput(), mrfFilter->GetOutput()->GetBufferedRegion() );
        outMRFIt.GoToBegin();

        ImageMaskIterator outMaskIt( outImageMask, outImageMask->GetBufferedRegion() );
        outMaskIt.GoToBegin();

        while ( !outMRFIt.IsAtEnd() )
        {
            outMaskIt.Set(outMRFIt.Get());
            ++outMaskIt;
            ++outMRFIt;
        }


        /*
        typedef itk::MRFImageFilter<  ArrayImageMaskType,  ImageMaskType> MRFImageFilterType;

        //Set the neighborhood radius.
        //For example, a neighborhood radius of 2 in a 3D image will result in a clique of size 5x5x5.
        //A neighborhood radius of 1 will result in a clique of size 3x3x3.
        MRFImageFilterType::NeighborhoodRadiusType radiusMRF;
        radiusMRF[0] = 2;
        radiusMRF[1] = 2;
        radiusMRF[2] = 2;

        typedef itk::Statistics::MinimumDecisionRule DecisionRuleType;
        DecisionRuleType::Pointer decisionRule = DecisionRuleType::New();

        typedef itk::ImageClassifierBase< ArrayImageMaskType, ImageMaskType> ClassifierType;
        ClassifierType::Pointer classifier = ClassifierType::New();
        classifier -> SetDecisionRule(decisionRule);


        typedef itk::Statistics::DistanceToCentroidMembershipFunction< FixedArrayMaskType > MembershipFunctionType;

        double meanDistance = 0.5;
        MembershipFunctionType::CentroidType centroid(1);

        for(unsigned int k = 0; k < 2; k++)
        {
            MembershipFunctionType::Pointer membershipFunction = MembershipFunctionType::New();
            centroid[0] = k;
            classifier -> AddMembershipFunction( membershipFunction );
        }

        typedef itk::ComposeImageFilter< ImageMaskType, ArrayImageMaskType > MaskToArrayFilterType;
        MaskToArrayFilterType::Pointer maskToArrayFilter = MaskToArrayFilterType::New();
        maskToArrayFilter -> SetInput(outImageMask.GetPointer());
        maskToArrayFilter -> Update();

//        typedef itk::CastImageFilter< ImageMaskType, ArrayImageMaskType> CastMRFImageFilterType;
//        CastMRFImageFilterType::Pointer caster = CastMRFImageFilterType::New();
//        caster -> SetInput(outImageMask.GetPointer());
//        caster -> Update();

//        typedef itk::CastImageFilter< ArrayImageMaskType, ImageMaskType> CastBackMRFImageFilterType;
//        CastBackMRFImageFilterType::Pointer casterBack = CastBackMRFImageFilterType::New();
//        casterBack -> SetInput(caster->GetOutput());
//        casterBack -> Update();

//        std::stringstream ssFileMRFcast;
//        ssFileMRFcast << "/home/tourbier/Desktop/DbgMasks/hrResMask_castMRF.nii.gz";

//        MaskWriterType::Pointer writerMRFcast = MaskWriterType::New();
//        writerMRFcast -> SetInput(casterBack->GetOutput());
//        writerMRFcast -> SetFileName(ssFileMRFcast.str().c_str());
//        writerMRFcast -> Update();

        MRFImageFilterType::Pointer mrfFilter = MRFImageFilterType::New();

        mrfFilter -> SetInput(maskToArrayFilter->GetOutput());

        mrfFilter -> SetNumberOfClasses(2);
        mrfFilter -> SetMaximumNumberOfIterations(50);
        mrfFilter -> SetErrorTolerance(1e-7);

        mrfFilter -> SetSmoothingFactor(1.0);

        mrfFilter -> SetNeighborhoodRadius(radiusMRF);

        mrfFilter -> SetClassifier(classifier);

        std::cout << "Run Markov Random Field Filtering... "; std::cout.flush();

        mrfFilter -> Update();

        */

        std::stringstream ssFileMRF;
        ssFileMRF << "/home/tourbier/Desktop/DbgMasks/hrResMask_afterMRF.nii.gz";

        MaskWriterType::Pointer writerMRF = MaskWriterType::New();
        writerMRF -> SetInput(outImageMask.GetPointer());
        writerMRF -> SetFileName(ssFileMRF.str().c_str());
        writerMRF -> Update();


        if (verbose){
            std::cout << "done." << std::endl;
        }

        //        //Fill in holes and gaps about the size of the structuring element
        //        typedef itk::BinaryBallStructuringElement<ImageMaskType::PixelType, ImageMaskType::ImageDimension> StructuringElementType;

        //        unsigned int radiusCl = 4;

        //        if(!stapleArg.isSet()) radiusCl = 4;

        //        StructuringElementType structuringElementCl;
        //        structuringElementCl.SetRadius(radiusCl);
        //        structuringElementCl.CreateStructuringElement();

        //        //Fill in 'holes' or 'gaps'
        //        typedef itk::BinaryMorphologicalClosingImageFilter <ImageMaskType, ImageMaskType, StructuringElementType>   BinaryMorphologicalClosingImageFilterType;
        //        BinaryMorphologicalClosingImageFilterType::Pointer closingFilter = BinaryMorphologicalClosingImageFilterType::New();
        //        closingFilter->SetInput(outImageMask);
        //        closingFilter->SetKernel(structuringElementCl);
        //        closingFilter->SetForegroundValue(1.0);
        //        closingFilter->Update();

        //        std::stringstream ssFile4;
        //        ssFile4 << "/home/ch176971/Desktop/DbgMasks/hrResMask_all_mv_afterClosing.nii.gz";

        //        MaskWriterType::Pointer writer4 = MaskWriterType::New();
        //        writer4 -> SetInput(closingFilter->GetOutput());
        //        writer4 -> SetFileName(ssFile4.str().c_str());
        //        writer4 -> Update();

        //        //Extract the largest connected component - discard the others (remove particles)
        //        typedef itk::ConnectedComponentImageFilter<ImageMaskType,ImageMaskType> ConnectedComponentImageFilterType;
        //        ConnectedComponentImageFilterType::Pointer connectedComponentFilter = ConnectedComponentImageFilterType::New();
        //        connectedComponentFilter -> SetInput(closingFilter->GetOutput());
        //        connectedComponentFilter -> Update();

        //        //Sort the connected components by size - label #1 is the largest, corresponding to the brain
        //        typedef itk::RelabelComponentImageFilter<ImageMaskType, ImageMaskType> RelabelComponentImageFilterType;
        //        RelabelComponentImageFilterType::Pointer relabelComponentFilter = RelabelComponentImageFilterType::New();
        //        relabelComponentFilter -> SetInput(connectedComponentFilter->GetOutput());
        //        //relabelComponentFilter -> SortByObjectSizeOn();
        //        relabelComponentFilter -> Update();

        //        typedef itk::BinaryThresholdImageFilter< ImageMaskType, ImageMaskType > BinaryThresholdImageFilterType;
        //        BinaryThresholdImageFilterType::Pointer thresholder = BinaryThresholdImageFilterType::New();
        //        thresholder -> SetInput(relabelComponentFilter->GetOutput());
        //        thresholder -> SetOutsideValue( 0.0 );
        //        thresholder -> SetInsideValue( 1.0 );
        //        thresholder -> SetLowerThreshold( 1.0 );
        //        thresholder -> SetUpperThreshold( 3.0 );
        //        thresholder -> Update();

        //        std::stringstream ssFile4b;
        //        ssFile4b << "/home/ch176971/Desktop/DbgMasks/hrResMask_all_mv_afterThresholder.nii.gz";

        //        MaskWriterType::Pointer writer4b = MaskWriterType::New();
        //        writer4b -> SetInput(thresholder->GetOutput());
        //        writer4b -> SetFileName(ssFile4b.str().c_str());
        //        writer4b -> Update();

        //        //Smooth the HR brain mask
        //        unsigned int radiusOp = 3;

        //        if(!stapleArg.isSet()) radiusOp = 3;

        //        StructuringElementType structuringElementOp;
        //        structuringElementOp.SetRadius(radiusOp);
        //        structuringElementOp.CreateStructuringElement();


        //        //Remove any 'small obects' about the size of the structuring element
        //        typedef itk::BinaryMorphologicalOpeningImageFilter <ImageMaskType, ImageMaskType, StructuringElementType>   BinaryMorphologicalOpeningImageFilterType;
        //        BinaryMorphologicalOpeningImageFilterType::Pointer openingFilter = BinaryMorphologicalOpeningImageFilterType::New();
        //        openingFilter->SetInput(thresholder->GetOutput());
        //        openingFilter->SetKernel(structuringElementOp);
        //        openingFilter->SetForegroundValue(1.0);
        //        openingFilter->Update();

        //        closingFilter->SetInput(openingFilter->GetOutput());
        //        closingFilter->SetKernel(structuringElementOp);
        //        closingFilter->SetForegroundValue(1.0);
        //        closingFilter->Update();

        //        std::stringstream ssFile5;
        //        ssFile5 << "/home/ch176971/Desktop/DbgMasks/hrResMask_all_mv_afterSmoothing.nii.gz";

        //        MaskWriterType::Pointer writer5 = MaskWriterType::New();
        //        writer5 -> SetInput(closingFilter->GetOutput());
        //        writer5 -> SetFileName(ssFile5.str().c_str());
        //        writer5 -> Update();

        //Refine the masks of LR stacks

        std::vector< ImageMaskType::Pointer > outLRImageMasks(numberOfImages);

        for(unsigned int s=0; s<numberOfImages; s++)
        {

            outLRImageMasks[s] = ImageMaskType::New();
            outLRImageMasks[s]->SetRegions(imageMasks[s]->GetLargestPossibleRegion());
            outLRImageMasks[s]->Allocate();
            outLRImageMasks[s]->FillBuffer(0.0);

            outLRImageMasks[s]->SetOrigin(imageMasks[s]->GetOrigin());
            outLRImageMasks[s]->SetSpacing(imageMasks[s]->GetSpacing());
            outLRImageMasks[s]->SetDirection(imageMasks[s]->GetDirection());
            if (verbose){
                std::cout << "Infos of Image LR # " << s << std::endl;
                std::cout << "Origin : " << imageMasks[s]->GetOrigin() << std::endl;
                std::cout << "Spacing : " << imageMasks[s]->GetSpacing() << std::endl;
                std::cout << "Direction : " << imageMasks[s]->GetDirection() << std::endl;
            }
            //unsigned int sizeX = m_ImageArray[0]->GetLargestPossibleRegion().GetSize()[0];
            //unsigned int sizeY = m_ImageArray[0]->GetLargestPossibleRegion().GetSize()[1];

            ImageMaskType::IndexType inputIndex = rois[s].GetIndex();
            ImageMaskType::SizeType  inputSize  = rois[s].GetSize();


            //Image of ones to select slice
            ImageMaskType::Pointer ones = ImageMaskType::New();
            ones->SetRegions(imageMasks[s]->GetLargestPossibleRegion());
            ones->Allocate();
            ones->FillBuffer(1.0);

            ones->SetOrigin(imageMasks[s]->GetOrigin());
            ones->SetSpacing(imageMasks[s]->GetSpacing());
            ones->SetDirection(imageMasks[s]->GetDirection());

            //TODO: Can we parallelize this ?
            //Iteration over the slices of the LR images

            unsigned int i=inputIndex[2] + inputSize[2];

            //Loop over images of the current stack
            for ( unsigned int i=inputIndex[2]; i < inputIndex[2] + inputSize[2]; i++ )
            {
                //std::cout << "process image # " << s << " slice #" << i << std::endl;



                ImageMaskType::RegionType wholeSliceRegion;
                wholeSliceRegion = rois[s];

                ImageMaskType::IndexType  wholeSliceRegionIndex = wholeSliceRegion.GetIndex();
                ImageMaskType::SizeType   wholeSliceRegionSize  = wholeSliceRegion.GetSize();

                wholeSliceRegionIndex[2]= i;
                wholeSliceRegionSize[2] = 1;

                wholeSliceRegion.SetIndex(wholeSliceRegionIndex);
                wholeSliceRegion.SetSize(wholeSliceRegionSize);

                //Extract slice
                ExtractImageMaskFilterType::Pointer sliceExtractor = ExtractImageMaskFilterType::New();
                sliceExtractor->SetExtractionRegion(wholeSliceRegion);
                sliceExtractor->SetInput(ones.GetPointer());
#if ITK_VERSION_MAJOR >= 4
                sliceExtractor->SetDirectionCollapseToIdentity(); // This is required.
#endif
                sliceExtractor->Update();

                ResamplerImageMaskFilterType::Pointer sliceResampler = ResamplerImageMaskFilterType::New();
                sliceResampler -> SetInput(sliceExtractor->GetOutput());
                sliceResampler -> SetTransform(invTransformsArray[s][i-inputIndex[2]]);
                sliceResampler -> SetInterpolator(nnInterpolator);
                sliceResampler -> SetOutputParametersFromImage(outImageMasks[s]);
                sliceResampler -> Update();

                StructuringElementType structuringElementDil;
                structuringElementDil.SetRadius( radiusDilation2D );
                structuringElementDil.CreateStructuringElement();

                MultiplyImageMaskFilterType::Pointer multiplyMaskFilter = MultiplyImageMaskFilterType::New();
                multiplyMaskFilter -> SetInput1(outImageMask.GetPointer());
                //multiplyMaskFilter -> SetInput1(dilateFilter->GetOutput());
                multiplyMaskFilter -> SetInput2(sliceResampler->GetOutput());
                multiplyMaskFilter -> Update();

                ResamplerImageMaskFilterType::Pointer sliceResamplerBack = ResamplerImageMaskFilterType::New();
                sliceResamplerBack -> SetInput(multiplyMaskFilter->GetOutput());
                //sliceResamplerBack -> SetInput(closingFilter->GetOutput());
                sliceResamplerBack -> SetTransform(transformsArray[s][i-inputIndex[2]]);
                sliceResamplerBack -> SetInterpolator(nnInterpolator);
                sliceResamplerBack -> SetOutputParametersFromImage(outLRImageMasks[s]);
                sliceResamplerBack -> Update();


                BinaryDilateImageFilterType::Pointer dilateFilter = BinaryDilateImageFilterType::New();
                dilateFilter->SetInput(sliceResamplerBack->GetOutput());
                dilateFilter->SetKernel(structuringElementDil);
                dilateFilter->SetForegroundValue(1.0);
                dilateFilter->Update();

                AddImageMaskFilter::Pointer addLRMaskFilter = AddImageMaskFilter::New();
                addLRMaskFilter -> SetInput1(outLRImageMasks[s]);
                addLRMaskFilter -> SetInput2(dilateFilter->GetOutput());
                addLRMaskFilter -> Update();

                outLRImageMasks[s] = addLRMaskFilter->GetOutput();
            }

            //std::stringstream ssFileLR;
            //ssFileLR << "/home/ch176971/Desktop/DbgMasks/lrRefMask_" << s << ".nii.gz";

            MaskWriterType::Pointer lrWriter = MaskWriterType::New();
            lrWriter -> SetInput(outLRImageMasks[s]);
            lrWriter -> SetFileName(outLRMask[s].c_str());
            lrWriter -> Update();

        }

        StructuringElementType structuringElementDilHR;
        structuringElementDilHR.SetRadius( radiusDilation3D );
        structuringElementDilHR.CreateStructuringElement();

        BinaryDilateImageFilterType::Pointer dilateFilterHRmask = BinaryDilateImageFilterType::New();
        dilateFilterHRmask -> SetInput( outImageMask.GetPointer() );
        dilateFilterHRmask -> SetKernel( structuringElementDilHR );
        dilateFilterHRmask -> SetForegroundValue(1.0);
        dilateFilterHRmask -> Update();

        //

        //std::cout << "h1" << std::endl;
        //end_time_unix = mialtk::getTime();;
        //std::cout << "h2" << std::endl;
        //diff_time_unix = end_time_unix - start_time_unix;

        //mialtk::printTime("TV (IGD)",diff_time_unix);


        // Write image
        //TODO Mask type!!!!!
        MaskWriterType::Pointer maskWriter =  MaskWriterType::New();
        maskWriter -> SetFileName( outMask );
        //writer -> SetInput( resampler -> GetOutput() );
        maskWriter -> SetInput( dilateFilterHRmask -> GetOutput() );

        if ( strcmp(outMask,"") != 0)
        {
            std::cout << "Writing " << outMask << " ... ";
            maskWriter->Update();
            std::cout << "done." << std::endl;
        }


    } catch (TCLAP::ArgException &e)  // catch any exceptions
    { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

    return EXIT_SUCCESS;
}
