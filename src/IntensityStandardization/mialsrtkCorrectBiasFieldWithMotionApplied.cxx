/*=========================================================================

Program: Bias Field Correction in the LR images given a global bias field and slice motion parameters
Language: C++
Date: $Date: 2016-28-06 15:00:00 +0100 (Tuesday, 28 June 2016) $
Version: $Revision: 1 $
Author: $Sebastien Tourbier$

==========================================================================*/

#include <tclap/CmdLine.h>
#include <iostream>
#include <limits>       // std::numeric_limits


#include "itkImage.h"
#include "itkImageMaskSpatialObject.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkExpImageFilter.h"
#include "itkDivideImageFilter.h"

#include "itkTransformFileReader.h"
#include "itkTransformFactory.h"
#include "mialsrtkSliceBySliceTransform.h"
#include "itkEuler3DTransform.h"
#include "itkVersorRigid3DTransform.h"

#include "vnl/vnl_matops.h"
#include "vnl/vnl_sparse_matrix.h"
#include "btkLinearInterpolateImageFunctionWithWeights.h"
#include "mialsrtkOrientedSpatialFunction.h"

#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "itkResampleImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"


const    unsigned int    Dimension3D = 3;
const    unsigned int    Dimension2D = 2;

using namespace itk;

int main(int argc, char *argv[])
{
    // Typesets
    typedef float InputPixelType;
    typedef float OutputPixelType;
    typedef unsigned char MaskPixelType;

    typedef itk::Image<InputPixelType, Dimension3D> ImageType;
    typedef itk::Image<OutputPixelType, Dimension3D> OutputImageType;
    typedef itk::Image<MaskPixelType, Dimension3D> ImageMaskType;

    typedef ImageType::RegionType               RegionType;
    typedef ImageType::SizeType    SizeType;
    typedef ImageType::IndexType   IndexType;
    typedef ImageType::SpacingType SpacingType;
    typedef ImageType::PointType   PointType;

    typedef itk::Image<InputPixelType, Dimension2D> SliceImageType;
    typedef itk::Image<MaskPixelType, Dimension2D> SliceMaskType;

    typedef itk::ImageFileReader<ImageType> ReaderType;
    typedef itk::ImageFileWriter<OutputImageType> WriterType;
    typedef itk::ImageFileReader<ImageMaskType> MaskReaderType;

    typedef itk::ImageMaskSpatialObject< Dimension3D >  MaskType;
    typedef itk::ImageMaskSpatialObject< Dimension2D >  Mask2DType;
    typedef MaskType::Pointer  MaskPointer;

    typedef mialsrtk::SliceBySliceTransformBase< double, Dimension3D > TransformBaseType;
    typedef mialsrtk::SliceBySliceTransform< double, Dimension3D > TransformType;
    typedef TransformType::Pointer                          TransformPointer;

    typedef itk::VersorRigid3DTransform<double> VersorTransformType;
    typedef itk::Euler3DTransform<double> Euler3DTransformType;

    // Register the SliceBySlice transform (a non-default ITK transform) with the TransformFactory of ITK
    TransformFactory<TransformType>::RegisterTransform();

    typedef itk::TransformFileReader     TransformReaderType;
    typedef TransformReaderType::TransformListType * TransformListType;

    typedef btk::LinearInterpolateImageFunctionWithWeights<ImageType, double> InterpolatorType;
    typedef InterpolatorType::Pointer InterpolatorPointer;

    typedef ContinuousIndex<double, Dimension3D> ContinuousIndexType;

    /**Oriented spatial function typedef. */
    typedef mialsrtk::OrientedSpatialFunction<double, 3, PointType> FunctionType;

    /**Const iterator typedef. */
    typedef ImageRegionConstIteratorWithIndex< ImageType >  ConstIteratorType;
    typedef ImageRegionIteratorWithIndex< ImageType >  IteratorType;

    typedef vnl_vector<float> VnlVectorType;
    typedef vnl_sparse_matrix<float> VnlSparseMatrixType;

    typedef vnl_vector<float>::iterator floatIter;
    typedef vnl_vector<int>::iterator intIter;

    typedef itk::ResampleImageFilter<ImageMaskType, ImageMaskType> ResampleImageMaskFilterType;
    typedef itk::NearestNeighborInterpolateImageFunction<ImageMaskType> NNInterpolatorType;


    const char *inputFileName = NULL;
    const char *maskFileName = NULL;
    const char *transformFileName = NULL;
    const char *inBiasFieldFileName = NULL;

    const char *outImageFileName = NULL;
    const char *outBiasFieldFileName = NULL;

    // Parse arguments

    TCLAP::CmdLine cmd("Register slice of a LR images to a template HR image", ' ', "Unversioned");

    TCLAP::ValueArg<std::string> inputArg("i","input","Input image file",true,"","string",cmd);
    TCLAP::ValueArg<std::string> maskArg("m","","Input mask file",true,"","string",cmd);

    TCLAP::ValueArg<std::string> transformArg("t","transform","Transform input file",true,"","string",cmd);

    TCLAP::ValueArg<std::string> inputBFArg("","input-bias-field","Input bias field image file (Typically the bias field globally estimated in the HR reconstructed image)",true,"","string",cmd);

    TCLAP::ValueArg<std::string> outputArg("o","output","Output bias field corrected image file",true,"","string",cmd);

    TCLAP::ValueArg<std::string> outputBFArg("","output-bias-field","Output bias field image file",true,"","string",cmd);

    // Parse the argv array.
    cmd.parse( argc, argv );

    inputFileName = inputArg.getValue().c_str();
    maskFileName = maskArg.getValue().c_str();
    transformFileName = transformArg.getValue().c_str();

    inBiasFieldFileName = inputBFArg.getValue().c_str();

    outImageFileName = outputArg.getValue().c_str();
    outBiasFieldFileName = outputBFArg.getValue().c_str();

    // Load input image and corresponding mask

    ReaderType::Pointer readerIm = ReaderType::New();
    readerIm -> SetFileName(inputFileName);
    readerIm -> Update();

    ImageType::Pointer lrIm = readerIm -> GetOutput();

    ReaderType::Pointer readerBiasFieldIm = ReaderType::New();
    readerBiasFieldIm -> SetFileName(inBiasFieldFileName);
    readerBiasFieldIm -> Update();

    ImageType::Pointer biasFieldIm = readerBiasFieldIm -> GetOutput();

    MaskReaderType::Pointer readerMask = MaskReaderType::New();
    readerMask->SetFileName(maskFileName);
    readerMask->Update();

    MaskType::Pointer mask = MaskType::New();
    RegionType roi;
    mask -> SetImage( readerMask->GetOutput() );
    roi = mask -> GetAxisAlignedBoundingBoxRegion();


    // Load the transform parameter file

    std::cout<<"Reading transform:"<< transformFileName <<std::endl;
    TransformReaderType::Pointer transformReader = TransformReaderType::New();
    transformReader -> SetFileName( transformFileName );
    transformReader -> Update();

    TransformListType transformsList = transformReader->GetTransformList();

    TransformReaderType::TransformListType::const_iterator titr = transformsList->begin();
    TransformPointer trans = static_cast< TransformType * >( titr->GetPointer() );

    int numberOfSlices = trans -> GetNumberOfSlices();

    std::vector<VersorTransformType::Pointer> transforms(numberOfSlices);
    for(unsigned int j=0; j< trans -> GetNumberOfSlices(); j++)
    {
        transforms[j] = trans -> GetSliceTransform(j);
    }


    //for(unsigned int j=0; j< trans -> GetNumberOfSlices(); j++)
    //    resampler -> SetTransform(i, j, trans -> GetSliceTransform(j) ) ;

    //    //Make the assumption the direction of acquisition is along the last dimension
    //    unsigned int k1 =  roi.GetIndex()[2];
    //    unsigned int k2 =  k1 + roi.GetSize()[2];

    //    //Loop over the slices
    //    unsigned int i = k1;
    //    //#pragma omp parallel for private(i) shared(outImageMask) schedule(dynamic)
    //    for ( i = k1; i < k2; i++ )
    //    {
    //        std::cout << "###########################################################################################" << std::endl;
    //        std::cout << std::endl;
    //        std::cout << "Processing slice #" << i << " / last slice #" << k2 << std::endl;

    //        //Register the 3D template to the 2D slice
    //        RegionType sliceRegion;
    //        ImageType::IndexType  imageRegionIndex;
    //        ImageType::SizeType   imageRegionSize;

    //        imageRegionIndex = image->GetLargestPossibleRegion().GetIndex();
    //        imageRegionIndex[2] = i;

    //        imageRegionSize = image->GetLargestPossibleRegion().GetSize();
    //        imageRegionSize[2] = 1;

    //        sliceRegion.SetIndex(imageRegionIndex);
    //        sliceRegion.SetSize(imageRegionSize);

    //    }//end of slice loops

    RegionType HR_ImageRegion = biasFieldIm -> GetLargestPossibleRegion();
    IndexType start_hr  = HR_ImageRegion.GetIndex();
    SizeType  size_hr   = HR_ImageRegion.GetSize();

    IndexType end_hr;
    end_hr[0] = start_hr[0] + size_hr[0] - 1 ;
    end_hr[1] = start_hr[1] + size_hr[1] - 1 ;
    end_hr[2] = start_hr[2] + size_hr[2] - 1 ;

    // Differential continuous indexes to perform the neighborhood iteration
    SpacingType spacing_lr = lrIm -> GetSpacing();
    SpacingType spacing_hr = biasFieldIm -> GetSpacing();

    //spacing_lr[2] is assumed to be the lowest resolution
    //compute the index of the PSF in the LR image resolution
    std::vector<ContinuousIndexType> deltaIndexes;

    double ratioLRHRX = spacing_lr[0] / spacing_hr[0];
    double ratioLRHRY = spacing_lr[1] / spacing_hr[1];

    double ratioHRLRX = spacing_hr[0] / spacing_lr[0];
    double ratioHRLRY = spacing_hr[1] / spacing_lr[1];

    double ratioLRHRZ = spacing_lr[2] / spacing_hr[2];
    double ratioHRLRZ = spacing_hr[2] / spacing_lr[2];

    bool ratioXisEven = true;
    bool ratioYisEven = true;
    bool ratioZisEven = true;

    if((((int)round(ratioLRHRX)) % 2)) ratioXisEven = false;
    if((int)round(ratioLRHRY) % 2) ratioYisEven = false;
    if((int)round(ratioLRHRZ) % 2) ratioZisEven = false;

    std::cout << "ratioXisEven : " << ratioXisEven << std::endl;
    std::cout << "ratioYisEven : " << ratioYisEven << std::endl;
    std::cout << "ratioZisEven : " << ratioZisEven << std::endl;

    float factorPSF=1.5;
    int npointsX = 0;
    float initpointX = 0.0;
    if(ratioXisEven)
    {
        int k = floor(0.5 * ((factorPSF-ratioHRLRX)/ratioHRLRX));
        npointsX = 2 * (k+1);
        std::cout << "npointx 1: " << npointsX << std::endl;
        initpointX = - (float)(0.5+k) * ratioHRLRX;
    }
    else
    {
        int k = floor(factorPSF*0.5 /ratioHRLRX);
        npointsX = 2*k + 1;
        std::cout << "npointx 2: " << npointsX << std::endl;
        initpointX = - (float)(k) * ratioHRLRX;
    }

    int npointsY = 0;
    float initpointY = 0.0;
    if(ratioYisEven)
    {
        int k = floor(0.5 * ((factorPSF-ratioHRLRY)/ratioHRLRY));
        npointsY = 2 * (k+1);
        std::cout << "npointy 1: " << npointsY << std::endl;
        initpointY = - (float)(0.5+k) * ratioHRLRY;
    }
    else
    {
        int k = floor(factorPSF*0.5 /ratioHRLRY);
        npointsY = 2*k + 1;
        std::cout << "npointy 2: " << npointsY << std::endl;
        initpointY = - (float)(k) * ratioHRLRY;
    }

    int npointsZ = 0;
    float initpointZ = 0.0;
    if(ratioZisEven)
    {
        int k = floor(0.5 * ((factorPSF-ratioHRLRZ)/ratioHRLRZ));
        npointsZ = 2 * (k+1);
        std::cout << "npointz 1: " << npointsZ << std::endl;
        initpointZ = - (float)(0.5+k) * ratioHRLRZ;
    }
    else
    {
        int k = floor(factorPSF*0.5 /ratioHRLRZ);
        npointsZ = 2*k + 1;
        std::cout << "npointz 2: " << npointsZ << std::endl;
        initpointZ = - (float)(k) * ratioHRLRZ;
    }

    std::cout << "Spacing LR X: " << spacing_lr[0] << " / Spacing HR X: " << spacing_hr[0]<< std::endl;
    std::cout << "Spacing LR Y: " << spacing_lr[1] << " / Spacing HR Y: " << spacing_hr[1]<< std::endl;
    std::cout << "Spacing LR Z: " << spacing_lr[2] << " / Spacing HR Z: " << spacing_hr[2]<< std::endl;

    std::cout << " , 1/2 * LR/HR X:" << 0.5 * ratioLRHRX << " , NPointsX : " << npointsX << std::endl;
    std::cout << " , 1/2 * LR/HR Y:" << 0.5 * ratioLRHRY << " , NPointsY : " << npointsY << std::endl;
    std::cout << " , 1/2 * LR/HR Z:" << 0.5 * ratioLRHRZ << " , NPointsZ : " << npointsZ << std::endl;


    ContinuousIndexType delta;
    delta[0] = 0;
    delta[1] = 0;
    delta[2] = 0;

    for(int i = 0; i < npointsX; i++)
    {
        for(int j = 0; j < npointsY; j++)
        {
            for(int k = 0; k < npointsZ; k++)
            {
                delta[0] = initpointX + (float)i * ratioHRLRX;
                delta[1] = initpointY + (float)j * ratioHRLRY;
                delta[2] = initpointZ + (float)k * ratioHRLRZ;

                deltaIndexes.push_back(delta);
                std::cout << " delta : " << delta[0] << " , " << delta[1] << " , " << delta[2] << std::endl;


            }
        }
    }

    // Set size of matrices
    unsigned int ncols = HR_ImageRegion.GetNumberOfPixels();

    unsigned int nrows = roi.GetNumberOfPixels();


    std::cout << "Initialize H and Z ..." << std::endl;
    vnl_sparse_matrix<float> H;
    H.set_size(nrows, ncols);

    vnl_vector<float> Y;
    Y.set_size(nrows);
    Y.fill(0.0);

    vnl_vector<float> X;
    X.set_size(ncols);
    X.fill(0.0);

    std::cout << "Size of H :  #rows = " << H.rows() << ", #cols = "<<H.cols() << std::endl;

    // Interpolator for HR image
    InterpolatorPointer interpolator = InterpolatorType::New();
    interpolator -> SetInputImage( biasFieldIm );

    SpacingType inputSpacing2 = lrIm -> GetSpacing();
    inputSpacing2[0] = inputSpacing2[0] ;
    inputSpacing2[1] = inputSpacing2[1] ;
    inputSpacing2[2] = inputSpacing2[2];

    std::cout << "input spacing 2 : " << inputSpacing2 << std::endl;


    // PSF definition
    FunctionType::Pointer function = FunctionType::New();
    function -> SetPSF(  FunctionType::GAUSSIAN );
    function -> SetDirection( lrIm -> GetDirection() );

    std::cout << "Image sizes : " << lrIm->GetLargestPossibleRegion().GetSize() << std::endl;
    std::cout << "Image direction : " << lrIm->GetDirection() << std::endl;
    std::cout << "Image spacing : " << lrIm->GetSpacing() << std::endl;

    function -> SetSpacing( inputSpacing2 );

    //function -> SetSpacing( lrIm -> GetSpacing() );
    //function -> Print(std::cout);

    //Define the ROI of the current LR image
    IndexType inputIndex = roi.GetIndex();
    SizeType  inputSize  = roi.GetSize();

    //Define all indexes needed for iteration over the slices
    IndexType lrIndex;              //index of a point in the LR image im
    IndexType lrDiffIndex;          //index of this point in the current ROI of the LR image im
    unsigned int lrLinearIndex;     //index lineaire de ce point dans le vecteur

    IndexType hrIndex;
    IndexType hrDiffIndex;
    ContinuousIndexType hrContIndex;
    unsigned int hrLinearIndex;

    ContinuousIndexType nbIndex;

    PointType lrPoint;            //PSF center in world coordinates (PointType = worldcoordinate for ITK)
    PointType nbPoint;            //PSF point in world coordinate
    PointType transformedPoint;   //after applying current transform (registration)

    // Iteration over slices
    for ( unsigned int i=inputIndex[2]; i < inputIndex[2] + inputSize[2]; i++ )
    {

        //TODO: outlier rejection scheme, if slice was excluded, we process directly the next one
        // It would probably require to save a list of outlier slices during motion correction and
        // to load it here as input and create a global vector.

        std::cout << "Process slice # " << i << std::endl;

        RegionType wholeSliceRegion;
        wholeSliceRegion = roi;

        IndexType  wholeSliceRegionIndex = wholeSliceRegion.GetIndex();
        SizeType   wholeSliceRegionSize  = wholeSliceRegion.GetSize();

        wholeSliceRegionIndex[2]= i;
        wholeSliceRegionSize[2] = 1;

        wholeSliceRegion.SetIndex(wholeSliceRegionIndex);
        wholeSliceRegion.SetSize(wholeSliceRegionSize);

        ConstIteratorType fixedIt( lrIm, wholeSliceRegion);

        double lrValue;
        double hrValue;

        for(fixedIt.GoToBegin(); !fixedIt.IsAtEnd(); ++fixedIt)
        {
            //Current index in the LR image
            lrIndex = fixedIt.GetIndex();

            //lrIndex[0] = lrIndex[0] + deltaIndexes[1][0];
            //lrIndex[1] = lrIndex[1] + deltaIndexes[1][1];
            //lrIndex[2] = lrIndex[2] + deltaIndexes[1][2];

            //World coordinates of lrIndex using the image header
            lrIm -> TransformIndexToPhysicalPoint( lrIndex, lrPoint );

            /*
                if ( m_Masks.size() > 0)
                    if ( ! m_Masks[im] -> IsInside(lrPoint) )
                        continue;
                */

            //Compute the coordinates in the SR using the estimated registration
            transformedPoint = transforms[i]->TransformPoint( lrPoint );
            //transformedPoint = lrPoint;

            //check if this point is in the SR image (biasFieldIm)
            /*
                if ( ! interpolator -> IsInsideBuffer( transformedPoint ) )
                    continue;
                */

            //From the LR image coordinates to the LR ROI coordinates
            lrDiffIndex[0] = lrIndex[0] - inputIndex[0];
            lrDiffIndex[1] = lrIndex[1] - inputIndex[1];
            lrDiffIndex[2] = lrIndex[2] - inputIndex[2];

            //Compute the corresponding linear index of lrDiffIndex
            if(1)
            {
                lrLinearIndex = lrDiffIndex[0] + lrDiffIndex[1]*inputSize[0] +
                        lrDiffIndex[2]*inputSize[0]*inputSize[1];
            }
            else
            {
                lrLinearIndex = lrDiffIndex[0] + lrDiffIndex[1]*inputSize[0] +
                        lrDiffIndex[2]*inputSize[0]*inputSize[1];
            }
            //Get the intensity value in the LR image
            //Y[lrLinearIndex + offset] = fixedIt.Get();

            if ( mask -> IsInside(lrPoint) )
            {
                //std::cout << "add point yk..." << std::endl;
                Y[lrLinearIndex] = fixedIt.Get();
            }


            //Set the center point of the PSF
            function -> SetCenter( lrPoint );

            //function -> Print(std::cout);

            //lrIndex[0] = lrIndex[0] - deltaIndexes[1][0];
            //lrIndex[1] = lrIndex[1] - deltaIndexes[1][1];
            //lrIndex[2] = lrIndex[2] - deltaIndexes[1][2];


            //std::cout << "Populates H ..." << std::endl;
            //Loop over points of the PSF

            //std::cout << "Loop over PSF points : " << deltaIndexes.size() << "points" << std::endl;
            for(unsigned int k=0; k<deltaIndexes.size(); k++)
            {
                //Coordinates in the LR image
                nbIndex[0] = deltaIndexes[k][0] + lrIndex[0];
                nbIndex[1] = deltaIndexes[k][1] + lrIndex[1];
                nbIndex[2] = deltaIndexes[k][2] + lrIndex[2];

                //World coordinates using LR image header
                lrIm -> TransformContinuousIndexToPhysicalPoint( nbIndex, nbPoint );

                //Compute the PSF value at this point
                lrValue = function -> Evaluate(nbPoint);

                if ( lrValue > 0)
                {
                    //Compute the world coordinate of this point in the SR image
                    // transformedPoint = nbPoint;
                    transformedPoint = transforms[i]->TransformPoint( nbPoint );


                    //Set this coordinate in continuous index in SR image space
                    biasFieldIm -> TransformPhysicalPointToContinuousIndex(
                                transformedPoint, hrContIndex );

                    // OLD VERSION (BTK V1)

                    bool isInsideHR = true;

                    // FIXME This checking should be done for all points first, and
                    // discard the point if al least one point is out of the reference
                    // image

                    if ( (hrContIndex[0] < start_hr[0]) || (hrContIndex[0] > end_hr[0]) ||
                         (hrContIndex[1] < start_hr[1]) || (hrContIndex[1] > end_hr[1]) ||
                         (hrContIndex[2] < start_hr[2]) || (hrContIndex[2] > end_hr[2]) )
                        isInsideHR = false;

                    if ( isInsideHR )
                    {
                        //Compute the corresponding value in the SR image -> useless
                        //Allows to compute the set of contributing neighbors
                        hrValue = interpolator -> Evaluate( transformedPoint );

                        //std::cout << "Number of contributing neighbors for point " << transformedPoint << " : " << interpolator -> GetContributingNeighbors() << std::endl;

                        //Loop over points affected using the interpolation
                        for(unsigned int n=0; n<interpolator -> GetContributingNeighbors();
                            n++)
                        {
                            //Index in the SR image
                            hrIndex = interpolator -> GetIndex(n);

                            //Index in the ROI of the SR index
                            hrDiffIndex[0] = hrIndex[0] - start_hr[0];
                            hrDiffIndex[1] = hrIndex[1] - start_hr[1];
                            hrDiffIndex[2] = hrIndex[2] - start_hr[2];

                            //Compute the corresponding linear index
                            hrLinearIndex = hrDiffIndex[0] + hrDiffIndex[1]*size_hr[0] +
                                    hrDiffIndex[2]*size_hr[0]*size_hr[1];

                            //Add the correct value in H !
                            H(lrLinearIndex, hrLinearIndex) += interpolator -> GetOverlap(n)* lrValue;
                            X[hrLinearIndex] = biasFieldIm->GetPixel(hrDiffIndex);
                            //H(lrLinearIndex + offset, hrLinearIndex) += interpolator -> GetOverlap(n)* 1.0;
                            //H(lrLinearIndex + offset, hrLinearIndex) +=  lrValue;
                            //H(lrLinearIndex + offset, hrLinearIndex) +=  1.0;


                        }

                    }



                    /*
                            //std::cout << "Bspline flag#1" << std::endl;
                            itkBSplineFunction::Pointer bsplineFunction = itkBSplineFunction::New();
                            itkBSplineFunction::WeightsType bsplineWeights;
                            bsplineWeights.SetSize(8); // (bsplineOrder + 1)^3
                            itkBSplineFunction::IndexType   bsplineStartIndex;
                            itkBSplineFunction::IndexType   bsplineEndIndex;
                            itkBSplineFunction::SizeType    bsplineSize;
                            RegionType                      bsplineRegion;

                            //Get the interpolation weight using itkBSplineInterpolationWeightFunction
                            bsplineFunction->Evaluate(hrContIndex,bsplineWeights,bsplineStartIndex);

                            //Get the support size for interpolation
                            bsplineSize = bsplineFunction->GetSupportSize();

                            //Check if the bspline support region is inside the HR image
                            bsplineEndIndex[0] = bsplineStartIndex[0] + bsplineSize[0];
                            bsplineEndIndex[1] = bsplineStartIndex[1] + bsplineSize[1];
                            bsplineEndIndex[2] = bsplineStartIndex[2] + bsplineSize[2];

                            //std::cout << "bsplineStart" << bsplineStartIndex[0] << "," << bsplineStartIndex[1] << "," << bsplineStartIndex[2] << std::endl;
                            //std::cout <<"bsplineEnd" << bsplineEndIndex[0] << "," << bsplineEndIndex[1] << "," << bsplineEndIndex[2] << std::endl;
                            //std::cout <<"bsplineSize" << bsplineSize[0] << "," << bsplineSize[1] << "," << bsplineSize[2] << std::endl;

                            //std::cout << "Bspline flag#2" << std::endl;

                            if(biasFieldIm->GetLargestPossibleRegion().IsInside(bsplineStartIndex)
                                    && biasFieldIm->GetLargestPossibleRegion().IsInside(bsplineEndIndex))
                            {
                                //Set the support region
                                bsplineRegion.SetSize(bsplineSize);
                                bsplineRegion.SetIndex(bsplineStartIndex);

                                //std::cout << "Bspline flag#3" << std::endl;

                                //Instantiate an iterator on HR image over the bspline region
                                ImageRegionConstIteratorWithIndex< ImageType > itHRImage(biasFieldIm,bsplineRegion);

                                //linear index of bspline weights
                                unsigned int weightLinearIndex = 0;

                                //Loop over the support region
                                for(itHRImage.GoToBegin(); !itHRImage.IsAtEnd(); ++itHRImage)
                                {

                                    //Get coordinate in HR image
                                    IndexType hrIndex = itHRImage.GetIndex();
                                    //Compute the corresponding linear index
                                    if(im==0)
                                    {
                                        hrLinearIndex = hrIndex[0] + hrIndex[1]*size_hr[0] + hrIndex[2]*size_hr[0]*size_hr[1];
                                    }
                                    else
                                    {
                                        hrLinearIndex = hrIndex[0] + hrIndex[1]*size_hr[0] + hrIndex[2]*size_hr[0]*size_hr[1];
                                        //hrLinearIndex = hrIndex[0] + hrIndex[2]*size_hr[0] + hrIndex[1]*size_hr[0]*size_hr[2];//working with coro
                                        //hrLinearIndex = hrIndex[2] + hrIndex[0]*size_hr[2] + hrIndex[1]*size_hr[2]*size_hr[0];
                                        //hrLinearIndex = hrIndex[1] + hrIndex[2]*size_hr[1] + hrIndex[0]*size_hr[2]*size_hr[1];
                                        //hrLinearIndex = hrIndex[1] + hrIndex[0]*size_hr[1] + hrIndex[2]*size_hr[1]*size_hr[0];
                                        //hrLinearIndex = hrIndex[2] + hrIndex[1]*size_hr[2] + hrIndex[0]*size_hr[2]*size_hr[1];
                                        //hrLinearIndex = hrIndex[2] + hrIndex[0]*size_hr[2] + hrIndex[1]*size_hr[2]*size_hr[0];
                                    }

                                    //Add weight*PSFValue to the corresponding element in H
                                    H(lrLinearIndex, hrLinearIndex)  +=1.0 * bsplineWeights[weightLinearIndex];//BOXCAR profile
                                    // H(lrLinearIndex, hrLinearIndex)  +=lrValue * bsplineWeights[weightLinearIndex];//Gaussian profile
                                    weightLinearIndex++;

                                } //end of loop over the support region
                                //std::cout << "Bspline flag#4" << std::endl;

                            }// end if bspline index inside sr image
                            //
                            */

                } // if psf point is inside sr image

            }//End of loop over PSF points

        }// Loop over all pixels of the slice

    }//Loop over all slices

    std::cout << "H was computed and normalized ..." << std::endl << std::endl;
    // Normalize H

    for (unsigned int i = 0; i < H.rows(); i++)
    {
        double sum = H.sum_row(i);

        VnlSparseMatrixType::row & r = H.get_row(i);
        VnlSparseMatrixType::row::iterator col_iter = r.begin();

        for ( ;col_iter != r.end(); ++col_iter)
            (*col_iter).second = (*col_iter).second / sum;
    }

    vnl_vector<float> Ybf;
    H.mult(X,Ybf);

    ImageType::Pointer lrBiasFieldIm = ImageType::New();
    lrBiasFieldIm->SetOrigin(lrIm->GetOrigin());
    lrBiasFieldIm->SetDirection(lrIm->GetDirection());
    lrBiasFieldIm->SetSpacing(lrIm->GetSpacing());
    lrBiasFieldIm->SetRegions(lrIm->GetLargestPossibleRegion());

    lrBiasFieldIm->Allocate();

    lrBiasFieldIm->FillBuffer(0.0);
    lrBiasFieldIm->Update();

    ImageType::Pointer lrROIIm = ImageType::New();
    lrROIIm->SetOrigin(lrIm->GetOrigin());
    lrROIIm->SetDirection(lrIm->GetDirection());
    lrROIIm->SetSpacing(lrIm->GetSpacing());
    lrROIIm->SetRegions(lrIm->GetLargestPossibleRegion());

    lrROIIm->Allocate();

    lrROIIm->FillBuffer(0.0);
    lrROIIm->Update();


    IteratorType itLrBiasFieldIm(lrBiasFieldIm,roi);
    IteratorType itLrROIIm(lrROIIm,roi);

    for(itLrBiasFieldIm.GoToBegin(), itLrROIIm.GoToBegin();!itLrBiasFieldIm.IsAtEnd();++itLrBiasFieldIm,++itLrROIIm)
    {
        lrIndex = itLrBiasFieldIm.GetIndex();
        lrIm -> TransformIndexToPhysicalPoint( lrIndex, lrPoint );

        //From the LR image coordinates to the LR ROI coordinates
        lrDiffIndex[0] = lrIndex[0] - inputIndex[0];
        lrDiffIndex[1] = lrIndex[1] - inputIndex[1];
        lrDiffIndex[2] = lrIndex[2] - inputIndex[2];

        //Compute the corresponding linear index of lrDiffIndex
        lrLinearIndex = lrDiffIndex[0] + lrDiffIndex[1]*inputSize[0] + lrDiffIndex[2]*inputSize[0]*inputSize[1];

        if(mask -> IsInside(lrPoint) )
        {
            itLrBiasFieldIm.Set(Ybf[lrLinearIndex]);
        }
        itLrROIIm.Set(Y[lrLinearIndex]);

    }

    lrBiasFieldIm->Print(std::cout);


    std::cerr << "Compute the image corrected... " << std::endl;
    typedef itk::ExpImageFilter<ImageType, ImageType> ExpFilterType;
    ExpFilterType::Pointer expFilter = ExpFilterType::New();
    expFilter->SetInput( lrBiasFieldIm.GetPointer() );
    expFilter->Update();


    expFilter->GetOutput()->Print(std::cout);
    lrIm->Print(std::cout);



    typedef itk::DivideImageFilter<ImageType, ImageType, ImageType> DividerType;
    DividerType::Pointer divider = DividerType::New();
    divider->SetInput1( lrROIIm.GetPointer() );
    divider->SetInput2( expFilter->GetOutput() );

    divider->Print(std::cout);

    divider->Update();


    //

    std::cerr << "Write ouput images ... " << std::endl;

    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(outImageFileName);
    writer->SetInput(divider->GetOutput());
    writer->Update();

    WriterType::Pointer fieldWriter = WriterType::New();
    fieldWriter->SetFileName(outBiasFieldFileName);
    fieldWriter->SetInput(lrBiasFieldIm.GetPointer());
    fieldWriter->Update();

    std::cerr << "Done! " << std::endl;

}
