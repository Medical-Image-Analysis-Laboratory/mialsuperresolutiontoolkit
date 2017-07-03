/* ============================================================================================

  Program :   High resolution Fetal and Pediatric MRI reconstruction
  Module  :   Compute image contrast measures (variance measure, L2 norm of gradient, Laplacian)
  File    :   mialtkEvaluateContrst.cxx
  Language:   C++
  Version :   $Revision: 1.0 $

  Copyright (c) 2017 Medical Image Analysis Laboratory (MIAL), Lausanne

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

  Latest modifications:  April 26, 2009

=============================================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
//#include "itkImageRegionIteratorWithIndex.h"

#include "itkImageDuplicator.h"

#include "itkRegionOfInterestImageFilter.h"

#include "itkConstantPadImageFilter.h"

#include "itkGradientMagnitudeImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"

vnl_matrix<double> hadamard( int size );
float compute_alpha(itk::Image<float,3>::Pointer Im);

int main( int argc, char * argv[] )
{
    if( argc < 2 )
    {
        std::cerr << "Usage: " << std::endl;
        std::cerr << argv[0] << "  inputImageFile " << std::endl;
        return EXIT_FAILURE;
    }

    const unsigned int  Dimension = 3;

    typedef    float    InputPixelType;
    typedef    float    ImagePixelType;

    typedef itk::Image< InputPixelType,  Dimension >   InputImageType;
    typedef itk::Image< ImagePixelType,  Dimension >   ImageType;

    typedef itk::ImageFileReader< InputImageType >  ReaderType;

    typedef itk::CastImageFilter< InputImageType, ImageType  > InputCastFilterType;

    InputCastFilterType::Pointer  icastfilter = InputCastFilterType::New();

    //std::cout << "Input image : " << argv[1] << std::endl;

    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( argv[1] );

    icastfilter->SetInput( reader->GetOutput() );
    icastfilter->Update();

    ImageType::Pointer inputImage = icastfilter->GetOutput();
    ImageType::RegionType inputRegion = inputImage->GetLargestPossibleRegion();
    ImageType::SizeType inputSize = inputRegion.GetSize();
    ImageType::SizeType brainSize = inputRegion.GetSize();

    typedef itk::ImageRegionIterator< ImageType > ImageIterator;
    ImageIterator iti( inputImage, inputRegion );

    //Extract the max pixel value
    float maxValue = 0.0f;
    for (iti.GoToBegin(); !iti.IsAtEnd(); ++iti)
    {
        if(iti.Get()>=maxValue) maxValue = iti.Get();
    }
    std::cout << "Maximum voxel value : " << maxValue << std::endl;


    /*
    ImageType::IndexType indexOrigCenter;
    indexOrigCenter[0] = floor(0.5 * inputSize[0]);
    indexOrigCenter[1] = floor(0.5 * inputSize[1]);
    indexOrigCenter[2] = floor(0.5 * inputSize[2]);
    */

    //Pad the image to have a size of 128 a power of 2;
    int size = 256;
    ImageType::SizeType fixedSize;
    fixedSize[0] = size;
    fixedSize[1] = size;
    fixedSize[2] = size;

    //ImageType::RegionType brainRegion;
    //brainRegion.SetIndex(indexOrigCenter);
    //brainRegion.SetSize(fixedSize);

    ImageType::SizeType padBoundSize;
    padBoundSize[0] = fixedSize[0] - inputSize[0];
    padBoundSize[1] = fixedSize[1] - inputSize[1];
    padBoundSize[2] = fixedSize[2] - inputSize[2];

    typedef itk::ConstantPadImageFilter <ImageType, ImageType> ConstantPadImageFilterType;

    ImageType::SizeType lowerExtendRegion;
    lowerExtendRegion[0] = ceil(0.5 * padBoundSize[0]);
    lowerExtendRegion[1] = ceil(0.5 * padBoundSize[1]);
    lowerExtendRegion[2] = ceil(0.5 * padBoundSize[2]);

    ImageType::SizeType upperExtendRegion;
    upperExtendRegion[0] = padBoundSize[0] - lowerExtendRegion[0];
    upperExtendRegion[1] = padBoundSize[1] - lowerExtendRegion[1];
    upperExtendRegion[2] = padBoundSize[2] - lowerExtendRegion[2];

    ImageType::IndexType brainIndex = inputRegion.GetIndex();
    brainIndex[0] = lowerExtendRegion[0];
    brainIndex[1] = lowerExtendRegion[1];
    brainIndex[2] = lowerExtendRegion[2];


    //std::cout << "Brain region : " << brainRegion << std::endl;

    ImageType::PixelType constantPixel = 0.0;

    ConstantPadImageFilterType::Pointer padFilter = ConstantPadImageFilterType::New();
    padFilter->SetInput(inputImage);
    //padFilter->SetPadBound(outputRegion); // Calls SetPadLowerBound(region) and SetPadUpperBound(region)
    padFilter->SetPadLowerBound(lowerExtendRegion);
    padFilter->SetPadUpperBound(upperExtendRegion);
    padFilter->SetConstant(constantPixel);
    padFilter->Update();

    inputImage = padFilter->GetOutput();
    inputRegion = inputImage->GetLargestPossibleRegion();
    inputSize = inputRegion.GetSize();

    ImageType::IndexType inputIndex = inputRegion.GetIndex();

    //std::cout << "input region : " << inputRegion << std::endl;
    //std::cout << "input region : " << inputRegion << std::endl;

    typedef itk::RegionOfInterestImageFilter< ImageType, ImageType > ROIFilterType;
    ROIFilterType::Pointer roiFilter = ROIFilterType::New();
    roiFilter->SetInput(inputImage);
    roiFilter->SetRegionOfInterest(inputRegion);
    roiFilter->Update();

    inputImage = roiFilter->GetOutput();
    inputRegion = inputImage->GetLargestPossibleRegion();
    inputSize = inputRegion.GetSize();
    inputIndex = inputRegion.GetIndex();

    //std::cout << "input region : " << inputRegion << std::endl;


    ImageType::IndexType brainIndexCrop;
    /*
    brainIndexCenter[0] = floor(0.5 * inputSize[0]) - 0.5 * floor(0.5 * inputSize[0]);
    brainIndexCenter[1] = floor(0.5 * inputSize[1]) - 0.5 * floor(0.5 * inputSize[0]);
    brainIndexCenter[2] = floor(0.5 * inputSize[2]) - 0.5 * floor(0.5 * inputSize[0]);

    brainIndexCrop[0] = inputIndex[0] + 0.5 * floor(0.5 * inputSize[0]);
    brainIndexCrop[1] = inputIndex[1] + 0.5 * floor(0.5 * inputSize[1]);
    brainIndexCrop[2] = inputIndex[2] + 0.5 * floor(0.5 * inputSize[2]);
    */

    brainIndexCrop[0] = 0;
    brainIndexCrop[1] = 0;
    brainIndexCrop[2] = 0;

    ImageType::IndexType brainIndex2;
    brainIndex2[0] = inputIndex[0] + 0.5 * floor(0.5 * inputSize[0]);
    brainIndex2[1] = inputIndex[1] + 0.5 * floor(0.5 * inputSize[1]);
    brainIndex2[2] = inputIndex[2] + 0.5 * floor(0.5 * inputSize[2]);

    size=0.5*size;
    fixedSize[0] = size;
    fixedSize[1] = size;
    fixedSize[2] = size;

    ImageType::RegionType brainRegionCrop;
    brainRegionCrop.SetIndex(brainIndexCrop);
    brainRegionCrop.SetSize(fixedSize);

    ImageType::RegionType brainRegion2;
    brainRegion2.SetIndex(brainIndex2);
    brainRegion2.SetSize(fixedSize);


    //std::cout << "brain region crop: " << brainRegionCrop << std::endl;
    //std::cout << "brain region 2: " << brainRegion2 << std::endl;

    ImageType::Pointer inputImage2 = ImageType::New();
    inputImage2->SetOrigin(inputImage->GetOrigin());
    inputImage2->SetDirection(inputImage->GetDirection());
    inputImage2->SetSpacing(inputImage->GetSpacing());
    inputImage2->SetBufferedRegion(brainRegionCrop);
    inputImage2->SetLargestPossibleRegion(brainRegionCrop);
    //inputImage2->FillBuffer(0.0);
    inputImage2->Allocate();

    ImageIterator cropIt(inputImage2,brainRegionCrop);
    ImageIterator inputIt(inputImage,brainRegion2);

    int cpt=0;
    float value = 0.0;
    for(cropIt.GoToBegin(),inputIt.GoToBegin(); !inputIt.IsAtEnd(); ++cropIt,++inputIt)
    {
        value = inputIt.Get();
        cropIt.Set(value);
    }
    inputImage2->Update();

    float alpha = compute_alpha(inputImage2.GetPointer());

    /*
    typedef itk::ImageFileWriter<ImageType> WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName("/Users/sebastientourbier/Desktop/original_image2.nii.gz");
    writer->SetInput(inputImage2.GetPointer());
    writer->Update();
    */

    //inputImage = inputImage2;
    inputRegion = inputImage2->GetLargestPossibleRegion();
    inputSize = inputRegion.GetSize();

    //std::cout << "input region : " << inputRegion << std::endl;

    ImageType::IndexType indexCenter;
    indexCenter[0] = floor(0.5 * inputSize[0]);
    indexCenter[1] = floor(0.5 * inputSize[1]);
    indexCenter[2] = floor(0.5 * inputSize[2]);
    //std::cout << "Index center : [" << indexCenter[0] << "," << indexCenter[1] << "," << indexCenter[2] << "]" << std::endl;

    ImageType::RegionType bgRegion = brainRegionCrop;
    ImageType::RegionType  fgRegion = brainRegionCrop;

    ImageType::SizeType bgSize;
    ImageType::SizeType fgSize;

    ImageType::IndexType bgIndexStart;
    ImageType::IndexType fgIndexStart;

    int windowRadius = 1;
    int bgBound = 1;
    int iter = 0;

    float bgMean = 0.0;
    float fgMean = 0.0;

    double CMI = 0.0; //Contrast Measurement Index based on the Logarithmic Image Processing (LIP) model

    std::cout << "#########################################################" << std::endl;
    while(brainRegionCrop.IsInside(bgRegion))
    {
        iter++;
        //std::cout << "" << iter << std::endl;
        fgIndexStart[0] = indexCenter[0] - windowRadius;
        fgIndexStart[1] = indexCenter[1] - windowRadius;
        fgIndexStart[2] = indexCenter[2] - windowRadius;

        bgIndexStart[0] = fgIndexStart[0] - bgBound;
        bgIndexStart[1] = fgIndexStart[1] - bgBound;
        bgIndexStart[2] = fgIndexStart[2] - bgBound;

        fgSize[0] = 2 * windowRadius + 1;
        fgSize[1] = 2 * windowRadius + 1;
        fgSize[2] = 2 * windowRadius + 1;

        bgSize[0] = fgSize[0] + 2*bgBound;
        bgSize[1] = fgSize[1] + 2*bgBound;
        bgSize[2] = fgSize[2] + 2*bgBound;

        bgRegion.SetIndex(bgIndexStart);
        bgRegion.SetSize(bgSize);
        //bgRegion.Print(std::cout);

        fgRegion.SetIndex(fgIndexStart);
        fgRegion.SetSize(fgSize);
        //fgRegion.Print(std::cout);

        if(!brainRegionCrop.IsInside(bgRegion)) break;

        ImageIterator itBg( inputImage2, bgRegion );
        ImageIterator itFg( inputImage2, fgRegion );

        //Compute the mean value in the foreground region
        for (itFg.GoToBegin(); !itFg.IsAtEnd(); ++itFg)
        {
            fgMean += itFg.Get();
        }
        fgMean = fgMean / (fgSize[0]*fgSize[1]*fgSize[2]);
        //std::cout << "Foreground mean : " << fgMean << " , ";

        //Compute the mean value in the background region
        for (itBg.GoToBegin(); !itBg.IsAtEnd(); ++itBg)
        {
            bgMean += itBg.Get();
        }
        bgMean = bgMean / (bgSize[0]*bgSize[1]*bgSize[2]);
        //std::cout << "Background mean : " << bgMean << std::endl;

        //Implementation based on Mridul Trivedi et al., "A No-Reference Image Quality Index for Contrastand Sharpness Measurement", 2011
        CMI += log( abs( (fgMean + bgMean - ( (fgMean*bgMean) / maxValue) ) / ( maxValue * ((fgMean - bgMean) / (maxValue - bgMean)) ) ) );

        //Compute alpha : maximum of the Hadamard Transform computed for the original input image I

        //Implement CSMI - Contrast and Sharpness Measurement Index
        //std::cout << "---------------------------------------------------------" << std::endl;
        windowRadius++;
    }

    CMI = CMI / iter;

    std::cout << "Contrast Measurement Index (CMI) before mult. by alpha: " << CMI << std::endl;



    std::cout << "Alpha : " << alpha << std::endl;

    CMI = alpha * CMI;

    std::cout << "Contrast Measurement Index (CMI) : " << CMI << std::endl;

    /*
    double InputImageMeanIntensityValues = 0;

    for (iti.GoToBegin(); !iti.IsAtEnd(); ++iti)
        InputImageMeanIntensityValues += iti.Get();

    InputImageMeanIntensityValues /= inputRegion.GetNumberOfPixels();

    //  std::cout << "Input image intensity mean:      " << InputImageMeanIntensityValues << std::endl;

    double varianceSharpnessMeasure = 0;

    for (iti.GoToBegin(); !iti.IsAtEnd(); ++iti)
        varianceSharpnessMeasure += pow((iti.Get() - InputImageMeanIntensityValues),2);

    varianceSharpnessMeasure /= inputRegion.GetNumberOfPixels();

    gradientFilter->SetInput( inputImage );
    gradientFilter->Update();

    ImageType::Pointer gradientMagnitudeImage = gradientFilter->GetOutput();

    ImageIterator itgm( gradientMagnitudeImage, gradientMagnitudeImage->GetLargestPossibleRegion() );

    double M2norm = 0;

    for (itgm.GoToBegin(); !itgm.IsAtEnd(); ++itgm)
        M2norm += pow( itgm.Get(), 2 );

    std::cout << varianceSharpnessMeasure << "  ";
    std::cout << M2norm << std::endl;
    */

    return EXIT_SUCCESS;
}

vnl_matrix<double> hadamard( int matSize)
{
    vnl_matrix<double> hadamardMat(matSize,matSize,0.0);
    hadamardMat(0,0) = 1.0;
    hadamardMat(1,0) = 1.0;
    hadamardMat(0,1) = 1.0;
    hadamardMat(1,1) = -1.0;

    int e = (int)(log2(matSize));

    std::cout << "Exponent : " << e << std::endl;

    for( int l=2; l<=e; l++ )
    {
        int block = pow(2,l);


        //std::cout << l << " , " << block << std::endl;

        for( int i=0; i<block; i++)
        {
            for( int j=0; j<block; j++)
            {
                int dup_block = pow(2,(l-1));
                //std::cout << dup_block << std::endl;


                if( (i < dup_block) && (j >= dup_block) )
                {
                    //std::cout<< "block2" << std::endl;
                    hadamardMat(i,j) = hadamardMat(i,j-dup_block);
                }
                else if( (i >= dup_block) && (j < dup_block) )
                {
                    //std::cout<< "block3" << std::endl;
                    hadamardMat(i,j) = hadamardMat(i-dup_block,j);
                }
                else if( (i >= dup_block) && (j >= dup_block) )
                {
                    //std::cout<< "block4" << std::endl;
                    hadamardMat(i,j) = -1.0 * hadamardMat(i-dup_block,j-dup_block);
                }
                else
                {
                    //std::cout<< "block1" << std::endl;
                    hadamardMat(i,j) = hadamardMat(i,j);
                }
            }
        }
    }

    std::cout<< "Hadamard matrix of size : " << matSize << std::endl;
    /*
    for( int i = 0; i<matSize; i++)
        for( int j=0; j<matSize; j++)
        {
            if(j<matSize-1)
            {
                if(hadamardMat(i,j) > 0) std::cout << " ";
                std::cout << hadamardMat(i,j) << " ";
            }
            else
            {
                if(hadamardMat(i,j) > 0) std::cout << " ";
                std::cout << hadamardMat(i,j) << std::endl;
            }
        }
    */

    return hadamardMat;
}

float compute_alpha(itk::Image<float,3>::Pointer Im)
{
    typedef itk::Image<float,3> ImageType;
    typedef itk::Image<float,2> SliceType;

    ImageType::RegionType imRegion = Im->GetLargestPossibleRegion();
    ImageType::SizeType   imSize   = imRegion.GetSize();
    ImageType::IndexType   imIndex   = imRegion.GetIndex();

    //std::cout << Im->GetBufferedRegion() << std::endl;
    //std::cout << Im->GetLargestPossibleRegion() << std::endl;

    //Compute 3D Hadamard transform based on Matlab source code: Faster 3D Walsh - Hadamard Transform (sequency, natural)
    //Provided in the file exchange
    //Make the assumption image size along x and y directions are equal

    vnl_matrix<double> W = hadamard(imSize[0]);

    //itk::ImageDuplicator<ImageType>::Pointer duplicator = itk::ImageDuplicator<ImageType>::New();
    //duplicator->SetInputImage(Im);
    //duplicator->Update();

    ImageType::Pointer outImage = ImageType::New();
    outImage->SetOrigin(Im->GetOrigin());
    outImage->SetDirection(Im->GetDirection());
    outImage->SetSpacing(Im->GetSpacing());
    outImage->SetBufferedRegion(Im->GetLargestPossibleRegion());
    outImage->Allocate();

    //duplicator->GetOutput();

    //First do 2D transform of the x-y-plane through all z layers
    typedef itk::RegionOfInterestImageFilter< ImageType, ImageType > ROI2DFilterType;
    ROI2DFilterType::Pointer roi2DFilter = ROI2DFilterType::New();

    //std::cout << imSize << std::endl;

    for(int sliceID=0; sliceID<imSize[2]; sliceID++)
    {
        //std::cout << "slice id (x/y): " << sliceID << std::endl;
        ImageType::IndexType  sliceIndex;
        sliceIndex[0] = imIndex[0];
        sliceIndex[1] = imIndex[1];
        sliceIndex[2] = sliceID;

        ImageType::SizeType sliceSize;
        sliceSize[0] = imSize[0];
        sliceSize[1] = imSize[1];
        sliceSize[2] = 1;

        ImageType::RegionType sliceRegion;
        sliceRegion.SetIndex(sliceIndex);
        sliceRegion.SetSize(sliceSize);

        roi2DFilter->SetInput(Im);
        roi2DFilter->SetRegionOfInterest(sliceRegion);
        roi2DFilter->Update();

        ImageType::Pointer slice = roi2DFilter->GetOutput();

        itk::ImageRegionIteratorWithIndex<ImageType> sliceIt(slice,slice->GetLargestPossibleRegion());
        vnl_matrix<double> sliceX(imSize[0],imSize[1],0.0);

        //Convert slice into vnl matrix
        for (sliceIt.GoToBegin(); !sliceIt.IsAtEnd(); ++sliceIt)
        {
            ImageType::IndexType ind = sliceIt.GetIndex();
            sliceX(ind[0],ind[1]) = sliceIt.Get();
            //if(sliceX(ind[0],ind[1])>0) std::cout << sliceX(ind[0],ind[1]) << std::endl;
        }

        //Apply 2D Hadamard transform on the current slice
        sliceX = (W * sliceX) * W;

        itk::ImageRegionIteratorWithIndex<ImageType> sliceImIt(outImage,sliceRegion);
        for (sliceImIt.GoToBegin(); !sliceImIt.IsAtEnd(); ++sliceImIt)
        {
            ImageType::IndexType ind = sliceImIt.GetIndex();
            sliceImIt.Set(sliceX(ind[0],ind[1]));
        }

    }

    /*
    typedef itk::ImageFileWriter<ImageType> WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName("/Users/sebastientourbier/Desktop/hadamard_image_2d.nii.gz");
    writer->SetInput(outImage);
    writer->Update();
    */

    //Now perform 1D transform along the z direction

    for(int sliceID=0; sliceID<imSize[1]; sliceID++)
    {
        //std::cout << "slice id (z): " << sliceID << std::endl;
        ImageType::IndexType  sliceIndex;
        sliceIndex[0] = imIndex[0];
        sliceIndex[1] = sliceID;
        sliceIndex[2] = imIndex[2];

        ImageType::SizeType sliceSize;
        sliceSize[0] = imSize[0];
        sliceSize[1] = 1;
        sliceSize[2] = imSize[2];

        ImageType::RegionType sliceRegion;
        sliceRegion.SetIndex(sliceIndex);
        sliceRegion.SetSize(sliceSize);

        roi2DFilter->SetInput(outImage);
        roi2DFilter->SetRegionOfInterest(sliceRegion);
        roi2DFilter->Update();

        ImageType::Pointer slice = roi2DFilter->GetOutput();

        itk::ImageRegionIteratorWithIndex<ImageType> sliceIt(slice,slice->GetLargestPossibleRegion());
        vnl_matrix<double> sliceX(imSize[0],imSize[2],0.0);

        //Convert slice into vnl matrix
        for (sliceIt.GoToBegin(); !sliceIt.IsAtEnd(); ++sliceIt)
        {
            ImageType::IndexType ind = sliceIt.GetIndex();
            //std::cout << ind << std::endl;
            sliceX(ind[0],ind[2]) = sliceIt.Get();
            //if(sliceX(ind[0],ind[1])>0) std::cout << sliceX(ind[0],ind[1]) << std::endl;
        }

        //Apply 2D Hadamard transform on the current slice

        //float factor = 1 / pow(2,(0.5*(log2(imSize[0])))+(0.5*(log2(imSize[1])))+(0.5*(log2(imSize[2]))));
        //sliceX = (sliceX * W) * factor;
        sliceX = (sliceX * W) / sqrt(imSize[0]*imSize[1]*imSize[2]); //faster

        itk::ImageRegionIteratorWithIndex<ImageType> sliceImIt(outImage,sliceRegion);
        for (sliceImIt.GoToBegin(); !sliceImIt.IsAtEnd(); ++sliceImIt)
        {
            ImageType::IndexType ind = sliceImIt.GetIndex();
            sliceImIt.Set(sliceX(ind[0],ind[2]));
        }

    }

    /*
    writer->SetFileName("/Users/sebastientourbier/Desktop/hadamard_image_3d.nii.gz");
    writer->SetInput(outImage);
    writer->Update();
    */

    //Extract maximum value in the Hadamard transform : it corresponds to the value of alpha
    itk::ImageRegionIteratorWithIndex<ImageType> sliceImIt(outImage,outImage->GetLargestPossibleRegion());
    float alpha = 0.0;

    for (sliceImIt.GoToBegin(); !sliceImIt.IsAtEnd(); ++sliceImIt)
        if(sliceImIt.Get()>=alpha) alpha = sliceImIt.Get();

    return alpha;

}

