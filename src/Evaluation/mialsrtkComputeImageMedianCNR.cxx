/* ============================================================================================

  Program :   High resolution Fetal and Pediatric MRI reconstruction
  Module  :   Compute image median CNR
  File    :   mialsrtkComputeImageMedianCNR.cxx
  Language:   C++
  Version :   $Revision: 1.0 $

  Copyright (c) 2017 Medical Image Analysis Laboratory (MIAL), Lausanne

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

  Latest modifications:  April 28, 2009

=============================================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImageMaskSpatialObject.h"
//#include "itkImageRegionIteratorWithIndex.h"

#include "itkImageDuplicator.h"

#include "itkRegionOfInterestImageFilter.h"

#include "itkConstantPadImageFilter.h"

#include "itkGradientMagnitudeImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"

#include "itkLabelObject.h"
#include "itkLabelMap.h"
#include "itkLabelImageToLabelMapFilter.h"
#include "itkLabelMapMaskImageFilter.h"

#include "mialsrtkMaths.h"


int main( int argc, char * argv[] )
{
    if( argc < 6 )
    {
        std::cerr << "Usage: " << std::endl;
        std::cerr << argv[0] << "  inputImageFile inputLabelFile roiFile label1Index label2Index " << std::endl;
        return EXIT_FAILURE;
    }

    const unsigned int  Dimension = 3;

    typedef    float    PixelType;

    typedef itk::Image< PixelType,  Dimension >   ImageType;

    typedef itk::ImageMaskSpatialObject<Dimension> ImageMaskSpatialObject;
    typedef itk::ImageMaskSpatialObject<Dimension>::ImageType ImageMaskType;

    typedef itk::ImageFileReader< ImageType >  ReaderType;
    typedef itk::ImageFileReader< ImageMaskType >  ReaderMaskType;


    ReaderType::Pointer reader1 = ReaderType::New();
    reader1->SetFileName( argv[1] );
    reader1->Update();

    /*
    typedef itk::ImageRegionIterator< ImageType > ImageIterator;
    ImageIterator iti( inputImage, inputRegion );

    //Extract the max pixel value
    float maxValue = 0.0f;
    for (iti.GoToBegin(); !iti.IsAtEnd(); ++iti)
    {
        if(iti.Get()>=maxValue) maxValue = iti.Get();
    }
    std::cout << "Maximum voxel value : " << maxValue << std::endl;
    */

    ReaderType::Pointer reader2 = ReaderType::New();
    reader2->SetFileName( argv[2] );
    reader2->Update();

    ReaderMaskType::Pointer reader3 = ReaderMaskType::New();
    reader3->SetFileName( argv[3] );
    reader3->Update();

    ImageMaskSpatialObject::Pointer maskSO = ImageMaskSpatialObject::New();
    maskSO->SetImage ( reader3->GetOutput() );
    ImageMaskType::RegionType ROI  = maskSO->GetAxisAlignedBoundingBoxRegion();
    std::cout << "ROI: " << ROI << std::endl;

    typedef itk::RegionOfInterestImageFilter< ImageType, ImageType > ROIFilterType;
    ROIFilterType::Pointer roiFilter1 = ROIFilterType::New();
    roiFilter1->SetInput(reader1->GetOutput());
    roiFilter1->SetRegionOfInterest(ROI);
    roiFilter1->Update();

    ImageType::Pointer ROIImage = roiFilter1->GetOutput();

    ROIFilterType::Pointer roiFilter2 = ROIFilterType::New();
    roiFilter2->SetInput(reader2->GetOutput());
    roiFilter2->SetRegionOfInterest(ROI);
    roiFilter2->Update();

    ImageType::Pointer ROIlabels = roiFilter2->GetOutput();

    typedef itk::LabelObject< PixelType, Dimension >  LabelObjectType;
    typedef itk::LabelMap< LabelObjectType >          LabelMapType;

    // convert the label image into a LabelMap
    typedef itk::LabelImageToLabelMapFilter< ImageType, LabelMapType > LabelImage2LabelMapType;
    LabelImage2LabelMapType::Pointer convert = LabelImage2LabelMapType::New();
    convert->SetInput(ROIlabels.GetPointer());

    //Mask label 1
    typedef itk::LabelMapMaskImageFilter< LabelMapType, ImageType > FilterType;
    FilterType::Pointer maskLabelFilter1 = FilterType::New();
    maskLabelFilter1->SetInput( convert->GetOutput() );
    maskLabelFilter1->SetFeatureImage( ROIImage.GetPointer() );

    // The label to be used to mask the image is passed via SetLabel
    maskLabelFilter1->SetLabel( atoi(argv[4]) );
    // The background in the output image (where the image is masked)
    // is passed via SetBackground
    maskLabelFilter1->SetBackgroundValue( 0.0 );
    // The user can choose to mask the image outside the label object
    // (default behavior), or inside the label object with the chosen label,
    // by calling SetNegated().
    //maskLabelFilter1->SetNegated( negated );
    // Finally, the image can be cropped to the masked region, by calling
    // SetCrop( true ), or to a region padded by a border, by calling both
    // SetCrop() and SetCropBorder().
    // The crop border defaults to 0, and the image is not cropped by default.
    maskLabelFilter1->SetCrop( true );
    //FilterType::SizeType border;
    //border.Fill( borderSize );
    //maskLabelFilter1->SetCropBorder( border );
    maskLabelFilter1->Update();

    ImageType::Pointer label1Im = maskLabelFilter1->GetOutput();


    //Mask label2
    typedef itk::LabelMapMaskImageFilter< LabelMapType, ImageType > FilterType;
    FilterType::Pointer maskLabelFilter2 = FilterType::New();
    maskLabelFilter2->SetInput( convert->GetOutput() );
    maskLabelFilter2->SetFeatureImage( ROIImage.GetPointer() );
    maskLabelFilter2->SetLabel( atoi(argv[5]) );
    maskLabelFilter2->SetBackgroundValue( 0.0 );
    maskLabelFilter2->SetCrop( true );
    maskLabelFilter2->Update();

    ImageType::Pointer label2Im = maskLabelFilter2->GetOutput();

    //Count number of voxels related to label1 / label2 respectively and store grayscale into two vectors

    typedef itk::ImageRegionIterator< ImageType > ImageIterator;

    ImageIterator itLabel1(label1Im,label1Im->GetLargestPossibleRegion());
    int cpt1 = 0;
    for (itLabel1.GoToBegin(); !itLabel1.IsAtEnd(); ++itLabel1)
    {
        if(itLabel1.Get()>0) cpt1++;
    }

   std::vector<double> vLabel1(cpt1,0.0);
    cpt1 = 0;
    for (itLabel1.GoToBegin(); !itLabel1.IsAtEnd(); ++itLabel1)
    {
        if(itLabel1.Get()>0)
        {
            vLabel1[cpt1] = itLabel1.Get();
            cpt1++;
        }
    }

    ImageIterator itLabel2(label2Im,label2Im->GetLargestPossibleRegion());
    int cpt2 = 0;
    for (itLabel2.GoToBegin(); !itLabel2.IsAtEnd(); ++itLabel2)
    {
        if(itLabel2.Get()>0) cpt2++;
    }

    std::vector<double> vLabel2(cpt2,0.0);
    cpt2 = 0;
    for (itLabel2.GoToBegin(); !itLabel2.IsAtEnd(); ++itLabel2)
    {
        if(itLabel2.Get()>0)
        {
            vLabel2[cpt2] = itLabel2.Get();
            cpt2++;
        }
    }

    //Compute median value for label1 and label2
    double median1 = mialsrtkMedian(vLabel1);
    double median2 = mialsrtkMedian(vLabel2);
    std::cout << "Label 1 median : " << median1 << std::endl;
    std::cout << "Label 2 median : " << median2 << std::endl;

    std::vector<double>::iterator it;
    for (it=vLabel1.begin(); it < vLabel1.end(); it++)
    {
           *it = abs(*it - median1);
    }

    std::vector<double>::iterator it2;
    for (it2=vLabel2.begin(); it2 < vLabel2.end(); it2++)
    {
           *it2 = abs(*it2 - median2);
    }

    double mad1 = mialsrtkMedian(vLabel1);
    double mad2 = mialsrtkMedian(vLabel2);
    std::cout << "Label 1 MAD : " << mad1 << std::endl;
    std::cout << "Label 2 MAD : " << mad2 << std::endl;

    double absCNR = abs(median1-median2) / abs(mad1-mad2);
    double sqCNR = ((median1-median2)*(median1-median2)) / ((mad1-mad2)*(mad1-mad2));

    std::cout << "absolute CNR : " << absCNR << std::endl;
    std::cout << "squared CNR : " << sqCNR << std::endl;

    return EXIT_SUCCESS;
}
