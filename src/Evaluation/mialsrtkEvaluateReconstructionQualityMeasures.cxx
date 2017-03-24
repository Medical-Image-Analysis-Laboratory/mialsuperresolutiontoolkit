
/*=========================================================================

Program: Computes Reconstruction Quality Measures (MSE, RMSE, PSNR) given a Reference Image
Language: C++
Date: $Date$
Version: 1.0
Author: Sebastien Tourbier

=========================================================================*/
/* Standard includes */
#include <tclap/CmdLine.h>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>

/* ITK */
#include "itkImageFileReader.h"
#include "itkImage.h"
#include "itkImageMaskSpatialObject.h"

#include "itkOrientImageFilter.h"

#include "itkMinimumMaximumImageCalculator.h"
#include "itkStatisticsImageFilter.h"

#include "itkMultiThreader.h"

#include "vcl_algorithm.h"

/* MIALTK */
#include "mialtkMaths.h"

/* VTK */
#include <vtkVersion.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>
#include <vtkChartXY.h>
#include <vtkTable.h>
#include <vtkPlot.h>
#include <vtkFloatArray.h>
#include <vtkContextView.h>
#include <vtkContextScene.h>
#include <vtkPen.h>

void prompt_start(std::vector< std::string > & inputFileNames, const char* refFileName,  const char* csvFileName)
{
    unsigned int numberOfImages = inputFileNames.size();

    std::cout << std::endl << "----------------------------------------------------------------"<<std::endl;
    std::cout << " Evaluation Program for reconstruction quality " << std::endl;
    std::cout << "----------------------------------------------------------------"<<std::endl<<std::endl;
    std::cout << std::endl << "Number of images : " << inputFileNames.size() << std::endl << std::endl;

    for(unsigned int i=0; i < numberOfImages; i++)
    {
        std::cout << "Input image " << int2str(i) << ":" <<inputFileNames[i] << std::endl;
    }

    std::cout << "Ref image: " <<refFileName << std::endl;
    std::cout << "CSV file: " << csvFileName << std::endl << std::endl;

    std::cout << "###################################################### \n" << std::endl;

};

int main( int argc, char * argv [] )
{
    try {
        std::vector< std::string > inputFileNames;

        const char *refFileName;
        const char *csvFileName;

        const char *undefined = "";

        // Parse arguments

        TCLAP::CmdLine cmd("Evaluation of reconstruction quality given a reference", ' ', "Unversioned");

        TCLAP::MultiArg<std::string> inputArg("i","input","reconstructed image file",true,"string",cmd);
        TCLAP::ValueArg<std::string> refArg  ("r","ref","reference image file",true,"","string",cmd);
        TCLAP::ValueArg<std::string> csvArg  ("","csv","CSV file where measures are saved",false,"","string",cmd);


        // Parse the argv array.
        cmd.parse( argc, argv );

        inputFileNames = inputArg.getValue();
        refFileName = refArg.getValue().c_str();
        csvFileName = csvArg.getValue().c_str();

        prompt_start( inputFileNames , refFileName, csvFileName );

        //Typedef
        const unsigned int Dimension = 3;
        typedef short PixelType;

        typedef itk::Image< PixelType, Dimension > ImageType;
        typedef ImageType::Pointer ImagePointer;

        typedef ImageType::RegionType RegionType;
        typedef std::vector< RegionType > RegionArrayType;

        typedef itk::ImageFileReader< ImageType > ImageReaderType;

        typedef itk::Image< unsigned char, Dimension > ImageMaskType;
        typedef ImageMaskType::Pointer ImageMaskPointer;

        typedef itk::ImageFileReader< ImageMaskType > MaskReaderType;

        typedef itk::ImageMaskSpatialObject< Dimension > MaskType;
        typedef MaskType::Pointer MaskPointer;

        typedef itk::OrientImageFilter<ImageType,ImageType> OrientImageFilterType;
        typedef OrientImageFilterType::Pointer OrientImageFilterPointer;

        typedef itk::OrientImageFilter<ImageMaskType,ImageMaskType> OrientImageMaskFilterType;
        typedef OrientImageMaskFilterType::Pointer OrientImageMaskFilterPointer;

        typedef itk::StatisticsImageFilter<ImageType> StatisticsImageFilterType;


        // Number of images being evaluated
        unsigned int numberOfImages = inputFileNames.size();

        //Read reconstructed images
        std::vector<ImagePointer> images(numberOfImages);

        for(unsigned int i=0; i< numberOfImages; i++)
        {
            ImageReaderType::Pointer imReader = ImageReaderType::New();
            imReader -> SetFileName( inputFileNames[i].c_str() );
            imReader -> Update();

            OrientImageFilterPointer orienter = OrientImageFilterType::New();
            orienter->UseImageDirectionOn();
            orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS);

            orienter->SetInput(imReader -> GetOutput());
            orienter->Update();

            images[i] = orienter -> GetOutput();
            //images[i] = imReader -> GetOutput();

        }

        //Read reference image
        ImagePointer refImage = ImageType::New();

        ImageReaderType::Pointer refReader = ImageReaderType::New();
        refReader -> SetFileName( refFileName );
        refReader -> Update();

        OrientImageFilterPointer refOrienter = OrientImageFilterType::New();
        refOrienter->UseImageDirectionOn();
        refOrienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS);

        refOrienter->SetInput(refReader -> GetOutput());
        refOrienter->Update();

        refImage = refOrienter -> GetOutput();
        //refImage = refReader -> GetOutput();

        //Get common region between the image images[0] to be evaluated and the reference image
        ImageType::RegionType imRegion = images[0] -> GetLargestPossibleRegion();
        ImageType::IndexType imIndex = imRegion.GetIndex();

        //Correspondance between index of the image being evaluated and index of the reference image
        ImageType::PointType imPoint;
        images[0] -> TransformIndexToPhysicalPoint( imIndex , imPoint );

        ImageType::IndexType refIndex;
        refImage -> TransformPhysicalPointToIndex( imPoint , refIndex );

        ImageType::SizeType imSize = imRegion.GetSize();

        ImageType::RegionType newRefRegion( refIndex , imSize );

        std::cout << "Summary of Reference Image Crop" << std::endl << std::endl;
        std::cout << "Old reference region : " << refImage -> GetLargestPossibleRegion() << std::endl;
        std::cout << "New reference region : " << newRefRegion << std::endl;

        //Vectorize the reference image
        itk::ImageRegionConstIteratorWithIndex < ImageType > refIt( refImage , newRefRegion );
        unsigned int linearIndex = 0;
        int counter=0;
        vnl_vector<float> ref;
        ref.set_size( newRefRegion.GetNumberOfPixels() );
        for (refIt.GoToBegin(); !refIt.IsAtEnd(); ++refIt, linearIndex++)
        {
            ref[linearIndex] = refIt.Get();
            if(refIt.Get()>0)
                counter++;
        }
        std::cout << "Number of voxels in brain mask : " << counter << "( " << newRefRegion.GetNumberOfPixels() << " in total )" << std::endl;

        //Vectorize all images being evaluated
        std::vector < vnl_vector<float> > ims(numberOfImages);

        for(unsigned int i = 0 ; i < numberOfImages ; i++ )
        {
            itk::ImageRegionConstIteratorWithIndex < ImageType > imIt( images[i] , imRegion );
            unsigned int linearIndex = 0;
            ims[i].set_size( imRegion.GetNumberOfPixels() );

            for (imIt.GoToBegin(); !imIt.IsAtEnd(); ++imIt, linearIndex++)
            {
                ims[i][linearIndex] = imIt.Get();
            }
        }

        //Compute dynamic range of reference image (used to compute NRMSE and PSNR)
        StatisticsImageFilterType::Pointer statisticsImageFilter = StatisticsImageFilterType::New ();
        statisticsImageFilter->SetInput(refImage);
        statisticsImageFilter->Update();

        float refDynRange = statisticsImageFilter -> GetMaximum() - statisticsImageFilter -> GetMinimum();

        std::cout << "###################################################### \n" << std::endl;
        std::cout << std::endl << "Statistics of reference image  " << std::endl;
        std::cout << "Mean: " << statisticsImageFilter->GetMean() << std::endl;
        std::cout << "Std.: " << statisticsImageFilter->GetSigma() << std::endl;
        std::cout << "Min: " << statisticsImageFilter->GetMinimum() << std::endl;
        std::cout << "Max: " << statisticsImageFilter->GetMaximum() << std::endl;
        std::cout << "Dyn. range: " << refDynRange << std::endl << std::endl;

        std::cout << "###################################################### \n" << std::endl;

        //Perform evaluation of each image i with respect to the reference within the brain mask (Manually)
        float level = 0.0;
        std::vector<float> mseValues( numberOfImages );
        std::vector<float> rmseValues( numberOfImages );
        std::vector<float> nrmseValues( numberOfImages );
        std::vector<float> psnrValues( numberOfImages );

        for( unsigned int i = 0 ; i < numberOfImages ; i++)
        {
            std::cout << "Processing image # " << int2str(i) << " : " << std::endl;
            mseValues[i] = mialtkComputeMSE( ref , ims[i] , level );
            psnrValues[i] = mialtkComputePSNR( ref , ims[i] , level );

            rmseValues[i] = std::sqrt( mseValues[i] );
            nrmseValues[i] = rmseValues[i] / refDynRange;

            std::cout << "Intensity level : " << level << std::endl;
        }

        std::cout << "###################################################### \n" << std::endl;
        //Display the measures obtained for each image. Measures are saved in a csv file (if provided).
        for( unsigned int i=0; i < numberOfImages; i++)
        {
            std::cout << "Input image" << int2str(i) << " : " << inputFileNames[i] << std::endl;
            std::cout << "MSE : " << mseValues[i] << std::endl;
            std::cout << "RMSE : " << rmseValues[i] << std::endl;
            std::cout << "NRMSE : " << nrmseValues[i] << std::endl;
            std::cout << "PSNR : " << psnrValues[i] << " dB \n" << std::endl;

            if( csvFileName != undefined )
            {
                std::ofstream fout(csvFileName, std::ios_base::out | std::ios_base::app);
                fout << " imageFile = "<< inputFileNames[i] << " , ";
                fout << mseValues[i] << " , " << rmseValues[i] << " , " << nrmseValues[i] << " , " << psnrValues[i] << std::endl;
                fout.close();
            }

            std::cout << "------------------------------------------------------------------------------------------------------------------------- \n" << std::endl;

        }

        //Plot XY measures
        /*

        // Create a table with some points in it
        vtkSmartPointer<vtkTable> table =  vtkSmartPointer<vtkTable>::New();

        vtkSmartPointer<vtkFloatArray> arrX = vtkSmartPointer<vtkFloatArray>::New();
        arrX->SetName("Outer iteration index");
        table->AddColumn(arrX);

        vtkSmartPointer<vtkFloatArray> arrY = vtkSmartPointer<vtkFloatArray>::New();
        arrY->SetName("PSNR");
        table->AddColumn(arrY);

        // Fill in the table with some example values
        table->SetNumberOfRows(numberOfImages);
        for (int i = 0; i < numberOfImages; ++i)
        {
            table->SetValue(i, 0, i );
            table->SetValue(i, 1, psnrValues[i]);
        }

        // Set up the view
        vtkSmartPointer<vtkContextView> view =  vtkSmartPointer<vtkContextView>::New();
        view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);

        // Add multiple line plots, setting the colors etc
        vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
        view->GetScene()->AddItem(chart);
        vtkPlot *line = chart->AddPlot(vtkChart::LINE);

#if VTK_MAJOR_VERSION <= 5
        line->SetInput(table, 0, 1);
#else
        line->SetInputData(table, 0, 1);
#endif

        line->SetColor(0, 0, 255, 255);
        line->SetWidth(1.0);
        line = chart->AddPlot(vtkChart::LINE);

        // Start interactor
        view->GetInteractor()->Initialize();
        view->GetInteractor()->Start();

        */

        return EXIT_SUCCESS;

    }
    catch (TCLAP::ArgException &e) // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }
};
