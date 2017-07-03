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

/* Itk includes */
#include "itkImage.h"
#include "itkImageRegionIterator.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageMaskSpatialObject.h"
#include "itkCastImageFilter.h"
#include "itkImageDuplicator.h"

#include "itkDiscreteGaussianImageFilter.h"
#include "itkCurvatureAnisotropicDiffusionImageFilter.h"

int main( int argc, char *argv[] )
{

    try {

        const char *input = NULL;
        const char *outImage = NULL;

        double conductance = 0.0;
        double timeStep = 0.0;

        int numberOfIterations = 0;

        // Parse arguments

        TCLAP::CmdLine cmd("Performs anisotropic Gaussian denoising.", ' ', "Unversioned");

        TCLAP::ValueArg<std::string> inputArg("i","input","Low-resolution image file",true,"","string",cmd);
        TCLAP::ValueArg<std::string> outArg  ("o","output","Super resolution output image",true,"","string",cmd);

        TCLAP::ValueArg<double> condArg  ("","conductance","Conductance (default = 3.0)",false, 3.0,"double",cmd);
        TCLAP::ValueArg<double> tsArg  ("","time-step","Time step (default = 0.0125)",false, 0.0125,"double",cmd);
        TCLAP::ValueArg<int> iterArg  ("","iter","Number of iterations (denoising) (default = 3)",false, 3,"int",cmd);


        // Parse the argv array.
        cmd.parse( argc, argv );

        input = inputArg.getValue().c_str();
        outImage = outArg.getValue().c_str();

        numberOfIterations = iterArg.getValue();
        conductance = condArg.getValue();
        timeStep = tsArg.getValue();

        /** Typedef */
        const   unsigned int    Dimension = 3;

        typedef float  PixelType;

        typedef itk::Image< PixelType, Dimension >  ImageType;
        typedef ImageType::Pointer                  ImagePointer;
        typedef std::vector<ImagePointer>           ImagePointerArray;

        typedef itk::Image< unsigned char, Dimension >  ImageMaskType;
        typedef itk::ImageFileReader< ImageMaskType > MaskReaderType;
        typedef itk::ImageMaskSpatialObject< Dimension > MaskType;

        typedef ImageType::SizeType    SizeType;

        typedef ImageType::RegionType               RegionType;
        typedef std::vector< RegionType >           RegionArrayType;

        typedef itk::ImageRegionConstIteratorWithIndex< ImageType > IteratorType;

        typedef itk::ImageFileReader< ImageType >   ImageReaderType;
        typedef itk::ImageFileWriter< ImageType >   WriterType;

        typedef itk::DiscreteGaussianImageFilter< ImageType, ImageType > GaussianFilterType;
        typedef itk::CurvatureAnisotropicDiffusionImageFilter< ImageType, ImageType >  ADGaussianFilterType;

        /** Load image(s). */
        std::cout << "Loading the input image : " << input << std::endl;
        ImageReaderType::Pointer inputReader = ImageReaderType::New();
        inputReader -> SetFileName( input );

        /** Anisotropic diffusion Gaussian filtering */
        std::cout << "Performing anisotropic Gaussian denoising ( ";
        std::cout << "Number of iterations : " << numberOfIterations << " , ";
        std::cout << "timestep : " << timeStep << " , " ;
        std::cout << "conductance : " << conductance << " , " ;
        std::cout << std::endl;
        ADGaussianFilterType::Pointer ADGaussianFilter = ADGaussianFilterType::New();
        ADGaussianFilter -> SetInput( inputReader -> GetOutput() );
        ADGaussianFilter -> SetNumberOfIterations( numberOfIterations );
        ADGaussianFilter -> SetTimeStep( timeStep );
        ADGaussianFilter -> SetConductanceParameter( conductance );

        std::cout << "Writing the output image : " << outImage << std::endl;
        WriterType::Pointer writer = WriterType::New();
        writer -> SetInput( ADGaussianFilter -> GetOutput() );
        writer -> SetFileName( outImage );
        writer -> Update();

    } catch (TCLAP::ArgException &e)  // catch any exceptions
    { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

    return EXIT_SUCCESS;
}

