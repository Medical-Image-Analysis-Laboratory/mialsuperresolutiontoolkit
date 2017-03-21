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
#include "itkWrapPadImageFilter.h"
#include "itkConstantPadImageFilter.h"
#include "itkForwardFFTImageFilter.h"
#include "itkFFTShiftImageFilter.h"
#include "itkComplexToRealImageFilter.h"
#include "itkComplexToImaginaryImageFilter.h"
#include "itkComplexToModulusImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkFFTPadImageFilter.h"

int main( int argc, char *argv[] )
{

    try {

        const char *input = NULL;
        const char *outRealImage = NULL;
        const char *outImgImage = NULL;
        const char *outModImage = NULL;

        double conductance = 0.0;
        double timeStep = 0.0;

        int numberOfIterations = 0;

        // Parse arguments

        TCLAP::CmdLine cmd("Performs anisotropic Gaussian denoising.", ' ', "Unversioned");

        TCLAP::ValueArg<std::string> inputArg("i","input","Low-resolution image file",true,"","string",cmd);
        TCLAP::ValueArg<std::string> outRealArg  ("","output-real","Real part output image",true,"","string",cmd);
        TCLAP::ValueArg<std::string> outImgArg  ("","output-img","Img part output image",true,"","string",cmd);
        TCLAP::ValueArg<std::string> outModArg  ("","output-mod","Magnitude output image",true,"","string",cmd);


        // Parse the argv array.
        cmd.parse( argc, argv );

        input = inputArg.getValue().c_str();
        outRealImage = outRealArg.getValue().c_str();
        outImgImage = outImgArg.getValue().c_str();
        outModImage = outModArg.getValue().c_str();

        /* typedef */
        const unsigned int Dimension = 3;
        typedef float                                   FloatPixelType;
        typedef itk::Image< FloatPixelType, Dimension > FloatImageType;
        typedef itk::ImageFileReader< FloatImageType >  ReaderType;


        typedef unsigned char UnsignedCharPixelType;
        typedef itk::Image< UnsignedCharPixelType, Dimension > UnsignedCharImageType;
        typedef itk::ImageFileWriter< UnsignedCharImageType > WriterType;


        //typedef itk::ConstantPadImageFilter< FloatImageType, FloatImageType > PadFilterType;
         typedef itk::FFTPadImageFilter< FloatImageType, FloatImageType > PadFilterType;

        typedef itk::ForwardFFTImageFilter< FloatImageType > FFTType;
        typedef FFTType::OutputImageType FFTOutputImageType;

        //FFT shift: center the zero frequency component
        typedef itk::FFTShiftImageFilter<  UnsignedCharImageType, UnsignedCharImageType > FFTShiftFilterType;
         FFTShiftFilterType::Pointer fftShiftFilter = FFTShiftFilterType::New();

        typedef itk::ComplexToRealImageFilter< FFTOutputImageType, FloatImageType> RealFilterType;
        typedef itk::ComplexToImaginaryImageFilter< FFTOutputImageType, FloatImageType> ImaginaryFilterType;
        typedef itk::ComplexToModulusImageFilter< FFTOutputImageType, FloatImageType> ModulusFilterType;

        typedef itk::RescaleIntensityImageFilter< FloatImageType, UnsignedCharImageType > RescaleFilterType;


        ReaderType::Pointer reader = ReaderType::New();
        reader->SetFileName( input );
        reader->Update();

        FloatImageType::SizeType size = reader->GetOutput()->GetLargestPossibleRegion().GetSize();

        // Some FFT filter implementations, like VNL's, need the image size to be a
        // multiple of small prime numbers.

        PadFilterType::Pointer padFilter = PadFilterType::New();
        padFilter->SetInput( reader->GetOutput() );
        PadFilterType::SizeType padding;
        // Input size is [48, 62, 42].  Pad to [48, 64, 48].
//        padding[0] = (int)((512 - size[0])/2);
//        padding[1] = (int)((512 - size[1])/2);
//        padding[2] =  (int)((512 - size[2])/2);
//        padFilter->SetPadUpperBound( padding );
//        padding[0] =  512 - size[0] - padding[0];
//        padding[1] =  512 - size[1] - padding[1];
//        padding[2] = 512 - size[2] - padding[2];

        padding[0] = (int)(((floor(size[0]/2)*2+2) - size[0])/2);
        padding[1] = (int)(((floor(size[1]/2)*2+2) - size[1])/2);
        padding[2] =  (int)(((floor(size[2]/2)*2+2) - size[2])/2);
        //padFilter->SetPadUpperBound( padding );

        padding[0] =  (floor(size[0]/2)*2+2) - size[0] - padding[0];
        padding[1] =  (floor(size[1]/2)*2+2) - size[1] - padding[1];
        padding[2] = (floor(size[2]/2)*2+2) - size[2] - padding[2];
        //padFilter->SetPadLowerBound( padding );

        //padFilter->SetConstant(0.0);
        padFilter->Update();

        FFTType::Pointer fftFilter = FFTType::New();
        fftFilter->SetInput( padFilter->GetOutput() );
        fftFilter->Update();

        // Extract the real part

        RealFilterType::Pointer realFilter = RealFilterType::New();
        realFilter->SetInput(fftFilter->GetOutput());

        RescaleFilterType::Pointer realRescaleFilter = RescaleFilterType::New();
        realRescaleFilter->SetInput(realFilter->GetOutput());
        realRescaleFilter->SetOutputMinimum( itk::NumericTraits< UnsignedCharPixelType >::min() );
        realRescaleFilter->SetOutputMaximum( itk::NumericTraits< UnsignedCharPixelType >::max() );

        fftShiftFilter->SetInput( realRescaleFilter->GetOutput() );
        fftShiftFilter->Update();

        WriterType::Pointer realWriter = WriterType::New();
        realWriter->SetFileName( outRealImage );
        realWriter->SetInput(fftShiftFilter->GetOutput() );
        try
        {
            realWriter->Update();
        }
        catch( itk::ExceptionObject & error )
        {
            std::cerr << "Error: " << error << std::endl;
            return EXIT_FAILURE;
        }

        // Extract the imaginary part

        ImaginaryFilterType::Pointer imaginaryFilter = ImaginaryFilterType::New();
        imaginaryFilter->SetInput(fftFilter->GetOutput());
        RescaleFilterType::Pointer imaginaryRescaleFilter = RescaleFilterType::New();
        imaginaryRescaleFilter->SetInput(imaginaryFilter->GetOutput());
        imaginaryRescaleFilter->SetOutputMinimum( itk::NumericTraits< UnsignedCharPixelType >::min() );
        imaginaryRescaleFilter->SetOutputMaximum( itk::NumericTraits< UnsignedCharPixelType >::max() );

        fftShiftFilter->SetInput( imaginaryRescaleFilter->GetOutput() );
        fftShiftFilter->Update();

        WriterType::Pointer complexWriter = WriterType::New();
        complexWriter->SetFileName( outImgImage );
        complexWriter->SetInput( fftShiftFilter->GetOutput() );
        try
        {
            complexWriter->Update();
        }
        catch( itk::ExceptionObject & error )
        {
            std::cerr << "Error: " << error << std::endl;
            return EXIT_FAILURE;
        }

        // Compute the magnitude

        ModulusFilterType::Pointer modulusFilter = ModulusFilterType::New();
        modulusFilter->SetInput(fftFilter->GetOutput());
        RescaleFilterType::Pointer magnitudeRescaleFilter = RescaleFilterType::New();
        magnitudeRescaleFilter->SetInput(modulusFilter->GetOutput());
        magnitudeRescaleFilter->SetOutputMinimum( itk::NumericTraits< UnsignedCharPixelType >::min() );
        magnitudeRescaleFilter->SetOutputMaximum( itk::NumericTraits< UnsignedCharPixelType >::max() );

        fftShiftFilter->SetInput( magnitudeRescaleFilter->GetOutput() );
        fftShiftFilter->Update();

        WriterType::Pointer magnitudeWriter = WriterType::New();
        magnitudeWriter->SetFileName( outModImage );
        magnitudeWriter->SetInput( fftShiftFilter->GetOutput() );
        try
        {
            magnitudeWriter->Update();
        }
        catch( itk::ExceptionObject & error )
        {
            std::cerr << "Error: " << error << std::endl;
            return EXIT_FAILURE;
        }

    } catch (TCLAP::ArgException &e)  // catch any exceptions
    { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

    return EXIT_SUCCESS;
}

