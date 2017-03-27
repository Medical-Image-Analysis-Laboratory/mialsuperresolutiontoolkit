/*==========================================================================

  © Université de Lausanne (UNIL) & Centre Hospitalier Universitaire de Lausanne (CHUV) - Centre d'Imagerie BioMédicale

  Date: 22/05/14
  Author(s): Sebastien Tourbier (sebastien.tourbier@unil.ch)

  ==========================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

/* Standard includes */
#include <tclap/CmdLine.h>
#include "stdio.h"

/* Itk includes */
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkStatisticsImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"

/* mialsrtk includes */
#include "mialsrtkMaths.h"

int main( int argc, char *argv[] )
{
    try {

        std::vector< std::string > inputs;
        std::vector< std::string > outputs;

        // Parse arguments
        TCLAP::CmdLine cmd("Rescale intensities between 0 and 255 in all images", ' ', "Unversioned");

        TCLAP::MultiArg<std::string> inputsArg("i","input","Input Image file",true,"string",cmd);
        TCLAP::MultiArg<std::string> outputsArg("o","output","Output Image file",true,"string",cmd);


        // Parse the argv array.
        cmd.parse( argc, argv );

        inputs = inputsArg.getValue();
        outputs = outputsArg.getValue();

        //Typedefs
        const    unsigned int    Dimension3D = 3;
        typedef  float           PixelType;

        typedef itk::Image< PixelType, Dimension3D >  ImageType;
        typedef ImageType::Pointer                  ImagePointer;

        typedef ImageType::RegionType               ImageRegionType;
        typedef std::vector< ImageRegionType >           ImageRegionArrayType;

        typedef itk::ImageFileReader< ImageType  >  ImageReaderType;
        typedef itk::ImageFileWriter< ImageType  >  ImageWriterType;

        typedef itk::StatisticsImageFilter<ImageType>  StatisticsImageFilterType;


        //Filter setup
        unsigned int numberOfImages = inputs.size();
        std::vector< ImagePointer >         images(numberOfImages);
        std::vector< ImagePointer >         outImages(numberOfImages);

        //TODO: create an array of vector for storing the normalize intensity differences for each slice in each image, i.e., std::vector < std::vector < float > > qualityMeasures(numberOfImages);


        //Load inputs
        for (unsigned int i=0; i<numberOfImages; i++)
        {
            std::cout<<"Reading image : "<<inputs[i].c_str()<<"\n";
            ImageReaderType::Pointer imageReader = ImageReaderType::New();
            imageReader -> SetFileName( inputs[i].c_str() );
            imageReader -> Update();
            images[i] = imageReader -> GetOutput();
        }
        std::cout << std::endl;

        //Extract max intensity in all images
        double maxIntensity = 0.0;
       std::vector<  double >  maxIntensities(numberOfImages);
        unsigned int imageIndexWithMax = 0;
        for (unsigned int i=0; i<numberOfImages; i++)
        {
            std::cout << "Process image # " << int2str(i) << std::endl;
            //Compute mean intensity in the whole image (Might not be mandatory as it seems we use only normalized intensity differences)
            StatisticsImageFilterType::Pointer statsImage = StatisticsImageFilterType::New();
            statsImage -> SetInput( images[i] );
            statsImage -> Update();

            maxIntensities[i] = statsImage -> GetMaximum();

            if(statsImage -> GetMaximum() > maxIntensity )
            {
                imageIndexWithMax = i;
                maxIntensity = statsImage -> GetMaximum();
                std::cout << "Global max updated to "<< maxIntensity << std::endl << std::endl;
            }
            else
            {
                std::cout << "Global max not updated" << std::endl << std::endl;
            }
        }

        for (unsigned int i=0; i<numberOfImages; i++)
        {
            //if(i != imageIndexWithMax)
            //{

                double newMax = ( maxIntensities[i] / maxIntensity ) * 255.0;
                std::cout << "Image # "<< int2str(i) <<" : Max set to "<< newMax << std::endl;
                itk::RescaleIntensityImageFilter<ImageType,ImageType>::Pointer rescaler = itk::RescaleIntensityImageFilter<ImageType,ImageType>::New();
                rescaler -> SetInput( images[i] );
                rescaler -> SetOutputMinimum( 0.0 );
                rescaler -> SetOutputMaximum(newMax);
                rescaler -> Update();

                outImages[i] = rescaler -> GetOutput();
           // }
           // else
           //{
           //     outImages[i] = images[i];
           //}
        }

        for (unsigned int i=0; i<numberOfImages; i++)
        {
            ImageWriterType::Pointer writer = ImageWriterType::New();
            writer -> SetFileName( outputs[i].c_str() );
            writer -> SetInput(outImages[i].GetPointer());
            writer -> Update();
        }

        return EXIT_SUCCESS;

    } catch (TCLAP::ArgException &e)  // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }
}

