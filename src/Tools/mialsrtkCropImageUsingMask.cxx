/*==========================================================================

  © Université de Strasbourg - Centre National de la Recherche Scientifique

  Date: 26/06/2012
  Author(s): Marc Schweitzer (marc.schweitzer@unistra.fr)
             modified by Frederic CHAMP (champ@unistra.fr)

  This software is governed by the CeCILL-B license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL-B
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited
  liability.

  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.

  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL-B license and that you accept its terms.

==========================================================================*/

/* ITK */
#include "itkImage.h"
#include "itkMaskImageFilter.h"
#include "itkImageIOBase.h"
#include "itkExtractImageFilter.h"
#include "itkJoinSeriesImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkDisplacementFieldTransform.h"

/* BTK */
#include "btkImageHelper.h"
/*
#include "btkDiffusionSequence.h"
#include "btkDiffusionSequenceHelper.h"
#include "btkDiffusionSequenceFileHelper.h"
*/
#include "btkIOImageHelper.h"
#include "btkResampleImagesToBiggestImageFilter.h"
#include "btkWeightedSumOfImagesFilter.h"
#include "btkCropImageUsingMaskFilter.h"
#include "btkGradientDirection.h"

/* OTHERS */
#include "iostream"
#include <tclap/CmdLine.h>


// Definitions
//typedef itk::ExtractImageFilter< btk::DiffusionSequence,itk::Image< short,3 > > ExtractImageFilter;
//typedef itk::JoinSeriesImageFilter< itk::Image< short,3 >,btk::DiffusionSequence> JoinSeriesFilterType;



template< class TImage,class TMask >
void Process(std::vector< std::string > &inputFileNames, std::vector< std::string > &outputFileNames, std::vector< std::string > &maskFileNames, const int Dimension)
{
    // Read Masks files

    std::vector< typename TMask::Pointer >   masks = btk::ImageHelper< TMask >::ReadImage(maskFileNames);

    // Compute the cropping mask by using the union of all masks
    typename TMask::Pointer croppingMask = NULL;

    if(masks.size() > 1)
    {
        // First, resample masks if needed
        if(!btk::ImageHelper< TMask >::IsInSamePhysicalSpace(masks))
        {
            btkCoutMacro("Resampling masks in same physical space... ");

            typename btk::ResampleImagesToBiggestImageFilter< TMask >::Pointer resampleFilter = btk::ResampleImagesToBiggestImageFilter< TMask >::New();
            resampleFilter->SetInputs(masks);
            resampleFilter->SetInterpolator(itk::NearestNeighborInterpolateImageFunction< TMask >::New());
            resampleFilter->Update();
            masks = resampleFilter->GetOutputs();

            btkCoutMacro("done.");
        }

        // Combine the masks
        btkCoutMacro("Combining masks... ");

        std::vector< float > weights(masks.size(), 1.0f);

        typename btk::WeightedSumOfImagesFilter< TMask >::Pointer sumFilter = btk::WeightedSumOfImagesFilter< TMask >::New();
        sumFilter->SetInputs(masks);
        sumFilter->SetWeights(weights);
        sumFilter->Update();

        croppingMask = sumFilter->GetOutput();

        btkCoutMacro("done.");
    }
    else // masks.size() == 1
    {
        croppingMask = masks[0];
    }


    switch(Dimension)
    {
        //cropping for 3D images
        case 3:
        {
            std::vector< typename TImage::Pointer > inputs = btk::ImageHelper< TImage >::ReadImage(inputFileNames);

            btkCoutMacro("Cropping image... ");
            typename btk::CropImageUsingMaskFilter< TImage,TMask >::Pointer cropFilter = btk::CropImageUsingMaskFilter< TImage,TMask >::New();
            cropFilter->SetMask(croppingMask);
            cropFilter->SetInputs(inputs);
            cropFilter->Update();
            std::vector< typename TImage::Pointer > outputs = cropFilter->GetOutputs();
            btk::ImageHelper< TImage >::WriteImage(outputs, outputFileNames);
            btkCoutMacro("done.");
        }
        break;

        //cropping for 4D images
        /*
        case 4:
        {
            std::vector< btk::DiffusionSequence::Pointer > inputs = btk::DiffusionSequenceHelper::ReadSequenceArray(inputFileNames);

            ExtractImageFilter::Pointer extractor = ExtractImageFilter::New();
            JoinSeriesFilterType::Pointer outputs  = JoinSeriesFilterType::New();

            btk::DiffusionSequence::RegionType  region4D = inputs[0]->GetLargestPossibleRegion();
            btk::DiffusionSequence::SizeType input4DSize = region4D.GetSize();
            btk::DiffusionSequence::SizeType input3DSize;
            input3DSize[0] = input4DSize[0];
            input3DSize[1] = input4DSize[1];
            input3DSize[2] = input4DSize[2];
            input3DSize[3] = 0;

            outputs->SetOrigin( inputs[0]->GetOrigin()[3] );
            outputs->SetSpacing(inputs[0]->GetSpacing()[3] );

            extractor->SetInput(inputs[0]); //Maybe need to be change...

            btk::DiffusionSequence::IndexType start = region4D.GetIndex();
            uint numberOf3Dimages = input4DSize[3];

            btkCoutMacro("Cropping images... ");
            for (uint i = 0; i < numberOf3Dimages; i++)
            {

                start[3] = i;

                std::vector< itk::Image<short,3>::Pointer > *vector3DImage = new std::vector< itk::Image<short,3>::Pointer>;

                btk::DiffusionSequence::RegionType region3D;
                region3D.SetSize( input3DSize );
                region3D.SetIndex( start );

                extractor->SetExtractionRegion( region3D );
                extractor->SetDirectionCollapseToSubmatrix();
                extractor->Update();

                vector3DImage->push_back(extractor->GetOutput());

                std::cout<<"\r-> Cropping image "<<i<<" ... "<<std::flush;
                typename btk::CropImageUsingMaskFilter< itk::Image<short,3>,TMask >::Pointer cropFilter = btk::CropImageUsingMaskFilter< itk::Image<short,3>,TMask >::New();
                cropFilter->SetMask(croppingMask);
                cropFilter->SetInputs(*vector3DImage);
                cropFilter->Update();

                outputs->PushBackInput (cropFilter->GetOutput());


            }
            btkCoutMacro("done.");

            btk::ImageHelper<  btk::DiffusionSequence>::WriteImage(outputs->GetOutput(), outputFileNames[0]);

            std::vector< unsigned short > bValues = inputs[0]->GetBValues();
            std::string bValuesName = btk::FileHelper::GetRadixOf(outputFileNames[0])+".bval";
            btk::DiffusionSequenceFileHelper::WriteBValues(bValues,bValuesName);


            std::vector< btk::GradientDirection > GradientTable  = inputs[0]->GetGradientTable();
            std::string GradientTableName = btk::FileHelper::GetRadixOf(outputFileNames[0])+".bvec";
            btk::DiffusionSequenceFileHelper::WriteGradientTable(GradientTable,GradientTableName);


        }
        break;*/

        default:
            std::cout<<"Only dimension  3 or 4 are supported\n";

    }

}

int main(int argc, char * argv[])
{
    try
    {
        //
        // Command line parser
        //

        TCLAP::CmdLine cmd("btkCropImageUsingMask: Crop an image (3D - anatomical, or 4D - diffusion weighted) using a 3D mask (non-zero values). Input image and mask image must have the same pixel type.", ' ', "2.0", true);

        TCLAP::MultiArg< std::string > inputFileNamesArg("i", "input", "Input image file", true, "string", cmd);
        TCLAP::MultiArg< std::string > maskFileNamesArg("m", "mask", "Mask image file", true, "string", cmd);
        TCLAP::MultiArg< std::string > outputFileNamesArg("o", "output", "Output image file", true, "string", cmd);

        // Parse the args.
        cmd.parse( argc, argv );

        std::vector< std::string > inputFileNames = inputFileNamesArg.getValue();
        std::vector< std::string > maskFileNames = maskFileNamesArg.getValue();
        std::vector< std::string > outputFileNames = outputFileNamesArg.getValue();

        //
        // Processing
        //

        // Check arguments
        if(inputFileNames.size() != maskFileNames.size() || inputFileNames.size() != outputFileNames.size())
        {
            btkException("The number of inputs/masks/outputs is not coherent ! There should be a mask and an output for each input !");
        }

        // Check image informations
        itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(inputFileNames[0].c_str(), itk::ImageIOFactory::ReadMode);
        imageIO->SetFileName(inputFileNames[0]);
        imageIO->ReadImageInformation();

        int Dimension = imageIO->GetNumberOfDimensions();
        btkCoutMacro("Dimension of input image :" << Dimension);

        switch(imageIO->GetPixelType())
        {
            case itk::ImageIOBase::VECTOR:
            {
                btkCoutMacro("Pixel type: Vector");

                typedef itk::DisplacementFieldTransform< double,3 >::DisplacementFieldType DisplacementField;
                Process< DisplacementField,itk::Image< unsigned char,3 > >(inputFileNames, outputFileNames, maskFileNames, Dimension);
            }
                break;

            case itk::ImageIOBase::SCALAR:
            {
                btkCoutMacro("Pixel type: Scalar");

                switch(imageIO->GetComponentType())
                {
                    case itk::ImageIOBase::UCHAR:
                        btkCoutMacro("Scalar type: Unsigned Char");
                        Process< itk::Image< unsigned char,3 >,itk::Image< unsigned char,3 > >(inputFileNames, outputFileNames, maskFileNames, Dimension);
                        break;

                    case itk::ImageIOBase::CHAR:
                        btkCoutMacro("Scalar type: Char");
                        Process< itk::Image< char,3 >,itk::Image< unsigned char,3 > >(inputFileNames, outputFileNames, maskFileNames, Dimension);
                        break;

                    case itk::ImageIOBase::SHORT:
                        btkCoutMacro("Scalar type: Short");
                        Process< itk::Image< short,3 >,itk::Image< unsigned char,3 > >(inputFileNames, outputFileNames, maskFileNames, Dimension);
                        break;

                    case itk::ImageIOBase::USHORT:
                        btkCoutMacro("Scalar type: Unsigned Short");
                        Process< itk::Image< unsigned short,3 >,itk::Image< unsigned char,3 > >(inputFileNames, outputFileNames, maskFileNames,Dimension);
                        break;

                    case itk::ImageIOBase::FLOAT:
                        btkCoutMacro("Scalar type: Float");
                        Process< itk::Image< float,3 >,itk::Image< unsigned char,3 > >(inputFileNames, outputFileNames, maskFileNames,Dimension);
                        break;

                    case itk::ImageIOBase::DOUBLE:
                        btkCoutMacro("Scalar type: Double");
                        Process< itk::Image< double,3 >,itk::Image< unsigned char,3 > >(inputFileNames, outputFileNames, maskFileNames,Dimension);
                        break;

                    default:
                        btkException("Unsupported component type (only short, float or double types are supported) !");
                }
            }
                break;

            default:
                btkException("Unsupported pixel type (only scalar and vector are supported) !");
        }
    }
    catch(TCLAP::ArgException &e)  // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }
    catch(std::string &e)  // catch any exceptions
    {
        std::cerr << "error: " << e <<std::endl;
    }
    catch(itk::ExceptionObject &e)
    {
        btkCerrMacro(e.GetDescription());
    }

    return EXIT_SUCCESS;
}
