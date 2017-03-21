/*==========================================================================

  © Université de Strasbourg - Centre National de la Recherche Scientifique

  Date: 11/04/2012
  Author(s): Marc Schweitzer (marc.schweitzer(at)unistra.fr)
             Julien Pontabry (pontabry@unistra.fr)

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


#ifndef BTK_IMAGE_HELPER_H
#define BTK_IMAGE_HELPER_H

// STL includes
#include "string"
#include "iostream"
#include "fstream"
#include "sstream"


// ITK includes
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkNumericTraits.h"
#include "itkVector.h"
#include "itkCastImageFilter.h"

namespace btk
{
    /**
     * @class ImageHelper
     * @brief Helper class for image management (read, write and create operations)
     * @author Marc Schweitzer, Julien Pontabry
     * @ingroup Tools
     */
    template < class TImageInput, class TImageOutput=TImageInput >
    class ImageHelper
    {
        public:

            /**
             * @brief Image reader type.
             */
            typedef itk::ImageFileReader< TImageInput > ImageReader;

            /**
             * @brief Image writer type.
             */
            typedef itk::ImageFileWriter< TImageInput > ImageWriter;

            typedef itk::CastImageFilter< TImageInput, TImageOutput> CastImageFilter;


            /**
             * @brief Write an image.
             * @param image Image to write.
             * @param fileName File name of the image to write.
             */
            static void WriteImage(typename TImageInput::Pointer image, const std::string &fileName);

            /**
             * @brief Write a vector of images.
             * @param images vector of images to write.
             * @param fileNames File names of the images to write.
             */
            static void WriteImage(std::vector< typename TImageInput::Pointer > &images, std::vector< std::string > &fileNames);

            /**
             * @brief Read an image.
             * @param fileName File name of the image to read.
             * @return A pointer to the image that have been read.
             */
            static typename TImageInput::Pointer ReadImage(const std::string &fileName);

            /**
             * @brief Read an image and return an constant pointer.
             * @param fileName File name of the image to read.
             * @return A constant pointer to the image that have been read.
             */
            static typename TImageInput::ConstPointer ReadConstImage(const std::string &fileName);

            /**
             * @brief Read a vector of images.
             * @param fileNames File names of the images to read.
             * @return A reference to a vector containing the images that have been read.
             */
            static std::vector< typename TImageInput::Pointer > &ReadImage(std::vector< std::string> &fileNames);

            /**
             * @brief Create a new image in the same physical space of a current image.
             * @param image Image of which physical space will be used for creation.
             * @param defaultValue Default value of pixel in new image.
             * @return New image in the same physical space.
             */
            static typename TImageOutput::Pointer CreateNewImageFromPhysicalSpaceOf(typename TImageInput::Pointer image, typename TImageOutput::PixelType defaultValue=itk::NumericTraits< typename TImageOutput::PixelType >::ZeroValue());

            /**
             * @brief Create a new image in the same physical space of a current image.
             * @param image Image of which physical space will be used for creation.
             * @param defaultValue Default value of pixel in new image.
             * @return New image in the same physical space.
             */
            static typename TImageOutput::Pointer CreateNewImageFromPhysicalSpaceOfConst(typename TImageInput::ConstPointer image, typename TImageOutput::PixelType defaultValue=itk::NumericTraits< typename TImageOutput::PixelType >::ZeroValue());

            /**
             * @brief Create new images in the same physical space of current images.
             * @param images Vector of images of which physical space will be used for creation.
             * @param defaultValue Default value of pixel in new image.
             * @return Vector of new images in the same physical space.
             */
            static std::vector< typename TImageOutput::Pointer > &CreateNewImageFromPhysicalSpaceOfConst(std::vector< typename TImageInput::ConstPointer > &images, typename TImageOutput::PixelType defaultValue=itk::NumericTraits< typename TImageOutput::PixelType >::ZeroValue());

            /**
             * @brief Create new images in the same physical space of current images.
             * @param images Vector of images of which physical space will be used for creation.
             * @param defaultValue Default value of pixel in new image.
             * @return Vector of new images in the same physical space.
             */
            static std::vector< typename TImageOutput::Pointer > &CreateNewImageFromPhysicalSpaceOf(std::vector< typename TImageInput::Pointer > &images, typename TImageOutput::PixelType defaultValue=itk::NumericTraits< typename TImageOutput::PixelType >::ZeroValue());

            /**
             * @brief Test if images are in the same physical space.
             * @param firstImage First image.
             * @param secondImage Second Image.
             * @return True if the two images are in the same physical space, false otherwise.
             */
            static bool IsInSamePhysicalSpace(typename TImageInput::Pointer firstImage, typename TImageOutput::Pointer secondImage, double epsilon = 10e-7);

            /**
             * @brief Test if images are in the same physical space.
             * @param images Images to compare.
             * @return True if the two images are in the same physical space, false otherwise.
             */

            static bool IsInSamePhysicalSpace(const std::vector< typename TImageInput::Pointer > &images, double epsilon = 10e-7);
            
             /**
             * @brief Read an image if the file exists, otherwise create an image with a constant pixel value.
             * @param fileName File name of the image to read.
             * @param image Image of which physical space will be used for creation.
             * @param pixel value used for initialisation in case of image creation.
             * @return A pointer to the image that have been red.
             */
            static typename TImageOutput::Pointer ReadOrCreateImage(const std::string &fileName, typename TImageInput::Pointer image, typename TImageOutput::PixelType defaultValue=itk::NumericTraits< typename TImageOutput::PixelType >::ZeroValue());

            /**
             * @brief Make a deep copy of an image.
             * @param image Constant pointer on the image to copy.
             * @return A deep copy of the image parameter.
             */
            static typename TImageOutput::Pointer DeepCopy(typename TImageInput::ConstPointer image);

            /**
             * @brief Check if two images are orthogonals
             * @param two image pointers
             * @return A boolean, true if they are orthogonal, false if not.
             */

            static bool AreOrthos(typename TImageInput::Pointer image1, typename TImageInput::Pointer image2, float threshold = 0.5);

            /**
             * @brief Cast image in another type (TInputImage into TOutputImage)
             * @param An Image Pointer (TInputImage)
             * @return A Pointer to a TOutputImage.
             *
             */

            static typename TImageOutput::Pointer CastImage(typename TImageInput::Pointer image);

            /**
             * @brief Cast a vector of images in another type (TInputImage into TOutputImage)
             * @param A vector of Image Pointer (TInputImage)
             * @return A vector of TOutputImage::Pointer.
             *
             */

            static  std::vector< typename TImageOutput::Pointer > &CastImage(std::vector< typename TImageInput::Pointer > &images);
    };

} // namespace btk

#include "btkImageHelper.txx"

#endif // BTK_IMAGE_HELPER_H
