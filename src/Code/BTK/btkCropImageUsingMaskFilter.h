/*==========================================================================
  
  © Université de Strasbourg - Centre National de la Recherche Scientifique
  
  Date: 12/10/2012
  Author(s): Julien Pontabry (pontabry@unistra.fr)
  
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

#ifndef BTK_CROP_IMAGE_USING_MASK_FILTER_H
#define BTK_CROP_IMAGE_USING_MASK_FILTER_H

// ITK includes
#include "itkSmartPointer.h"
#include "itkImageToImageFilter.h"
#include "itkImageMaskSpatialObject.h"


namespace btk
{

template< class TImage,class TMask >
class CropImageUsingMaskFilter : public itk::ImageToImageFilter< TImage,TImage >
{
    public:
        typedef CropImageUsingMaskFilter                 Self;
        typedef itk::ImageToImageFilter< TImage,TImage > Superclass;
        typedef itk::SmartPointer< Self >                Pointer;

        typedef itk::ImageMaskSpatialObject< TImage::ImageDimension > Mask;

        itkNewMacro(Self);
        itkTypeMacro(CropImageUsingMaskFilter,ImageToImageFilter);

        /**
         * @brief Set input mask to filter.
         * @param mask Pointer on mask image.
         */
        void SetMask(typename TMask::Pointer maskImage);

        /**
         * @brief Set input images.
         * @param inputs Vector of input images.
         */
        void SetInputs(const std::vector< typename TImage::Pointer > &inputs);

        /**
         * @brief Get output images of the filter.
         * @return A vector of pointer to output images.
         */
        std::vector< typename TImage::Pointer > GetOutputs();


    protected:
        /**
         * @brief Constructor.
         */
        CropImageUsingMaskFilter();

        /**
         * @brief Destructor.
         */
        ~CropImageUsingMaskFilter();

        /**
         * @brief Generate data.
         */
        virtual void GenerateData();


    private:
        /**
         * @brief Structure for storing images.
         */
        std::vector< typename TImage::Pointer > m_Images; // Using this structure is needed since ITK may produce output images with a bad physical header (I do not know why...).

        /**
         * @brief Mask image for cropping purposes.
         */
        typename TMask::Pointer m_Mask;
};

} // namespace btk

#include "btkCropImageUsingMaskFilter.txx"

#endif // BTK_CROP_IMAGE_USING_MASK_FILTER_H
