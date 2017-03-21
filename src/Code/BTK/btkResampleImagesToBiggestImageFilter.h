/*==========================================================================

  © Université de Strasbourg - Centre National de la Recherche Scientifique

  Date: 15/10/2012
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

#ifndef BTK_RESAMPLE_IMAGES_TO_BIGGEST_IMAGE_FILTER_H
#define BTK_RESAMPLE_IMAGES_TO_BIGGEST_IMAGE_FILTER_H

// STL includes
#include "vector"

// ITK includes
#include "itkSmartPointer.h"
#include "itkImageToImageFilter.h"
#include "itkInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"


namespace btk
{
/**
 *@class ResampleImagesToBiggestImageFilter
 *@brief Resample images int space of the bigest one
 *@author Julien Pontabry
 *@ingroup ImageFilter
 */
template< class TImage >
class ResampleImagesToBiggestImageFilter : public itk::ImageToImageFilter< TImage,TImage >
{
    public:
        typedef ResampleImagesToBiggestImageFilter       Self;
        typedef itk::ImageToImageFilter< TImage,TImage > Superclass;
        typedef itk::SmartPointer< Self >                Pointer;

        typedef itk::ResampleImageFilter< TImage,TImage > ResampleImageFilter;
        typedef itk::InterpolateImageFunction< TImage >   Interpolator;

        itkNewMacro(Self);
        itkTypeMacro(ResampleImagesToBiggestImageFilter,ImageToImageFilter);

        btkSetMacro(Interpolator, Interpolator *);


        /**
         * @brief Set input images.
         * @param inputs Vector of input images.
         */
        void SetInputs(const std::vector< typename TImage::Pointer > &inputs);

        /**
         * @brief Get output images.
         * @return Vector of resampled images.
         */
        std::vector< typename TImage::Pointer > GetOutputs();


    protected:
        /**
         * @brief Constructor.
         */
        ResampleImagesToBiggestImageFilter();

        /**
         * @brief Destructor.
         */
        ~ResampleImagesToBiggestImageFilter();

        /**
         * @brief Generate data.
         */
        virtual void GenerateData();

    private:
        /**
         * @brief Structure for storing input images.
         */
        std::vector< typename TImage::Pointer > m_InputImages; // Using this structure is needed since ITK may produce output images with a bad physical header (I do not know why...).

        /**
         * @brief Structure for storing output images.
         */
        std::vector< typename TImage::Pointer > m_OutputImages;

        /**
         * @brief Interpolator to use when resampling images.
         */
        typename Interpolator::Pointer m_Interpolator;
};

} // namespace btk

#include "btkResampleImagesToBiggestImageFilter.txx"

#endif // BTK_RESAMPLE_IMAGE_TO_BIGGEST_IMAGE_FILTER_H
