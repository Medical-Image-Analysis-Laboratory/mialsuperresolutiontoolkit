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

#ifndef BTK_WEIGHTED_SUM_OF_IMAGES_FILTER_H
#define BTK_WEIGHTED_SUM_OF_IMAGES_FILTER_H

// STL includes
#include "vector"

// ITK includes
#include "itkSmartPointer.h"
#include "itkImageToImageFilter.h"
#include "itkImageMaskSpatialObject.h"


namespace btk
{
/**
 *@class WeightedSumOfImagesFilter
 *@brief This class do a weighted sum of images
 *@author Julien Pontabry
 *@ingroup ImageFilter
 */
template< class TImage,typename TPrecision=float >
class WeightedSumOfImagesFilter : public itk::ImageToImageFilter< TImage,TImage >
{
    public:
        typedef WeightedSumOfImagesFilter                Self;
        typedef itk::ImageToImageFilter< TImage,TImage > Superclass;
        typedef itk::SmartPointer< Self >                Pointer;

        itkNewMacro(Self);
        itkTypeMacro(WeightedSumOfImagesFilter,ImageToImageFilter);


        /**
         * @brief Set input images.
         * @param inputs Vector of input images.
         */
        void SetInputs(const std::vector< typename TImage::Pointer > &inputs);

        /**
         * @brief Set weights corresponding to input images.
         * @param weights Vector weights.
         */
        void SetWeights(const std::vector< TPrecision > &weights);


    protected:
        /**
         * @brief Constructor.
         */
        WeightedSumOfImagesFilter();

        /**
         * @brief Destructor.
         */
        ~WeightedSumOfImagesFilter();

        /**
         * @brief Generate data.
         */
        virtual void GenerateData();

    private:
        /**
         * @brief Weights of input images.
         */
        std::vector< TPrecision > m_Weights;
};

} // namespace btk

#include "btkWeightedSumOfImagesFilter.txx"

#endif // BTK_WEIGHTED_SUM_OF_IMAGES_FILTER_H
