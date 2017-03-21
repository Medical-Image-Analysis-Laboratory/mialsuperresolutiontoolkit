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

#include "btkWeightedSumOfImagesFilter.h"


// ITK includes
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

// BTK includes
#include "btkMacro.h"
#include "btkImageHelper.h"


namespace btk
{

template< class TImage,typename TPrecision >
WeightedSumOfImagesFilter< TImage,TPrecision >::WeightedSumOfImagesFilter() : Superclass()
{
    Self::SetNumberOfIndexedOutputs(1);
}

//------------------------------------------------------------------------------------------------

template< class TImage,typename TPrecision >
WeightedSumOfImagesFilter< TImage,TPrecision >::~WeightedSumOfImagesFilter()
{
    // ----
}

//------------------------------------------------------------------------------------------------

template< class TImage,typename TPrecision >
void WeightedSumOfImagesFilter< TImage,TPrecision >::GenerateData()
{
    // Exceptions
    if(m_Weights.size() != Self::GetNumberOfIndexedInputs())
    {
        btkException("WeightedSumOfImagesFilter: The number of weights is different than the number of input images !");
    }


    // Create new output image from the same physical space of input ones (and initialize it to 0).
    typename TImage::Pointer outputImage = btk::ImageHelper< TImage >::CreateNewImageFromPhysicalSpaceOfConst(Self::GetInput(0));

    // Define an iterator on output image.
    itk::ImageRegionIterator< TImage > outputIt(outputImage, outputImage->GetLargestPossibleRegion());

    // Compute weighted sum.
    for(int i = 0; i < Self::GetNumberOfIndexedInputs(); i++)
    {
        // Define an iterator on current input image.
        itk::ImageRegionConstIterator< TImage > inputIt(Self::GetInput(i), Self::GetInput(i)->GetLargestPossibleRegion());

        // Add the image to output
        for(inputIt.GoToBegin(), outputIt.GoToBegin(); !inputIt.IsAtEnd() && !outputIt.IsAtEnd(); ++inputIt, ++outputIt)
        {
            outputIt.Set( outputIt.Get() + (inputIt.Get() * m_Weights[i]) );
        } // for each voxel
    } // for each image

    // Set output image for ITK filter
    Self::SetNthOutput(0,outputImage);
}

//------------------------------------------------------------------------------------------------

template< class TImage,typename TPrecision >
void WeightedSumOfImagesFilter< TImage,TPrecision >::SetInputs(const std::vector< typename TImage::Pointer > &inputs)
{
    // Exceptions
    if(inputs.size() < 1)
    {
        btkException("WeightedSumOfImagesFilter: There is not enough input images (it should be at least 1) !");
    }

    if(!ImageHelper< TImage >::IsInSamePhysicalSpace(inputs))
    {
        btkException("WeightedSumOfImagesFilter: Input images are not in the same physical space !");
    }


    // Tells about number of input images.
    Self::SetNumberOfIndexedInputs(inputs.size());

    // Add each images as input of current filter.
    for(int i = 0; i < inputs.size(); i++)
    {
        Self::SetNthInput(i, inputs[i]);
    }
}

//------------------------------------------------------------------------------------------------

template< class TImage,typename TPrecision >
void WeightedSumOfImagesFilter< TImage,TPrecision >::SetWeights(const std::vector< TPrecision > &weights)
{
    // Set weights.
    m_Weights = weights;
}

} // namespace btk
