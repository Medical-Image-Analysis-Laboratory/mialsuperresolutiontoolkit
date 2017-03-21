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


// ITK includes
#include "itkResampleImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
//#include "itkCropImageFilter.h"
#include "itkExtractImageFilter.h"

// Local includes
#include "btkImageHelper.h"


namespace btk
{

template< class TImage,class TMask >
CropImageUsingMaskFilter< TImage,TMask >::CropImageUsingMaskFilter()
{
    // ----
    Self::SetNumberOfRequiredInputs(0);
}

//------------------------------------------------------------------------------------------------

template< class TImage,class TMask >
CropImageUsingMaskFilter< TImage,TMask >::~CropImageUsingMaskFilter()
{
    // ----
}

//------------------------------------------------------------------------------------------------

template< class TImage,class TMask >
void CropImageUsingMaskFilter< TImage,TMask >::SetMask(typename TMask::Pointer maskImage)
{
    m_Mask = maskImage;
}

//------------------------------------------------------------------------------------------------

template< class TImage,class TMask >
void CropImageUsingMaskFilter< TImage,TMask >::GenerateData()
{
    // Set the number of outputs images
    Self::SetNumberOfIndexedOutputs(m_Images.size());

    // Crop images
    for(int i = 0; i < Self::GetNumberOfIndexedOutputs(); i++)
    {
        // Resample mask image on current input image
        typedef itk::ResampleImageFilter< TMask,TMask > ResampleImageFilter;
        typename ResampleImageFilter::Pointer resampleFilter = ResampleImageFilter::New();

        resampleFilter->SetOutputOrigin(m_Images[i]->GetOrigin());
        resampleFilter->SetOutputSpacing(m_Images[i]->GetSpacing());
        resampleFilter->SetOutputDirection(m_Images[i]->GetDirection());
        resampleFilter->SetSize(m_Images[i]->GetLargestPossibleRegion().GetSize());
        resampleFilter->SetInput(m_Mask);
        resampleFilter->SetInterpolator(itk::NearestNeighborInterpolateImageFunction< TMask >::New());
        resampleFilter->Update();

        // Create mask object
        typename Self::Mask::Pointer mask = Self::Mask::New();
        mask->SetImage(resampleFilter->GetOutput());

        // Define extract filter with bounding box of the mask
        typename itk::ExtractImageFilter< TImage,TImage >::Pointer extractFilter = itk::ExtractImageFilter< TImage,TImage >::New();
        extractFilter->SetExtractionRegion(mask->GetAxisAlignedBoundingBoxRegion());
        extractFilter->SetInput(m_Images[i]);
        extractFilter->Update();

        Self::SetNthOutput(i, extractFilter->GetOutput());
    }
}

//------------------------------------------------------------------------------------------------

template< class TImage,class TMask >
void CropImageUsingMaskFilter< TImage,TMask >::SetInputs(const std::vector< typename TImage::Pointer > &inputs)
{
    // Exceptions
    if(inputs.empty())
    {
        btkException("CropImageUsingMaskFilter: There should be at least one image to crop !");
    }


    m_Images = inputs;
}

//------------------------------------------------------------------------------------------------

template< class TImage,class TMask >
std::vector< typename TImage::Pointer > CropImageUsingMaskFilter< TImage,TMask >::GetOutputs()
{
    std::vector< typename TImage::Pointer > outputs;

    for(int i = 0; i < Self::GetNumberOfIndexedOutputs(); i++)
    {
        outputs.push_back(Self::GetOutput(i));
    }

    return outputs;
}

} // namespace btk
