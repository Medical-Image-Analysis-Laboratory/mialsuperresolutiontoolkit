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

#include "btkResampleImagesToBiggestImageFilter.h"


// ITK includes
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

// BTK includes
#include "btkMacro.h"
#include "btkImageHelper.h"


namespace btk
{

template< class TImage >
ResampleImagesToBiggestImageFilter< TImage >::ResampleImagesToBiggestImageFilter() : Superclass()
{
    Self::SetNumberOfRequiredInputs(0);
    Self::SetNumberOfRequiredOutputs(0);

    m_Interpolator = Self::ResampleImageFilter::LinearInterpolatorType::New();
}

//------------------------------------------------------------------------------------------------

template< class TImage >
ResampleImagesToBiggestImageFilter< TImage >::~ResampleImagesToBiggestImageFilter()
{
    // ----
}

//------------------------------------------------------------------------------------------------

template< class TImage >
void ResampleImagesToBiggestImageFilter< TImage >::GenerateData()
{
    // Search the bounding box of all of the images.
    typename TImage::PointType minPoint, maxPoint;
    typename TImage::IndexType index;
    typename TImage::SizeType size = m_InputImages[0]->GetLargestPossibleRegion().GetSize();

    index[0] = 0; index[1] = 0; index[2] = 0;
    m_InputImages[0]->TransformIndexToPhysicalPoint(index, minPoint);

    index[0] = size[0]-1; index[1] = size[1]-1; index[2] = size[2]-1;
    m_InputImages[0]->TransformIndexToPhysicalPoint(index, maxPoint);

    for(int i = 1; i < m_InputImages.size(); i++)
    {
        typename TImage::PointType point;

        index[0] = 0; index[1] = 0; index[2] = 0;
        m_InputImages[i]->TransformIndexToPhysicalPoint(index, point);

        for(int d=0; d < 3; d++)
        {
          if(point[d] < minPoint[d])
          {
              minPoint[d] = point[d];
          }

          if(point[d] > maxPoint[d])
          {
              maxPoint[d] = point[d];
          }
        }

        size = m_InputImages[i]->GetLargestPossibleRegion().GetSize();
        index[0] = size[0]-1; index[1] = size[1]-1; index[2] = size[2]-1;
        m_InputImages[i]->TransformIndexToPhysicalPoint(index, point);

        for(int d=0; d < 3; d++)
        {
          if(point[d] < minPoint[d])
          {
              minPoint[d] = point[d];
          }

          if(point[d] > maxPoint[d])
          {
              maxPoint[d] = point[d];
          }
        }
    }


    std::cout<<"min point : "<<minPoint<<std::endl;
    std::cout<<"max point : "<<maxPoint<<std::endl;

    // Compute the reference layer information
    typename TImage::SpacingType newspacing = m_InputImages[0]->GetSpacing();
    for(int i = 1; i < m_InputImages.size(); i++)
    {
      typename TImage::SpacingType currentspacing = m_InputImages[i]->GetSpacing();

      for(int d=0; d < 3; d++)
      {
        if(currentspacing[d] < newspacing[d] )
          newspacing[d] = currentspacing[d];
      }
    }

    std::cout<<"Spacing of the output image:"<<newspacing[0]<<" "<<newspacing[1]<<" "<<newspacing[2]<<std::endl;


    typename TImage::SizeType newsize;
    newsize[0] = std::abs(maxPoint[0]-minPoint[0])/newspacing[0]; newsize[1] = std::abs(maxPoint[1]-minPoint[1])/newspacing[1]; newsize[2] = std::abs(maxPoint[2]-minPoint[2])/newspacing[2];

    std::cout<<"Size of the output image:"<<newsize[0]<<" "<<newsize[1]<<" "<<newsize[2]<<std::endl;

    typename TImage::PointType neworigin = minPoint;

    typename TImage::DirectionType newdirection;
    newdirection.SetIdentity();

    // Create the reference image
    typename TImage::Pointer reference = TImage::New();
    reference->SetRegions(newsize);
    reference->SetOrigin(neworigin);
    reference->SetSpacing(newspacing);
    reference->SetDirection(newdirection);
    reference->Allocate();


    // Resample all input images and set them as output.
    for(int i = 0; i < m_InputImages.size(); i++)
    {
        // Define a resampler with the reference image with the maximal volume.
        typename Self::ResampleImageFilter::Pointer filter = ResampleImageFilter::New();

        filter->SetReferenceImage(reference);
        filter->UseReferenceImageOn();
        filter->SetInterpolator(m_Interpolator);
        filter->SetInput(m_InputImages[i]);
        filter->Update();
        m_OutputImages.push_back(filter->GetOutput());
    }
}

//------------------------------------------------------------------------------------------------

template< class TImage >
void ResampleImagesToBiggestImageFilter< TImage >::SetInputs(const std::vector< typename TImage::Pointer > &inputs)
{
    // Exceptions
    if(inputs.size() < 2)
    {
        btkException("ResampleImagesToBiggestImageFilter: There is not enough input images (it should be at least 2) !");
    }


    m_InputImages = inputs;
}

//------------------------------------------------------------------------------------------------

template< class TImage >
std::vector< typename TImage::Pointer > ResampleImagesToBiggestImageFilter< TImage >::GetOutputs()
{
    return m_OutputImages;
}

} // namespace btk
