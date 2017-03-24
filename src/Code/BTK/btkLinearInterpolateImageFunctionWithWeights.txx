/*==========================================================================

  © Université de Strasbourg - Centre National de la Recherche Scientifique

  Date: 02/12/2010
  Author(s): Estanislao Oubel (oubel@unistra.fr)

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

#ifndef __btkLinearInterpolateImageFunctionWithWeights_txx
#define __btkLinearInterpolateImageFunctionWithWeights_txx

// First, make sure that we include the configuration file.
// This line may be removed once the ThreadSafeTransform gets
// integrated into ITK.
#include "itkConfigure.h"
#include "btkLinearInterpolateImageFunctionWithWeights.h"
#include "vnl/vnl_math.h"

namespace btk
{

/**
 * Define the number of neighbors
 */
template<class TInputImage, class TCoordRep>
const unsigned long
LinearInterpolateImageFunctionWithWeights< TInputImage, TCoordRep >
::m_Neighbors = 1 << TInputImage::ImageDimension;


/**
 * Constructor
 */
template<class TInputImage, class TCoordRep>
LinearInterpolateImageFunctionWithWeights< TInputImage, TCoordRep >
::LinearInterpolateImageFunctionWithWeights()
{

}


/**
 * PrintSelf
 */
template<class TInputImage, class TCoordRep>
void
LinearInterpolateImageFunctionWithWeights< TInputImage, TCoordRep >
::PrintSelf(std::ostream& os, Indent indent) const
{
  this->Superclass::PrintSelf(os,indent);
}


/**
 * Evaluate at image index position
 */
template<class TInputImage, class TCoordRep>
typename LinearInterpolateImageFunctionWithWeights< TInputImage, TCoordRep >
::OutputType
LinearInterpolateImageFunctionWithWeights< TInputImage, TCoordRep >
::EvaluateAtContinuousIndex(
  const ContinuousIndexType& index) const
{
  m_Indexes.resize(0);
  m_Overlaps.resize(0);


  unsigned int dim;  // index over dimension

  /**
   * Compute base index = closet index below point
   * Compute distance from point to base index
   */
  signed long baseIndex[ImageDimension];
  double distance[ImageDimension];
  long tIndex;

  for( dim = 0; dim < ImageDimension; dim++ )
    {
    // The following "if" block is equivalent to the following line without
    // having to call floor.
    //    baseIndex[dim] = (long) vcl_floor(index[dim] );
    if (index[dim] >= 0.0)
      {
      baseIndex[dim] = (long) index[dim];
      }
    else
      {
      tIndex = (long) index[dim];
      if (double(tIndex) != index[dim])
        {
        tIndex--;
        }
      baseIndex[dim] = tIndex;
      }
    distance[dim] = index[dim] - double( baseIndex[dim] );
    }

  /**
   * Interpolated value is the weighted sum of each of the surrounding
   * neighbors. The weight for each neighbor is the fraction overlap
   * of the neighbor pixel with respect to a pixel centered on point.
   */
  RealType value = NumericTraits<RealType>::Zero;

  typedef typename NumericTraits<InputPixelType>::ScalarRealType ScalarRealType;
  ScalarRealType totalOverlap = NumericTraits<ScalarRealType>::Zero;

  for( unsigned int counter = 0; counter < m_Neighbors; counter++ )
    {

    double overlap = 1.0;          // fraction overlap
    unsigned int upper = counter;  // each bit indicates upper/lower neighbour
    IndexType neighIndex;

    // get neighbor index and overlap fraction
    for( dim = 0; dim < ImageDimension; dim++ )
      {

      if ( upper & 1 )
        {
        neighIndex[dim] = baseIndex[dim] + 1;
#ifdef ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY
        // Take care of the case where the pixel is just
        // in the outer upper boundary of the image grid.
        if( neighIndex[dim] > this->m_EndIndex[dim] )
          {
          neighIndex[dim] = this->m_EndIndex[dim];
          }
#endif
        overlap *= distance[dim];
        }
      else
        {
        neighIndex[dim] = baseIndex[dim];
#ifdef ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY
        // Take care of the case where the pixel is just
        // in the outer lower boundary of the image grid.
        if( neighIndex[dim] < this->m_StartIndex[dim] )
          {
          neighIndex[dim] = this->m_StartIndex[dim];
          }
#endif
        overlap *= 1.0 - distance[dim];
        }

      upper >>= 1;

      }

    // get neighbor value only if overlap is not zero
    if( overlap )
      {
      value += static_cast<RealType>( this->GetInputImage()->GetPixel( neighIndex ) ) * overlap;
      totalOverlap += overlap;

      m_Indexes.push_back( neighIndex );
      m_Overlaps.push_back( overlap );

      }

    if( totalOverlap == 1.0 )
      {
      // finished
      break;
      }

    }

  return ( static_cast<OutputType>( value ) );
}

} // end namespace btk

#endif
