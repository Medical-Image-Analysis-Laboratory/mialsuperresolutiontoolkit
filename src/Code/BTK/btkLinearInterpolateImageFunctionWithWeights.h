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

#ifndef __btkLinearInterpolateImageFunctionWithWeights_h
#define __btkLinearInterpolateImageFunctionWithWeights_h

// First make sure that the configuration is available.
// This line can be removed once the optimized versions
// gets integrated into the main directories.
#include "itkConfigure.h"

#include "itkInterpolateImageFunction.h"

namespace btk
{

using namespace itk;

/** \class LinearInterpolateImageFunction
 * \brief Linearly interpolate an image at specified positions.
 *
 * LinearInterpolateImageFunction linearly interpolates image intensity at
 * a non-integer pixel position. This class is templated
 * over the input image type and the coordinate representation type
 * (e.g. float or double).
 *
 * This function works for N-dimensional images.
 *
 * \warning This function work only for images with scalar pixel
 * types. For vector images use VectorLinearInterpolateImageFunction.
 *
 * \sa VectorLinearInterpolateImageFunction
 *
 * \ingroup ImageFunctions ImageInterpolators
 */
template <class TInputImage, class TCoordRep = double>
class LinearInterpolateImageFunctionWithWeights :
  public InterpolateImageFunction<TInputImage,TCoordRep>
{
public:
  /** Standard class typedefs. */
  typedef LinearInterpolateImageFunctionWithWeights                  Self;
  typedef InterpolateImageFunction<TInputImage,TCoordRep> Superclass;
  typedef SmartPointer<Self>                              Pointer;
  typedef SmartPointer<const Self>                        ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(LinearInterpolateImageFunctionWithWeights, InterpolateImageFunction);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** OutputType typedef support. */
  typedef typename Superclass::OutputType OutputType;

  /** InputImageType typedef support. */
  typedef typename Superclass::InputImageType InputImageType;

  /** InputPixelType typedef support. */
  typedef typename Superclass::InputPixelType InputPixelType;

  /** RealType typedef support. */
  typedef typename Superclass::RealType RealType;

  /** Dimension underlying input image. */
  itkStaticConstMacro(ImageDimension, unsigned int,Superclass::ImageDimension);

  /** Index typedef support. */
  typedef typename Superclass::IndexType IndexType;

  /** ContinuousIndex typedef support. */
  typedef typename Superclass::ContinuousIndexType ContinuousIndexType;

  /** Evaluate the function at a ContinuousIndex position
   *
   * Returns the linearly interpolated image intensity at a
   * specified point position. No bounds checking is done.
   * The point is assume to lie within the image buffer.
   *
   * ImageFunction::IsInsideBuffer() can be used to check bounds before
   * calling the method. */
  virtual OutputType EvaluateAtContinuousIndex(
    const ContinuousIndexType & index ) const;

  unsigned int GetContributingNeighbors()
  {
    return m_Indexes.size();
  }

  IndexType GetIndex(unsigned int i)
  {
    return m_Indexes[i];
  }

  double GetOverlap(unsigned int i)
  {
    return m_Overlaps[i];
  }


protected:
  LinearInterpolateImageFunctionWithWeights();
  ~LinearInterpolateImageFunctionWithWeights(){};
  void PrintSelf(std::ostream& os, Indent indent) const;

private:
  LinearInterpolateImageFunctionWithWeights( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented

  /** Number of neighbors used in the interpolation */
  static const unsigned long  m_Neighbors;

  /** Indexes of neighbors used in the interpolation */
  mutable std::vector< IndexType > m_Indexes;

  /** Overlap of neighbors used in the interpolation */
  mutable std::vector< double > m_Overlaps;

};

} // end namespace btk

// Define instantiation macro for this template.
#define ITK_TEMPLATE_LinearInterpolateImageFunctionWithWeights(_, EXPORT, x, y) namespace itk { \
  _(2(class EXPORT LinearInterpolateImageFunctionWithWeights< ITK_TEMPLATE_2 x >)) \
  namespace Templates { typedef LinearInterpolateImageFunctionWithWeights< ITK_TEMPLATE_2 x > \
                                                  LinearInterpolateImageFunctionWithWeights##y; } \
  }

#if ITK_TEMPLATE_EXPLICIT
# include "Templates/LinearInterpolateImageFunctionWithWeights+-.h"
#endif

#if ITK_TEMPLATE_TXX
# include "btkLinearInterpolateImageFunctionWithWeights.txx"
#endif

#endif
