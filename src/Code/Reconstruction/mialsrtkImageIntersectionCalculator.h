/*==========================================================================

  © Université de Strasbourg - Centre National de la Recherche Scientifique

  Date: 14/09/2010
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

#ifndef __mialsrtkImageIntersectionCalculator_h
#define __mialsrtkImageIntersectionCalculator_h

#include "itkObject.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkImageMaskSpatialObject.h"

#include "itkImageFileWriter.h"

#include "itkNumericTraits.h"

namespace mialsrtk
{

/** @class ImageIntersectionCalculator
 * @brief Perform a bounding box mask, with intersection of orthogonal image
 * @author Estanislao Oubel
 * @ingroup Reconstruction
 */

template <typename TImage>
class ImageIntersectionCalculator : public itk::Object
{
public:
  /** Standard class typedefs. */
  typedef ImageIntersectionCalculator  Self;
  typedef itk::Object                                       Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageIntersectionCalculator, Object);

  /**  Type of the Fixed image. */
  typedef          TImage                               ImageType;
  typedef typename ImageType::Pointer                   ImagePointer;
  typedef          std::vector<ImagePointer>            ImageArrayPointer;

  typedef itk::Image< unsigned char,
                 ImageType::ImageDimension >            ImageMaskType;
  typedef typename ImageMaskType::Pointer               ImageMaskPointer;
  typedef          std::vector<ImageMaskPointer>        ImageMaskPointerArray;

  typedef itk::ImageMaskSpatialObject< ImageType::ImageDimension > MaskType;
  typedef typename MaskType::Pointer                          MaskPointer;
  typedef          std::vector<MaskPointer>                   MaskPointerArray;

  typedef typename ImageType::RegionType                ImageRegionType;
  typedef          std::vector< ImageRegionType >       ImageRegionArray;

  typedef itk::LinearInterpolateImageFunction< ImageType,
                                          double>       InterpolatorType;
  typedef typename InterpolatorType::Pointer            InterpolatorPointer;
  typedef          std::vector<InterpolatorPointer>     InterpolatorPointerArray;

  typedef typename ImageType::PointType                 PointType;
  typedef typename ImageType::IndexType               	IndexType;

  typedef typename ImageType::RegionType               	RegionType;
  typedef  std::vector<RegionType>                      RegionArray;

  typedef itk::ImageFileWriter< ImageMaskType >  ImageWriterType;

  typedef itk::ImageRegionIteratorWithIndex< ImageMaskType >  IteratorType;

  /** Write a specific mask to disk. */
  void WriteMask( unsigned int i, const char *filename );

  /** Get a specific bounding box. */
  RegionType GetBoundingBoxRegion( unsigned int i)
  {
    return m_RegionArray[i];
  }

  ImageMaskType* GetImageMask( unsigned int i)
  {
    return m_ImageMaskArray[i];
  }


  /** Add an image for intersection calculation */
  void AddImage( ImageType * image);

  /** Calculates the intersection */
  void Update();

protected:
  ImageIntersectionCalculator();
  virtual ~ImageIntersectionCalculator() {};
  void PrintSelf(std::ostream& os, itk::Indent indent) const;


private:
  ImageIntersectionCalculator(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  ImageArrayPointer 			         m_ImageArray;
  ImageMaskPointerArray            m_ImageMaskArray;
  MaskPointerArray                 m_MaskArray;
  InterpolatorPointerArray         m_InterpolatorArray;
  ImageRegionArray                 m_RegionArray;

  unsigned int                     m_NumberOfImages;

};


} // end namespace mialsrtk


#ifndef ITK_MANUAL_INSTANTIATION
#include "mialsrtkImageIntersectionCalculator.txx"
#endif

#endif
