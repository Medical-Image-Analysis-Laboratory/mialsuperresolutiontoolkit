/*=========================================================================

Copyright (c) 2017 Medical Image Analysis Laboratory (MIAL), Lausanne
  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
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
