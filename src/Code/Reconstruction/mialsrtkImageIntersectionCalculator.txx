/*=========================================================================

Copyright (c) 2017 Medical Image Analysis Laboratory (MIAL), Lausanne
  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
==========================================================================*/

#ifndef _mialsrtkImageIntersectionCalculator_cxx
#define _mialsrtkImageIntersectionCalculator_cxx

#include "mialsrtkImageIntersectionCalculator.h"

namespace mialsrtk
{

/*
 * Constructor
 */
template < typename ImageType >
ImageIntersectionCalculator<ImageType>
::ImageIntersectionCalculator()
{

  m_NumberOfImages = 0;

}

template < typename ImageType >
void ImageIntersectionCalculator<ImageType>
::AddImage( ImageType * image)
{
  unsigned int n = m_NumberOfImages;

  m_ImageArray.resize(n+1);
  m_ImageArray[n] = image;

  m_ImageMaskArray.resize(n+1);
  m_ImageMaskArray[n] = ImageMaskType::New();
  m_ImageMaskArray[n] -> SetRegions( m_ImageArray[n] -> GetLargestPossibleRegion() );
  m_ImageMaskArray[n] -> Allocate();

  m_ImageMaskArray[n] -> SetOrigin( m_ImageArray[n] -> GetOrigin() );
  m_ImageMaskArray[n] -> SetSpacing( m_ImageArray[n] -> GetSpacing() );
  m_ImageMaskArray[n] -> SetDirection( m_ImageArray[n] -> GetDirection() );
  m_ImageMaskArray[n] -> FillBuffer( 0 );

  m_MaskArray.resize(n+1);
  m_MaskArray[n] = MaskType::New();

  m_RegionArray.resize(n+1);

  m_InterpolatorArray.resize(n+1);
  m_InterpolatorArray[n] = InterpolatorType::New();
  m_InterpolatorArray[n] -> SetInputImage( image );

  m_NumberOfImages++;

}

/*
 * Writes a specific transform to file
 */
template < typename ImageType >
void
ImageIntersectionCalculator<ImageType>
::Update()
{
  IndexType index;
  PointType point;

  for (unsigned int i=0; i < m_NumberOfImages; i++)
  {
    IteratorType imageMaskIt( m_ImageMaskArray[i],
                         m_ImageMaskArray[i] -> GetLargestPossibleRegion() );

    // creates image mask
    for(imageMaskIt.GoToBegin(); !imageMaskIt.IsAtEnd(); ++imageMaskIt )
    {
      imageMaskIt.Set(1);
      index = imageMaskIt.GetIndex();
      m_ImageMaskArray[i] -> TransformIndexToPhysicalPoint(index, point);
      for (unsigned int j=0; j < m_NumberOfImages; j++)
      {
        if ( !(m_InterpolatorArray[j] -> IsInsideBuffer(point)) )
        {
          imageMaskIt.Set(0);
          break;
        }
      }
    }

    m_MaskArray[i] -> SetImage( m_ImageMaskArray[i] );
    m_RegionArray[i] = m_MaskArray[i] -> GetAxisAlignedBoundingBoxRegion();

  }
}

/*
 * Writes a specific mask to disk
 */
template < typename ImageType >
void
ImageIntersectionCalculator<ImageType>
::WriteMask( unsigned int i, const char *filename )
{

  typename ImageWriterType::Pointer imageWriter = ImageWriterType::New();

  imageWriter -> SetInput( m_ImageMaskArray[i] );
  imageWriter -> SetFileName ( filename );

  try
  {
    imageWriter -> Update();
  }
  catch ( itk::ExceptionObject & excp )
  {
    std::cerr << "Error while saving the resampled image" << std::endl;
    std::cerr << excp << std::endl;
    std::cout << "[FAILED]" << std::endl;
    throw excp;
  }

}


/*
 * PrintSelf
 */
template < typename ImageType >
void
ImageIntersectionCalculator<ImageType>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}


} // end namespace mialsrtk


#endif
