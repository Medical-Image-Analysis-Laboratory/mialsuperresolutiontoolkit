/*=========================================================================

Copyright (c) 2017 Medical Image Analysis Laboratory (MIAL), Lausanne
  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
==========================================================================*/

#ifndef _mialsrtkLowToHighImageResolutionMethod_txx
#define _mialsrtkLowToHighImageResolutionMethod_txx

#include "mialsrtkLowToHighImageResolutionMethod.h"

namespace mialsrtk
{

/*
 * Constructor
 */
template < typename ImageType, typename TransformType >
LowToHighImageResolutionMethod<ImageType, TransformType >
::LowToHighImageResolutionMethod()
{
  m_TransformArray.resize(0);
  m_InverseTransformArray.resize(0);
  m_Interpolator = 0;
  m_NumberOfImages = 0;
  m_TargetImage = 0;
  m_InitializeWithMask= true;
  m_Margin = 0.0;
  m_Iterations = 200;
  m_UseReference = false;
}

template < typename ImageType, typename TransformType >
void
LowToHighImageResolutionMethod<ImageType, TransformType >
::SetNumberOfImages(int N)
{

  m_TransformArray.resize(N);
  m_InverseTransformArray.resize(N);
  m_ImageArray.resize(N);
  m_ResampledImageArray.resize(N);
  m_RegionArray.resize(N);
  m_ImageMaskArray.resize(N);

  for(int i=m_NumberOfImages; i<N; i++)
  {
    m_ImageArray[i]=0;
    m_ResampledImageArray[i]=0;
  }

  m_NumberOfImages  = N;

}
/*
 * Initialize by setting the interconnects between components.
 */
template < typename ImageType, typename TransformType >
void
LowToHighImageResolutionMethod<ImageType, TransformType >
::Initialize() throw (ExceptionObject)
{
    // if we don't use a reference we use a LR image as a reference (set by int TargetImage)
    if(!m_UseReference)
    {
        m_ReferenceImage = m_ImageArray[m_TargetImage];
        m_ReferenceRegion = m_RegionArray[m_TargetImage];
        m_ReferenceMask = m_ImageMaskArray[m_TargetImage];
    }
    else
    {

         if((m_ReferenceImage.GetPointer()) == NULL
                 || (m_ReferenceMask.GetPointer()) == NULL)
         {
            btkWarningMacro("You must Set a Reference Image, mask and Region");

            btkException("ReferenceImage, Mask or region is not Set !");

         }




    }

  /* Create interpolator */
  m_Interpolator = InterpolatorType::New();

  /* Resampling matrix */
  SpacingType  fixedSpacing   = m_ReferenceImage->GetSpacing();
  SizeType     fixedSize      = m_ReferenceImage->GetLargestPossibleRegion().GetSize();
  PointType    fixedOrigin    = m_ReferenceImage->GetOrigin();

  //Get slice-select direction
  int sliceSelectDirectionIndex = 2;
  float sliceThickness = 0.0; 
  for(int i = 0; i < 3; i++)
  {
    if(fixedSpacing[i] > sliceThickness)
    {
      sliceThickness = fixedSpacing[i];
      sliceSelectDirectionIndex = i;
    }
  }

  // Combine masks to redefine the resampling region
  std::cout << "Combines masks to refine the resampling region..." << std::endl;

  IndexType indexMin;
  IndexType indexMax;
  IndexType index;
  IndexType targetIndex;
  PointType point;

  indexMin = m_ReferenceImage-> GetLargestPossibleRegion().GetIndex();

  indexMax[0] = fixedSize[0]-1;
  indexMax[1] = fixedSize[1]-1;
  indexMax[2] = fixedSize[2]-1;

  for (unsigned int i=0; i<m_NumberOfImages; i++)
  {
    if (i != m_TargetImage || m_UseReference == true)
    {
      IteratorType regionIt(m_ImageArray[i],m_RegionArray[i]);
      for(regionIt.GoToBegin(); !regionIt.IsAtEnd(); ++regionIt )
      {

        index = regionIt.GetIndex();
        m_ImageArray[i] -> TransformIndexToPhysicalPoint(index,point);
        m_ReferenceImage -> TransformPhysicalPointToIndex(point,targetIndex);

        for (unsigned int k=0; k<3; k++)
        {
          if (targetIndex[k]<indexMin[k])
            indexMin[k]=targetIndex[k];
          else
            if (targetIndex[k]>indexMax[k])
              indexMax[k]=targetIndex[k];
        }
      }
    }
  }

  for(int i = 0; i < 3; i++)
  {
    if(i != sliceSelectDirectionIndex)
    {
      m_ResampleSpacing[i] = fixedSpacing[i];
      m_ResampleSpacing[sliceSelectDirectionIndex] = fixedSpacing[i];
      m_ResampleSize[i] = floor((indexMax[i]-indexMin[i]+1)*fixedSpacing[i]/m_ResampleSpacing[i] + 0.5);
    }
    else
    {
      m_ResampleSize[i] = floor(((indexMax[i]-indexMin[i]+1)*fixedSpacing[i] + 2*m_Margin)/m_ResampleSpacing[i] + 0.5);
    }
  }

  //m_ResampleSpacing[0] = fixedSpacing[0];
  //m_ResampleSpacing[1] = fixedSpacing[0];
  //m_ResampleSpacing[2] = fixedSpacing[0];

  //m_ResampleSize[0] = floor((indexMax[0]-indexMin[0]+1)*fixedSpacing[0]/m_ResampleSpacing[0] + 0.5);
  //m_ResampleSize[1] = floor((indexMax[1]-indexMin[1]+1)*fixedSpacing[1]/m_ResampleSpacing[1] + 0.5);
  //m_ResampleSize[2] = floor(((indexMax[2]-indexMin[2]+1)*fixedSpacing[2] + 2*m_Margin)/m_ResampleSpacing[2] + 0.5);

  /* Create high resolution image */
  std::cout << "Creates the HR image... ";

  m_HighResolutionImage = ImageType::New();

  IndexType start;
  start[0] = 0;
  start[1] = 0;
  start[2] = 0;

  RegionType region;
  region.SetIndex(start);
  region.SetSize(m_ResampleSize);

  m_HighResolutionImage -> SetRegions( region );
  m_HighResolutionImage -> Allocate();
  m_HighResolutionImage -> SetOrigin( fixedOrigin );
  m_HighResolutionImage -> SetSpacing( m_ResampleSpacing );
  m_HighResolutionImage -> SetDirection( m_ReferenceImage-> GetDirection() );
  m_HighResolutionImage -> FillBuffer( 0 );

  IndexType newOriginIndex;
  PointType newOrigin;

   for(int i = 0; i < 3; i++)
  {
    if(i != sliceSelectDirectionIndex)
    {
      newOriginIndex[i] = floor(indexMin[i]*fixedSpacing[i]/m_ResampleSpacing[i] + 0.5);
    }
    else
    {
      newOriginIndex[i] = floor(indexMin[i]*fixedSpacing[i]/m_ResampleSpacing[i] + 0.5) - floor( m_Margin / m_ResampleSpacing[i] + 0.5);
    }
  }
      

  //newOriginIndex[0] = floor(indexMin[0]*fixedSpacing[0]/m_ResampleSpacing[0] + 0.5);
  //newOriginIndex[1] = floor(indexMin[1]*fixedSpacing[1]/m_ResampleSpacing[1] + 0.5);
  //newOriginIndex[2] = floor(indexMin[2]*fixedSpacing[2]/m_ResampleSpacing[2] + 0.5) - floor( m_Margin / m_ResampleSpacing[2] + 0.5);

  m_HighResolutionImage -> TransformIndexToPhysicalPoint(newOriginIndex, newOrigin);
  m_HighResolutionImage -> SetOrigin( newOrigin );

  std::cout << "done." << std::endl;

}

/*
 * Starts the Registration Process
 */
template < typename ImageType, typename TransformType >
void
LowToHighImageResolutionMethod<ImageType, TransformType >
::StartRegistration( void )
{

  try
  {
    // initialize the interconnects between components
    this->Initialize();
  }
  catch( ExceptionObject& err )
  {
    // pass exception to caller
    throw err;
  }

  //typename RegistrationBase::Pointer registration;
  typename RegistrationType::Pointer registration;
  typename TransformType::Pointer myTransform = TransformType::New();
  if(std::string(myTransform->GetNameOfClass()) == "VersorRigid3DTransform")
  {
     registration = RegistrationType::New();
  }
  //else if(std::string(myTransform->GetNameOfClass()) == "AffineTransform")
  //{
  //   registration = AffineRegistrationType::New();
  //}
  else
  {
      throw(std::string("Wrong Type of Transform . Only VersorRigid3DTransform can be used !"));
  }

  registration -> SetFixedImage(  m_ReferenceImage  );
  registration -> SetFixedImageRegion( m_ReferenceRegion );

  //std::cout << "Ref image " << std::endl << m_ReferenceImage;
  //std::cout << "Ref region " << std::endl << m_ReferenceRegion;


  if ( m_InitializeWithMask )
  {
    registration -> SetFixedImageMask( m_ReferenceMask );
    registration -> InitializeWithMask();
  }

  registration -> SetEnableObserver( false );
  registration -> SetIterations( m_Iterations );

  m_ResamplingStatus.resize( m_NumberOfImages );
  for (unsigned int i=0; i < m_ResamplingStatus.size(); i++)
    m_ResamplingStatus[i] = false;

  for (unsigned int i=0; i < m_NumberOfImages; i++)
  {

      registration->SetMovingImage( m_ImageArray[i] );
      registration->SetMovingImageMask( m_ImageMaskArray[i] );

      //std::cout << "Moving image " << std::endl << m_ImageArray[i];
      //std::cout << "Moving region " << std::endl << m_ImageMaskArray[i];

      //registration->SetInitialTransformParameters( m_InitialRigidParameters[i] );

      if (i != m_TargetImage || m_UseReference == true )
      {
          std::cout << "Runs registration... ";
          try
          {              
              registration->Update();
          }
          catch( itk::ExceptionObject & err )
          {
              std::cerr << "ExceptionObject caught !" << std::endl;
              std::cerr << err << std::endl;
              throw err;
          }
          std::cout << "done." << std::endl;

          m_TransformArray[i] = reinterpret_cast<TransformType*>(registration->GetTransform());

//          if(std::string(m_TransformArray[0]->GetNameOfClass()) == "Euler3DTransform")
//          {
//              m_TransformArray[i] = dynamic_cast< RegistrationType* >(registration)->GetTransform();
//          }
//          else if(std::string(m_TransformArray[0]->GetNameOfClass()) == "AffineTransform")
//          {
//             m_TransformArray[i] = dynamic_cast< AffineRegistrationType* >(registration)->GetTransform();;
//          }
//          else
//          {
//              throw(std::string("Wrong Type of Transform . Only Euler3DTransform and AffineTransform can be used !"));
//          }


      }
      else
      {
          m_TransformArray[i] = TransformType::New();
          m_TransformArray[i] -> SetIdentity();
          //m_TransformArray[i] -> SetCenter( registration -> GetTransformCenter() );
      }

  }

  std::cout<<"Inverses the Transforms... "<<std::endl;

  // Calculate inverse transform array
  for (unsigned int i=0; i < m_NumberOfImages; i++)
  {
    m_InverseTransformArray[i] = TransformType::New();
    m_InverseTransformArray[i] -> SetIdentity();
    m_InverseTransformArray[i] -> SetCenter( m_TransformArray[i] -> GetCenter() );
    m_InverseTransformArray[i] -> SetParameters( m_TransformArray[i] -> GetParameters() );

    //FIXME : When using AffineTransform VNL matrix produce error :
    //00 suspicious return value (3) from SVDC
    // When SetParameters is comment the error disapear !
    // No error when using Euler3DTransform...

    m_InverseTransformArray[i]->SetMatrix(m_TransformArray[i]->GetMatrix());
    m_InverseTransformArray[i] -> GetInverse( m_InverseTransformArray[i] );
    // NOTE : use this GetInverse
     //m_TransformArray[i] -> GetInverse( m_InverseTransformArray[i] );


    //std::cout<<"Inverse transform "<<i<<" : "<<m_InverseTransformArray[i]<<std::endl;
  }

  // Create combination of masks
  PointType physicalPoint;
  PointType transformedPoint;
  IndexType index;

  std::cout<<"Creates combination of masks... "<<std::endl;

  if ( m_InitializeWithMask )
  {
    m_ImageMaskCombination = ImageMaskType::New();
    m_ImageMaskCombination -> SetRegions( m_HighResolutionImage -> GetLargestPossibleRegion() );
    m_ImageMaskCombination -> Allocate();
    m_ImageMaskCombination -> SetSpacing(   m_HighResolutionImage -> GetSpacing() );
    m_ImageMaskCombination -> SetDirection( m_HighResolutionImage -> GetDirection() );
    m_ImageMaskCombination -> SetOrigin(    m_HighResolutionImage -> GetOrigin() );
    m_ImageMaskCombination -> FillBuffer( 0 );

    IteratorType imageIt( m_HighResolutionImage, m_HighResolutionImage -> GetLargestPossibleRegion() );
    ImageMaskIteratorType maskIt( m_ImageMaskCombination, m_ImageMaskCombination -> GetLargestPossibleRegion() );

    ImageMaskInterpolatorPointer imageMaskInterpolator = ImageMaskInterpolatorType::New();

    for(imageIt.GoToBegin(),maskIt.GoToBegin(); !imageIt.IsAtEnd(); ++imageIt, ++maskIt )
    {
      index = imageIt.GetIndex();
      m_ImageMaskCombination -> TransformIndexToPhysicalPoint( index, physicalPoint );

      for (unsigned int i=0; i < m_NumberOfImages; i++)
      {
        transformedPoint = m_TransformArray[i] -> TransformPoint(physicalPoint);
        imageMaskInterpolator -> SetInputImage( m_ImageMaskArray[i] );

        if ( imageMaskInterpolator -> IsInsideBuffer(transformedPoint) )
        {
          if ( imageMaskInterpolator -> Evaluate(transformedPoint) > 0 )
          {
            maskIt.Set(1);
            continue;
          }
        }
      }
    }

  }

  //std::cout << "m_ImageMaskCombination : " << m_ImageMaskCombination << std::endl;
  //std::cout << "m_ImageMaskCombination region: " << m_ImageMaskCombination->GetLargestPossibleRegion() << std::endl;

  // Average registered images

  //std::cout<<"Average the registered images : "<<std::endl;

  int value;
  unsigned int counter;

  //std::cout<< "m_HRImage : " << m_HighResolutionImage << std::endl;
  //std::cout<< "m_HRImage Region : " << m_HighResolutionImage->GetLargestPossibleRegion() << std::endl;
  IteratorType imageIt( m_HighResolutionImage, m_HighResolutionImage -> GetLargestPossibleRegion() );

  for (imageIt.GoToBegin(); !imageIt.IsAtEnd(); ++imageIt )
  {
    index = imageIt.GetIndex();

    m_HighResolutionImage -> TransformIndexToPhysicalPoint( index, physicalPoint);

    value = 0;
    counter = 0;

    for (unsigned int i=0; i < m_NumberOfImages; i++)
    {
      m_Interpolator -> SetInputImage( m_ImageArray[i] );
      transformedPoint = m_TransformArray[i]->TransformPoint(physicalPoint);
      if ( m_Interpolator -> IsInsideBuffer (transformedPoint) )
      {
        value+= m_Interpolator->Evaluate(transformedPoint);
        counter++;
      }
    }
    if ( counter>0 ) imageIt.Set(value/counter);

    if ( m_InitializeWithMask && ( m_ImageMaskCombination -> GetPixel(index) == 0 ) )
    {
      imageIt.Set(0);
    }

  }

  std::cout << "Average image created!!!" << std::endl;

}

/*
 * Writes transforms to file
 */
template < typename ImageType, typename TransformType >
void
LowToHighImageResolutionMethod<ImageType, TransformType >
::WriteTransforms( const char *folder )
{

  typename TransformWriterType::Pointer transformWriter = TransformWriterType::New();

  for (unsigned int i=0; i < m_NumberOfImages; i++)
  {

    transformWriter->SetInput( m_TransformArray[i] );

    char fullTrafoName[255]; strcpy ( fullTrafoName, folder );
    char trafoName[255];

    sprintf ( trafoName, "/%d.txt", i );
    strcat ( fullTrafoName,trafoName );

    transformWriter -> SetFileName ( fullTrafoName );

    try
    {
      std::cout << "Writing transform to " << fullTrafoName << " ... " ; std::cout.flush();
      transformWriter -> Update();
      std::cout << " done! " << std::endl;
    }
    catch ( itk::ExceptionObject & excp )
    {
      std::cerr << "Error while saving the transforms" << std::endl;
      std::cerr << excp << std::endl;
      std::cout << "[FAILED]" << std::endl;
      throw excp;
    }

  }

}

/*
 * Writes a specific transform to file
 */
template < typename ImageType, typename TransformType >
void
LowToHighImageResolutionMethod<ImageType, TransformType >
::WriteTransforms( unsigned int i, const char *filename )
{

  typename TransformWriterType::Pointer transformWriter = TransformWriterType::New();
  transformWriter->SetInput( m_TransformArray[i] );
  transformWriter -> SetFileName ( filename );

  try
  {
    std::cout << "Writing transform to " << filename << " ... " ; std::cout.flush();
    transformWriter -> Update();
    std::cout << " done! " << std::endl;
  }
  catch ( itk::ExceptionObject & excp )
  {
    std::cerr << "Error while saving the transforms" << std::endl;
    std::cerr << excp << std::endl;
    std::cout << "[FAILED]" << std::endl;
    throw excp;
  }

}

/*
 * Writes resampled images to disk
 */
template < typename ImageType, typename TransformType >
void
LowToHighImageResolutionMethod<ImageType, TransformType >
::WriteResampledImages( const char *folder )
{

  typename ImageWriterType::Pointer imageWriter = ImageWriterType::New();

  for (unsigned int i=0; i < m_NumberOfImages; i++)
  {
    if (!m_ResamplingStatus[i])
    {
      m_Resample =  ResampleType::New();
      m_Resample -> SetTransform( m_TransformArray[i] );
      m_Resample -> SetInput( m_ImageArray[i] );
      m_Resample -> SetReferenceImage( m_HighResolutionImage );
      m_Resample -> SetUseReferenceImage( true );
      m_Resample -> SetDefaultPixelValue( 0 );
      m_Resample -> Update();
      m_ResampledImageArray[i] = m_Resample -> GetOutput();
      m_ResamplingStatus[i] = true;
    }

    imageWriter->SetInput( m_ResampledImageArray[i] );

    char fullImageName[255]; strcpy ( fullImageName, folder );
    char imageName[255];

    sprintf ( imageName, "/%d2HR.nii.gz", i );
    strcat ( fullImageName, imageName );

    imageWriter -> SetFileName ( fullImageName );

    try
    {
      std::cout << "Writing resampled image to " << fullImageName << " ... " ; std::cout.flush();
      imageWriter -> Update();
      std::cout << " done! " << std::endl;
    }
    catch ( itk::ExceptionObject & excp )
    {
      std::cerr << "Error while saving the resampled image" << std::endl;
      std::cerr << excp << std::endl;
      std::cout << "[FAILED]" << std::endl;
      throw excp;
    }

  }

}

/*
 * Writes a specific resampled image to disk
 */
template < typename ImageType, typename TransformType >
void
LowToHighImageResolutionMethod<ImageType, TransformType >
::WriteResampledImages( unsigned int i, const char *filename )
{

  if (!m_ResamplingStatus[i])
  {
    m_Resample =  ResampleType::New();
    m_Resample -> SetTransform( m_TransformArray[i] );
    m_Resample -> SetInput( m_ImageArray[i] );
    m_Resample -> SetReferenceImage( m_HighResolutionImage );
    m_Resample -> SetUseReferenceImage( true );
    m_Resample -> SetDefaultPixelValue( 0 );
    m_Resample -> Update();
    m_ResampledImageArray[i] = m_Resample -> GetOutput();
    m_ResamplingStatus[i] = true;
  }

  typename ImageWriterType::Pointer imageWriter = ImageWriterType::New();

  imageWriter -> SetInput( m_ResampledImageArray[i] );
  imageWriter -> SetFileName ( filename );

  try
  {
    std::cout << "Writing resampled image to " << filename << " ... " ; std::cout.flush();
    imageWriter -> Update();
    std::cout << " done! " << std::endl;
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
template < typename ImageType, typename TransformType >
void
LowToHighImageResolutionMethod<ImageType, TransformType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}


} // end namespace mialsrtk


#endif
