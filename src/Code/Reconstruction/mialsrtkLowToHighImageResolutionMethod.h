/*=========================================================================

Copyright (c) 2017 Medical Image Analysis Laboratory (MIAL), Lausanne
  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
==========================================================================*/

#ifndef __mialsrtkLowToHighImageResolutionMethod_h
#define __mialsrtkLowToHighImageResolutionMethod_h

#include "itkProcessObject.h"
#include "itkEuler3DTransform.h"
#include "itkVersorRigid3DTransform.h"
#include "itkResampleImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegistrationMethod.h"

#include "mialsrtkRigidRegistration.h"
//#include "btkAffineRegistration.h"

#include "itkImageFileWriter.h"
#include "itkTransformFileWriter.h"
#include "itkTransformFactory.h"

#include "itkNumericTraits.h"
#include "btkUserMacro.h"

namespace mialsrtk
{

using namespace itk;

/** @class LowToHighImageResolutionMethod
 * @brief Class for obtaining an isovoxel image from low resolution images.
 *
 * Class for obtaining an isovoxel image from different low resolution images
 * by applying affine registration and averaging. First step of the method
 * described in:
 *
 * Rousseau, F., Glenn, O.A., Iordanova, B., Rodriguez-Carranza, C.,
 * Vigneron, D.B., Barkovich, J.A., Studholme, C.: Registration-based approach
 * for reconstruction of high-resolution in utero fetal MR brain images. Acad
 * Radiol 13(9) (Sep 2006) 1072–1081

 * @author Estanislao Oubel, modified by Sebastien Tourbier
 * \ingroup Reconstruction
 */

// TODO Change the template to provide two types of images (Low and High resolution)

template <typename TImage,typename TTransform>
class LowToHighImageResolutionMethod : public ProcessObject
{
public:
  /** Standard class typedefs. */
  typedef LowToHighImageResolutionMethod  Self;
  typedef ProcessObject                                Superclass;
  typedef SmartPointer<Self>                           Pointer;
  typedef SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(LowToHighImageResolutionMethod, ProcessObject);

  /**  Type of the Fixed image. */
  typedef          TImage                               ImageType;
  typedef typename ImageType::Pointer                   ImagePointer;
  typedef          std::vector<ImagePointer>            ImageArrayPointer;

  typedef typename ImageType::RegionType                ImageRegionType;
  typedef typename ImageType::PixelType                 PixelType;
  typedef typename ImageType::PointType                 PointType;
  typedef typename ImageType::SpacingType               SpacingType;
  typedef typename ImageType::SizeType               		SizeType;
  typedef typename ImageType::IndexType               	IndexType;

  typedef typename ImageType::RegionType               	RegionType;
  typedef  std::vector<RegionType>                      RegionArray;

  typedef itk::Image< unsigned char, 3 >                ImageMaskType;
  typedef typename ImageMaskType::Pointer								ImageMaskPointer;
  typedef std::vector<ImageMaskType::Pointer>           ImageMaskArray;

  /**  Type of the image writer. */
  typedef ImageFileWriter< ImageType >  ImageWriterType;

  /**  Type of the image iterator. */
  typedef ImageRegionIteratorWithIndex< ImageType >  IteratorType;

  /**  Type of the image mask iterator. */
  typedef ImageRegionIteratorWithIndex< ImageMaskType >  ImageMaskIteratorType;

  /**  Type of the Transform . */
  typedef TTransform                                   TransformType;
  //typedef MatrixOffsetTransformBase< double, 3 >                TransformType;
  //typedef AffineTransform<double,3>                             TransformType;
  //typedef Euler3DTransform< double >             				TransformType;
  typedef typename TransformType::Pointer               TransformPointer;
  typedef  std::vector<TransformPointer>                TransformPointerArray;

  /**  Type of the Interpolator. */
  typedef LinearInterpolateImageFunction<
                                    ImageType,
                                    double>             InterpolatorType;
  typedef typename InterpolatorType::Pointer            InterpolatorPointer;

  /**  Type of the image mask interpolator. */
  typedef LinearInterpolateImageFunction<
                                    ImageMaskType,
                                    double>             ImageMaskInterpolatorType;
  typedef typename ImageMaskInterpolatorType::Pointer   ImageMaskInterpolatorPointer;

   /**  Type of the registration method. */
  typedef mialsrtk::Registration<ImageType>   RegistrationBase;
  typedef typename RegistrationBase::Pointer             RegistrationPointerBase;

  typedef mialsrtk::RigidRegistration<ImageType>  				 RegistrationType;
  //typedef itk::ImageRegistrationMethod<ImageType, ImageType> RegistrationType;
  typedef typename RegistrationType::Pointer     RegistrationPointer;

  //typedef btk::AffineRegistration< ImageType >          AffineRegistrationType;
  //typedef typename AffineRegistrationType::Pointer      AffineRegistrationPointer;



  typedef ResampleImageFilter< ImageType, ImageType >    ResampleType;
  typedef typename ResampleType::Pointer 								 ResamplePointer;


  typedef  typename TransformType::ParametersType    ParametersType;
  typedef  std::vector<ParametersType>      ParametersArrayType;

  typedef TransformFileWriter TransformWriterType;

  /** Method that initiates the registration. */
  void StartRegistration();

  /** Write the final transforms to disk. */
  void WriteTransforms( const char *folder );

  /** Write a specific transform to disk. */
  void WriteTransforms( unsigned int i, const char *filename );

  /** Write the resampled images to disk. */
  void WriteResampledImages( const char *folder );

  /** Write a specific resampled image to disk. */
  void WriteResampledImages( unsigned int i, const char *filename );

  /** Get the resampled image array. */
  UserGetObjectMacro( ResampledImageArray, ImageType );

  /** Get the high resolution Image. */
  itkGetObjectMacro( HighResolutionImage, ImageType );

  /** Set/Get the FixedImageRegion. */
  itkSetMacro( FixedImageRegion, ImageRegionType );
  itkGetMacro( FixedImageRegion, ImageRegionType );

  /** Set/Get the Initialization. */
  itkSetMacro( InitializeWithMask, bool );
  itkGetMacro( InitializeWithMask, bool );

  /** Set/Get the target image. */
  itkSetMacro( TargetImage, unsigned int );
  itkGetMacro( TargetImage, unsigned int );

  /** Set/Get the margin. */
  itkSetMacro( Margin, double );
  itkGetMacro( Margin, double );

  /** Set/Get the transform. */
  UserSetObjectMacro( TransformArray, TransformType );
  UserGetObjectMacro( TransformArray, TransformType );

  /** Set/Get the inverse transform. */
  UserSetObjectMacro( InverseTransformArray, TransformType );
  UserGetObjectMacro( InverseTransformArray, TransformType );

  /** Set/Get the image array. */
  UserSetObjectMacro( ImageArray, ImageType );
  UserGetObjectMacro( ImageArray, ImageType );

  /** Set/Get the image array. */
  UserSetObjectMacro( ImageMaskArray, ImageMaskType );
  UserGetObjectMacro( ImageMaskArray, ImageMaskType );

  /** Set/Get the image array. */
  UserSetMacro( RegionArray, RegionType );
//  UserGetMacro( RegionArray, RegionType );

  /** Get the image mask combination. */
  itkGetObjectMacro( ImageMaskCombination, ImageMaskType );

  /** Set/Get the number of iterations. */
  itkSetMacro( Iterations, unsigned int );
  itkGetMacro( Iterations, unsigned int );

  /** Set/Get the reference Image if used */
  itkSetMacro(ReferenceImage, ImagePointer);
  itkGetMacro(ReferenceImage, ImagePointer);

  /** Set/Get use of reference image, if enable you must Set a Reference Image, Region and mask */
  itkSetMacro( UseReference, bool );
  itkGetMacro( UseReference, bool );

  /** Set/Get reference region is used */
  itkSetMacro(ReferenceRegion, RegionType);
  itkGetMacro(ReferenceRegion, RegionType);

  /** Set/Get Reference mask if used */
  itkSetMacro(ReferenceMask, ImageMaskPointer);
  itkGetMacro(ReferenceMask, ImageMaskPointer);




  /** Set the number of images */
  void SetNumberOfImages(int N);

  /** Initialize by setting the interconnects between the components.*/
  void Initialize() throw (ExceptionObject);


protected:
  LowToHighImageResolutionMethod();
  virtual ~LowToHighImageResolutionMethod() {};
  void PrintSelf(std::ostream& os, Indent indent) const;


private:
  LowToHighImageResolutionMethod(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  TransformPointerArray            m_TransformArray;
  TransformPointerArray            m_InverseTransformArray;
  ImageArrayPointer 			         m_ImageArray;
  RegionArray                      m_RegionArray;

  RegionType                       m_ReferenceRegion;
  ImageMaskPointer                 m_ReferenceMask;
  ImagePointer										 m_ReferenceImage;

  ImageMaskArray                   m_ImageMaskArray;

  SpacingType											 m_ResampleSpacing;
  SizeType											   m_ResampleSize;
  ImageArrayPointer						 		 m_ResampledImageArray;
  ResamplePointer                  m_Resample;
  std::vector<bool>                m_ResamplingStatus;
  double                           m_Margin;

  InterpolatorPointer              m_Interpolator;

  ImagePointer										 m_HighResolutionImage;

  ImageMaskPointer					 			 m_ImageMaskCombination;

  ParametersType                   m_InitialTransformParameters;
  ParametersType                   m_LastTransformParameters;
  ImageRegionType                  m_FixedImageRegion;

  unsigned int                     m_NumberOfImages;
  unsigned int                     m_TargetImage;

  bool                             m_InitializeWithMask;
  bool                             m_UseReference;
  unsigned int                     m_Iterations;

};


} // end namespace mialsrtk


#ifndef ITK_MANUAL_INSTANTIATION
#include "mialsrtkLowToHighImageResolutionMethod.txx"
#endif

#endif
