/*==========================================================================

  © Université de Strasbourg - Centre National de la Recherche Scientifique

  Date: 14/05/2010
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

#ifndef __mialsrtkResampleImageByInjectionFilter_h
#define __mialsrtkResampleImageByInjectionFilter_h

#include "itkFixedArray.h"
#include "itkTransform.h"
//#include "itkEuler3DTransform.h"
#include "itkVersorRigid3DTransform.h"

#include "mialsrtkSliceBySliceTransform.h"
#include "itkImageFunction.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageToImageFilter.h"
#include "itkInterpolateImageFunction.h"
#include "itkNeighborhoodAlgorithm.h"
#include "itkSize.h"
#include "btkUserMacro.h"
#include "itkImageMaskSpatialObject.h"
#include "mialsrtkSliceBySliceTransformBase.h"

//using namespace btk;

namespace mialsrtk
{

/** @class ResampleImageByInjectionFilter
 * @brief Resample an image via a coordinate transform
 *
 * ResampleImageByInjectionFilter resamples an existing image through some coordinate
 * transform, interpolating via some image function.  The class is templated
 * over the types of the input and output images.
 *
 * Output information (spacing, size and direction) for the output
 * image should be set. This information has the normal defaults of
 * unit spacing, zero origin and identity direction. Optionally, the
 * output information can be obtained from a reference image. If the
 * reference image is provided and UseReferenceImage is On, then the
 * spacing, origin and direction of the reference image will be used.
 *
 * Since this filter produces an image which is a different size than
 * its input, it needs to override several of the methods defined
 * in ProcessObject in order to properly manage the pipeline execution model.
 * In particular, this filter overrides
 * ProcessObject::GenerateInputRequestedRegion() and
 * ProcessObject::GenerateOutputInformation().
 * @author Estanislao Oubel
 *
 * @ingroup Reconstruction
 */
template <class TInputImage, class TOutputImage,
          class TInterpolatorPrecisionType=double>
class ResampleImageByInjectionFilter:
    public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef ResampleImageByInjectionFilter                Self;
  typedef itk::ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Number of dimensions. */
  itkStaticConstMacro(ImageDimension, unsigned int, TOutputImage::ImageDimension);
  itkStaticConstMacro(InputImageDimension, unsigned int, TInputImage::ImageDimension);

  typedef TInputImage                             InputImageType;
  typedef TOutputImage                            OutputImageType;
  typedef typename InputImageType::Pointer        InputImagePointer;
  typedef typename InputImageType::ConstPointer   InputImageConstPointer;
  typedef typename OutputImageType::Pointer       OutputImagePointer;


  typedef itk::Image<float,ImageDimension>    FloatImageType;
  typedef typename FloatImageType::Pointer FloatImagePointer;

  /** Type of the image mask. */
  typedef itk::Image< unsigned char, ImageDimension >    ImageMaskType;
  typedef typename ImageMaskType::Pointer ImageMaskPointer;

  /** Type of the mask. */
  typedef itk::ImageMaskSpatialObject< ImageDimension >  MaskType;

  typedef typename InputImageType::RegionType     InputImageRegionType;
  typedef std::vector<InputImageRegionType>       InputImageRegionVectorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ResampleImageByInjectionFilter, ImageToImageFilter);


  /** Image size typedef. */
  typedef itk::Size<itkGetStaticConstMacro(ImageDimension)> SizeType;

  /** Image index typedef. */
  typedef typename TOutputImage::IndexType IndexType;

  /** Image point typedef. */
  typedef typename TOutputImage::PointType    PointType;

  /** Image pixel value typedef. */
  typedef typename TOutputImage::PixelType   PixelType;
  typedef typename TInputImage::PixelType    InputPixelType;

  /** Typedef to describe the output image region type. */
  typedef typename TOutputImage::RegionType OutputImageRegionType;

  /** Image spacing, origin and direction typedef */
  typedef typename TOutputImage::SpacingType   SpacingType;
  typedef typename TOutputImage::PointType     OriginPointType;
  typedef typename TOutputImage::DirectionType DirectionType;

  /** base type for images of the current ImageDimension */
  typedef itk::ImageBase<itkGetStaticConstMacro(ImageDimension)> ImageBaseType;

  /** Type of the slice by slice transform. */
  typedef SliceBySliceTransformBase< double, ImageDimension,InputPixelType > TransformType;
  typedef typename TransformType::Pointer TransformPointer;

  typedef std::vector< TransformPointer > TransformPointerArrayType;


  /**Iterator typedef. */
  typedef itk::ImageRegionIteratorWithIndex< InputImageType >  IteratorType;

  /**Const iterator typedef. */
  typedef itk::ImageRegionConstIteratorWithIndex< InputImageType >  ConstIteratorType;
  typedef itk::ImageRegionConstIteratorWithIndex< ImageMaskType >  ConstMaskIteratorType;

  /**Const float iterator typedef. */
  typedef itk::ImageRegionConstIteratorWithIndex< FloatImageType >  ConstFloatIteratorType;

  /**Neighborhood iterator typedef. */
  typedef itk::NeighborhoodIterator< OutputImageType >  NeighborhoodIteratorType;

  /**Float neighborhood iterator typedef. */
  typedef itk::NeighborhoodIterator< FloatImageType >   FloatNeighborhoodIteratorType;

  /**Face calculator typedef. */
  typedef itk::NeighborhoodAlgorithm
  ::ImageBoundaryFacesCalculator< FloatImageType > FaceCalculatorType;
  typedef typename FaceCalculatorType::FaceListType FaceListType;

  typedef vnl_matrix<double> VnlMatrixType;
  typedef itk::Matrix<double,3,3> MatrixType;
  typedef vnl_vector<double> VnlVectorType;

  // Overrides SetInput to resize the transform

  void AddInput(InputImageType* _arg)
  {
    m_ImageArray.push_back(_arg);

    this -> SetInput(_arg);

    m_Transform.resize( m_Transform.size() + 1 );
  }

  /** Set a the transform for the image i. */
  void SetTransform( int i, TransformType* transform )
  {
    m_Transform[i] = transform;
  }

  /** Get a the transform for the image i. */
  TransformType* GetTransform( int i)
  {
    return this -> m_Transform[i].GetPointer();
  }

  /** Set the size of the output image. */
  itkSetMacro( Size, SizeType );

  /** Get the size of the output image. */
  itkGetConstReferenceMacro( Size, SizeType );

  /** Set the pixel value when a transformed pixel is outside of the
   * image.  The default default pixel value is 0. */
  itkSetMacro( DefaultPixelValue, PixelType );

  /** Get the pixel value when a transformed pixel is outside of the image */
  itkGetConstReferenceMacro( DefaultPixelValue, PixelType );

  /** Set the output image spacing. */
  itkSetMacro( OutputSpacing, SpacingType );
  virtual void SetOutputSpacing( const double* values );

  /** Get the output image spacing. */
  itkGetConstReferenceMacro( OutputSpacing, SpacingType );

  /** Set the output image origin. */
  itkSetMacro( OutputOrigin, OriginPointType );
  virtual void SetOutputOrigin( const double* values);

  /** Get the output image origin. */
  itkGetConstReferenceMacro( OutputOrigin, OriginPointType );

  /** Set the output direciton cosine matrix. */
  itkSetMacro( OutputDirection, DirectionType );
  itkGetConstReferenceMacro( OutputDirection, DirectionType );

  /** Helper method to set the output parameters based on this image */
  void SetOutputParametersFromImage ( const ImageBaseType * image );

  /** Set the start index of the output largest possible region.
   * The default is an index of all zeros. */
  itkSetMacro( OutputStartIndex, IndexType );

  /** Get the start index of the output largest possible region. */
  itkGetConstReferenceMacro( OutputStartIndex, IndexType );

  /** Copy the output information from another Image.  By default,
   *  the information is specified with the SetOutputSpacing, Origin,
   *  and Direction methods. UseReferenceImage must be On and a
   *  Reference image must be present to override the defaul behavior.
   *  NOTE: This function seems redundant with the
   *  SetOutputParametersFromImage( image ) function */
  void SetReferenceImage ( const TOutputImage *image );
  const TOutputImage * GetReferenceImage( void ) const;

  itkSetMacro( UseReferenceImage, bool );
  itkBooleanMacro( UseReferenceImage );
  itkGetConstMacro( UseReferenceImage, bool );

  itkSetMacro( UseDebluringKernel, bool );
  itkBooleanMacro( UseDebluringKernel );
  itkGetConstMacro( UseDebluringKernel, bool );

  /** ResampleImageByInjectionFilter produces an image which is a different size
   * than its input.  As such, it needs to provide an implementation
   * for GenerateOutputInformation() in order to inform the pipeline
   * execution model.  The original documentation of this method is
   * below. \sa ProcessObject::GenerateOutputInformaton() */
  virtual void GenerateOutputInformation( void );

  /** ResampleImageByInjectionFilter needs a different input requested region than
   * the output requested region.  As such, ResampleImageByInjectionFilter needs
   * to provide an implementation for GenerateInputRequestedRegion()
   * in order to inform the pipeline execution model.
   * \sa ProcessObject::GenerateInputRequestedRegion() */
  virtual void GenerateInputRequestedRegion( void );

  /** Method Compute the Modified Time based on changed to the components. */
  unsigned long GetMTime( void ) const;

  void AddRegion(InputImageRegionType _arg)
  {
    m_InputImageRegion.push_back(_arg);
  }

  void AddMask(ImageMaskPointer _arg)
  {
    m_ImageMaskArray.push_back(_arg);
  }

  /** Sets the mask where to perform the injection. */
  itkSetObjectMacro(ReferenceImageMask, ImageMaskType);


#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(OutputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<PixelType>));
  /** End concept checking */
#endif

protected:
  ResampleImageByInjectionFilter( void );
  ~ResampleImageByInjectionFilter( void ) {};

  void PrintSelf( std::ostream& os, itk::Indent indent ) const;

  /** ResampleImageFilter cannot be implemented as a multithreaded filter.  Therefore,
   * this implementation only provides a GenerateData() routine that allocates output
   * image data.
   * */
  void GenerateData();

  /** This method overrides the itkImageToImageFilter's
   *  This method do nothing, we don't want to verify if inputs are in the same physical space
   * */
  virtual void VerifyInputInformation() {};

private:
  ResampleImageByInjectionFilter( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented

  SizeType                    m_Size;              // Size of the output image
  TransformPointerArrayType   m_Transform;         // Coordinate transform to use
  InputImageRegionVectorType  m_InputImageRegion;

  std::vector<InputImagePointer> m_ImageArray;
  std::vector<ImageMaskPointer>  m_ImageMaskArray;
  ImageMaskPointer 							 m_ReferenceImageMask;

  PixelType                   m_DefaultPixelValue; // default pixel value
                                                     // if the point is
                                                     // outside the image
  SpacingType                 m_OutputSpacing;     // output image spacing
  OriginPointType             m_OutputOrigin;      // output image origin
  DirectionType               m_OutputDirection;   // output image direction cosines
  IndexType                   m_OutputStartIndex;  // output image start index
  bool                        m_UseReferenceImage;

  bool                        m_UseDebluringKernel;

};


} // end namespace mialsrtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "mialsrtkResampleImageByInjectionFilter.txx"
#endif

#endif
