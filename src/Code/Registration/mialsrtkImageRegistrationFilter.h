/*==========================================================================

  © Université de Strasbourg - Centre National de la Recherche Scientifique

  Date: 24/01/2011
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

#ifndef __mialsrtk_IMAGEREGISTRATIONFILTER_H__
#define __mialsrtk_IMAGEREGISTRATIONFILTER_H__

/* Standard includes */
#include <tclap/CmdLine.h>
#include "stdio.h"

/* Itk includes */
#include "itkNormalizedCorrelationImageToImageMetric.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkEuler3DTransform.h"
#include "itkTransformFileWriter.h"
#include "itkImage.h"
#include "itkImageMaskSpatialObject.h"
#include "itkCastImageFilter.h"
#include "itkImageDuplicator.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

/*Btk includes*/
//#include "btkSliceBySliceTransform.h"
#include "mialsrtkSliceBySliceTransform.h"
#include "mialsrtkVersorSliceBySliceTransform.h"
#include "mialsrtkSliceBySliceTransformBase.h"
#include "mialsrtkLowToHighImageResolutionMethod.h"
#include "mialsrtkResampleImageByInjectionFilter.h"
#include "mialsrtkImageIntersectionCalculator.h"

/* mialsrtk includes*/
#include "mialsrtkSliceBySliceRigidRegistration.h"
#include "mialsrtkBSplineInterpolateImageFunction.h"

namespace mialsrtk
{

/** @class ImageRegistrationFilter
 * @brief This class is usefull to update registration during sr rconstruction
 * @author Sebastien Tourbier & Marc Schweitzer
 * @ingroup Registration
 */
  template <class TImage, class TMask, unsigned int Dimension>
  class ImageRegistrationFilter
  {
    // typedefs

    public:

      typedef ImageRegistrationFilter Self;
      //const    unsigned int    Dimension = 3;

      typedef TImage  ImageType;
      typedef typename ImageType::Pointer                  ImagePointer;

      typedef typename ImageType::SizeType                 SizeType;

      typedef typename ImageType::RegionType               RegionType;
      typedef typename std::vector< RegionType >           RegionArrayType;

      typedef typename itk::ImageFileReader< ImageType  >  ImageReaderType;

      //typedef itk::Image< unsigned char, 3 >    ImageMaskType;
      typedef TMask   ImageMaskType;
      typedef typename ImageMaskType::Pointer                    ImageMaskPointer;

      typedef typename itk::ImageFileReader< ImageMaskType >     MaskReaderType;
      typedef itk::ImageMaskSpatialObject< Dimension >  MaskType;
      typedef typename MaskType::Pointer  MaskPointer;

      /* Registration type required in case of slice by slice transformations
      A rigid transformation is employed because there is not distortions like
      in diffusion imaging. We have performed a comparison of accuracy between
      both types of transformations. */
      typedef typename mialsrtk::SliceBySliceRigidRegistration<ImageType> RegistrationType;
      typedef typename RegistrationType::Pointer RegistrationPointer;

      // Registration type required in case of 3D affine trasforms
      //typedef btk::RigidRegistration<ImageType> Rigid3DRegistrationType;
      //typedef Rigid3DRegistrationType::Pointer Rigid3DRegistrationPointer;

      // Slice by slice transform definition (typically for in utero reconstructions)
      typedef mialsrtk::SliceBySliceTransformBase< double, Dimension > TransformBaseType;
      typedef mialsrtk::SliceBySliceTransform< double, Dimension > TransformType;
      //itk::TransformFactory<TransformType>::RegisterTransform(); // In principle, with a factory you only need to register the additional transform types that you expect to read.
      
      typedef typename TransformType::Pointer                          TransformPointer;

      // Rigid 3D transform definition (typically for reconstructions in adults)
      //typedef btk::Euler3DTransform< double > Rigid3DTransformType;
      //typedef Rigid3DTransformType::Pointer   Rigid3DTransformPointer;

      // This filter does a rigid registration over all the LR images and compute the average image in HR space
      //typedef typename btk::LowToHighImageResolutionMethod<ImageType,Rigid3DTransformType > LowToHighResFilterType;
      //LowToHighResFilterType::Pointer lowToHighResFilter = LowToHighResFilterType::New();

      // Resampler type required in case of a slice by slice transform
      typedef typename mialsrtk::ResampleImageByInjectionFilter< ImageType, ImageType>  ResamplerType;

      typedef typename itk::NormalizedCorrelationImageToImageMetric< ImageType,ImageType > NCMetricType;

      typedef typename mialsrtk::ImageIntersectionCalculator<ImageType> IntersectionCalculatorType;
      //IntersectionCalculatorType::Pointer intersectionCalculator = IntersectionCalculatorType::New();

      typedef typename itk::CastImageFilter<ImageType,ImageMaskType> CasterType;
      typedef typename itk::ImageDuplicator<ImageType> DuplicatorType;



      /** Adds an input image **/
      void AddImage(ImageType* image);

      /** Adds an input mask **/
      void AddMask(MaskType* mask);

      /** Sets an initialization transform **/
      void SetTransform(TransformType* transform);

      /** Sets the number max of iterations**/
      void SetMaxIterations(unsigned int iters);

      /** Sets the convergence threshold epsilon**/
      void SetEpsilon(double epsilon);

      /** Sets the margin to be added to the reconstructed image to compensate a small FOV in the reference image**/
      void SetMargin(double margin);

      /** Gets the current transforms **/

      /** Run registration of stacks **/
      void Update();

      /** Default constructor **/
      ImageRegistrationFilter();

    protected:
      
      /** Destructor */
      virtual ~ImageRegistrationFilter(){};

    private:

      std::vector<ImagePointer>                         m_Images;
      std::vector< ImageMaskPointer >                   m_ImagesMasks;
      std::vector<RegionType>                           m_Rois;
      std::vector<MaskPointer>                          m_Masks;
      ImagePointer                                      m_HrImage;
      ImagePointer                                      m_HrImageOld;
      ImagePointer                                      m_HrImageIni;
      ImagePointer                                      m_HrRefImage;

      std::vector< TransformPointer >                   m_Transforms;
      //std::vector< Rigid3DTransformPointer >            m_Rigid3DTransforms;

      std::vector< RegistrationPointer >                m_Registration;
      //std::vector< Rigid3DRegistrationPointer >         m_Rigid3DRegistration;

      unsigned int m_ItMax;
      double m_Epsilon;
      double m_Margin;

      //bool rigid3D;
      //bool m_noreg;

  };//end class

} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "mialsrtkImageRegistrationFilter.txx"
#endif

#endif /* mialsrtkIMAGEREGISTRATIONFILTER_H_ */
