/*==========================================================================

  © Université de Strasbourg - Centre National de la Recherche Scientifique

  Date: 09/05/2012
  Author(s): Marc Schweitzer (marc.schweitzer(at)unistra.fr)

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

#ifndef __mialsrtk_REGISTRATION_H__
#define __mialsrtk_REGISTRATION_H__

#include "itkImageRegistrationMethod.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkVersorRigid3DTransformOptimizer.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkAffineTransform.h"
#include "itkTransform.h"
#include "itkImageMaskSpatialObject.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkNormalizedCorrelationImageToImageMetric.h"
#include "itkNormalizedMutualInformationHistogramImageToImageMetric.h"

#include "itkNumericTraits.h"
#include "btkMacro.h"
#include "btkCommandIterationUpdate.h"

#include "mialsrtkBSplineInterpolateImageFunction.h"

namespace mialsrtk
{

/** @class Registration
 * @brief This is a base class for doing registration
 * @author Marc Schweitzer, modified by Sebastien Tourbier - Enable more control on the choice of metrics, optimization parameters, including min/max Step Length, gradient magnitude tolerance, relaxation factor.  (\today)
 * @ingroup Registration
 */

template <typename TImage>
class Registration :public itk::ImageRegistrationMethod <TImage,TImage>
{
    public:
        /** Standard class typedefs. */
        typedef Registration  Self;
        typedef itk::ImageRegistrationMethod<TImage,TImage>       Superclass;
        typedef itk::SmartPointer<Self>                           Pointer;
        typedef itk::SmartPointer<const Self>                     ConstPointer;


        /** Run-time type information (and related methods). */
        itkTypeMacro(Registration, ImageRegistrationMethod);

        /**  Type of the Fixed image. */
        typedef          TImage                               ImageType;
        typedef typename ImageType::Pointer                   ImagePointer;

        /**  Type of the image mask. */
        typedef itk::Image< unsigned char, 3 >                     ImageMaskType;
        typedef typename ImageMaskType::Pointer               ImageMaskPointer;

        /**  Type of the image mask spatial object. */
        typedef itk::ImageMaskSpatialObject< 3 >                   MaskType;
        typedef typename MaskType::Pointer                    MaskPointer;

        typedef typename ImageType::PointType                 PointType;


        /**  Type of the metric. */
        /**
   * --------------------- /!\ Warning /!\ ------------------------------
   *
   * - itk::MattesMutualInformationImageToImageMetric is NOT THREAD SAFE....
   *
   * ---------------------------------------------------------------------
   *
   * typedef MattesMutualInformationImageToImageMetric<
   */
        //typedef MattesMutualInformationImageToImageMetric<
        //                                      ImageType,
        //                                    ImageType >   MetricType;

        typedef itk::NormalizedCorrelationImageToImageMetric<
        ImageType,
        ImageType >   MetricType;

        //typedef itk::NormalizedMutualInformationHistogramImageToImageMetric<
        //ImageType,
        //ImageType >   NMIMetricType;


        typedef typename MetricType::Pointer                  MetricPointer;



        /**  Type of the Transform . */
        typedef itk::Transform< double, ImageType::ImageDimension,
                                        ImageType::ImageDimension > TransformType;
        typedef typename TransformType::Pointer               TransformPointer;

        typedef typename TransformType::ParametersType        ParametersType;

        /**  Type of the optimizer. */
        //typedef itk::RegularStepGradientDescentOptimizer           OptimizerType;
        typedef itk::VersorRigid3DTransformOptimizer           OptimizerType;

        /**  Type of the Interpolator. */
        //typedef itk::LinearInterpolateImageFunction<
        //                                  ImageType,
        //                                  double>             InterpolatorType;

        typedef mialsrtk::BSplineInterpolateImageFunction<
                                          ImageType, 
                                          double>             InterpolatorType;   
        typedef typename InterpolatorType::Pointer            InterpolatorPointer;


        /** Method that initiates the registration. */

        /** Set/Get transform center. */
        itkSetMacro(TransformCenter, PointType);

        virtual PointType GetTransformCenter() const
        {
            return m_TransformCenter;
        }


          virtual TransformType* GetTransform()
          {
            return m_Transform.GetPointer();
          }

        itkSetObjectMacro(FixedImageMask, ImageMaskType);
        itkGetObjectMacro(FixedImageMask, ImageMaskType);

        /** Set/Get image mask for moving image. */
        itkSetObjectMacro(MovingImageMask, ImageMaskType);
        itkGetObjectMacro(MovingImageMask, ImageMaskType);

        /** Set/Get iterations. */
        itkSetMacro(Iterations, unsigned int);
        itkGetMacro(Iterations, unsigned int);

        /** Set/Get gradient magnitude tolerance of the optimizer. */
        itkSetMacro(GradientMagnitudeTolerance,double);
        itkGetMacro(GradientMagnitudeTolerance,double);

        /** Set/Get min Step Length of the optimizer. */
        itkSetMacro(MinStepLength,double);
        itkGetMacro(MinStepLength,double);

        /** Set/Get max Step Length of the optimizer. */
        itkSetMacro(MaxStepLength,double);
        itkGetMacro(MaxStepLength,double);

        /** Set/Get relaxationFactor of the optimizer. */
        itkSetMacro(RelaxationFactor,double);
        itkGetMacro(RelaxationFactor,double);

        /** Set/Get observer status. */
        itkSetMacro(EnableObserver, bool);
        itkGetMacro(EnableObserver, bool);

        itkSetObjectMacro(FixedMask,MaskType);
        itkGetObjectMacro(FixedMask,MaskType);

        itkGetObjectMacro(Optimizer, OptimizerType);

        /** Initialization is performed with the provided transform. */
        virtual void InitializeWithTransform() = 0;


        /** Initialization is performed with the provided image masks. */
        virtual void InitializeWithMask() = 0;


    protected:
        Registration(){};
        virtual ~Registration() {};

        virtual void PrintSelf(std::ostream& os, itk::Indent indent) const
        {
            Superclass::PrintSelf( os, indent );
        }


        /** Initialize by setting the interconnects between the components.
   */
        virtual void Initialize() throw (itk::ExceptionObject)
        {
            Superclass::Initialize();
        }

        typename MetricType::Pointer        m_Metric;
        typename OptimizerType::Pointer     m_Optimizer;
        typename TransformType::Pointer         m_Transform;
        typename InterpolatorType::Pointer  m_Interpolator;

        unsigned int m_Iterations;
        double m_GradientMagnitudeTolerance;
        double m_MinStepLength;
        double m_MaxStepLength;
        double m_RelaxationFactor;
        bool m_EnableObserver;

        ImageMaskPointer     m_FixedImageMask;
        ImageMaskPointer     m_MovingImageMask;

        MaskPointer          m_FixedMask;

        btk::CommandIterationUpdate::Pointer  m_Observer;

        PointType                 m_TransformCenter;


    private:
        Registration(const Self&); //purposely not implemented
        void operator=(const Self&); //purposely not implemented


};


} // end namespace itk


#endif
