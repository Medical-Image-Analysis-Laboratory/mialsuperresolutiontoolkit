/*==========================================================================

  © Université de Strasbourg - Centre National de la Recherche Scientifique

  Date: 12/11/2010
  Author(s): Estanislao Oubel (oubel@unistra.fr)
             Marc Schweitzer (marc.schweitzer(at)unistra.fr)

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

#ifndef __mialsrtk_RIGIDREGISTRATION_TXX__
#define __mialsrtk_RIGIDREGISTRATION_TXX__

#include "mialsrtkRigidRegistration.h"

namespace mialsrtk
{

/*
 * Constructor
 */
template < typename ImageType >
RigidRegistration<ImageType>::RigidRegistration()
{
    Superclass::m_Iterations = 2000; // 200
    Superclass::m_GradientMagnitudeTolerance=1e-4; // As set by default inITK and  BTK
    Superclass::m_MinStepLength = 0.0001; // As set by default in BTK 0.001
    Superclass::m_MaxStepLength = 0.2; // As set by default in BTK 0.1
    Superclass::m_RelaxationFactor = 0.5; // As set by default in BTK 0.8
    Superclass::m_EnableObserver = true;
    Superclass::m_FixedImageMask = 0;
    Superclass::m_MovingImageMask = 0;

    m_InitializeWithTransform = false;
    m_InitializeWithMask = false;



}

/*
 * Initialize by setting the interconnects between components.
 */
template < typename ImageType >
void
RigidRegistration<ImageType>
::Initialize() throw (itk::ExceptionObject)
{
    // Configure transform
    Superclass::m_Transform = RigidTransformType::New();

    Superclass::m_Interpolator = InterpolatorType::New();
    Superclass::m_Metric = MetricType::New();
    Superclass::m_Optimizer = OptimizerType::New();

    if (m_InitializeWithMask)
    {

        TransformInitializerType::Pointer initializer = TransformInitializerType::New();

        initializer -> SetTransform( dynamic_cast<RigidTransformType*>(Superclass::m_Transform.GetPointer()));

        // TODO Perhaps here I could add methods to set masks (as spatial objects)
        // and then recover images by using the GetImage() method.

        initializer -> SetFixedImage(  this -> GetFixedImageMask() );
        initializer -> SetMovingImage( this -> GetMovingImageMask() );
        initializer -> MomentsOn();
        initializer -> InitializeTransform();

    } else if (m_InitializeWithTransform)
    {
        Superclass::m_Transform -> SetParameters( this -> GetInitialTransformParameters() );
        dynamic_cast<RigidTransformType*>(Superclass::m_Transform.GetPointer())-> SetCenter( this -> GetTransformCenter() );
    } else
    {

        PointType rotationCenter;
        IndexType centerIndex;

        IndexType  fixedImageRegionIndex = this -> GetFixedImageRegion().GetIndex();
        SizeType   fixedImageRegionSize  = this -> GetFixedImageRegion().GetSize();

        centerIndex[0] = fixedImageRegionIndex[0] + fixedImageRegionSize[0] / 2.0;
        centerIndex[1] = fixedImageRegionIndex[1] + fixedImageRegionSize[1] / 2.0;
        centerIndex[2] = fixedImageRegionIndex[2] + fixedImageRegionSize[2] / 2.0;

        this -> GetFixedImage() -> TransformIndexToPhysicalPoint(centerIndex, rotationCenter);

        dynamic_cast<RigidTransformType*>(Superclass::m_Transform.GetPointer())-> SetIdentity();
        dynamic_cast<RigidTransformType*>(Superclass::m_Transform.GetPointer())-> SetCenter( rotationCenter );

    }

    this -> SetInitialTransformParameters( Superclass::m_Transform -> GetParameters() );




    // Configure metric
    if (Superclass::m_FixedImageMask)
    {
        Superclass::m_FixedMask = MaskType::New();
        Superclass::m_FixedMask -> SetImage( this -> GetFixedImageMask() );
        Superclass::m_Metric -> SetFixedImageMask( Superclass::m_FixedMask );
    }

    // FIXME Uncomment if MI is used instead of NC
    //Superclass::m_Metric -> SetNumberOfHistogramBins( 64 );
    //Superclass::m_Metric -> UseAllPixelsOn();

    //Uncomment if NC is used
    Superclass::m_Metric -> SetNumberOfSpatialSamples(0.5*this -> GetFixedImageRegion().GetNumberOfPixels());

    //Uncomment if NMI is used
    //unsigned int numberOfHistogramBins = 32;
    //typename MetricType::HistogramType::SizeType histogramSize;
    //histogramSize.SetSize(1);
    //histogramSize[0] = numberOfHistogramBins;
    //Superclass::m_Metric->SetHistogramSize( histogramSize );
    //const unsigned int numberOfParameters = Superclass::m_Transform -> GetNumberOfParameters();
    //typedef typename MetricType::ScalesType ScalesType;
    //ScalesType scales( numberOfParameters );
    //const double transscale = 1.0 / 100.0;
    //scales.Fill( 1.0 );
    //scales[0] = 1.0;
    //scales[1] = 1.0;
    //scales[2] = 1.0;
    //scales[3] = transscale;
    //scales[4] = transscale;
    //scales[5] = transscale;
    //Superclass::m_Metric->SetDerivativeStepLengthScales(scales);

    if(std::string(Superclass::m_Transform->GetNameOfClass()) == "VersorRigid3DTransform")
    {
        // Configure optimizer

        Superclass::m_Optimizer->MinimizeOn();// We want to minimize the negative mutual information
        Superclass::m_Optimizer->SetMaximumStepLength( Superclass::m_MaxStepLength );
        Superclass::m_Optimizer->SetMinimumStepLength( Superclass::m_MinStepLength );
        Superclass::m_Optimizer->SetNumberOfIterations( Superclass::m_Iterations );
        Superclass::m_Optimizer->SetRelaxationFactor( Superclass::m_RelaxationFactor );
        Superclass::m_Optimizer->SetGradientMagnitudeTolerance( Superclass::m_GradientMagnitudeTolerance );

        OptimizerScalesType optimizerScales( Superclass::m_Transform -> GetNumberOfParameters() );

        optimizerScales[0] =  1.0;
        optimizerScales[1] =  1.0;
        optimizerScales[2] =  1.0;
        optimizerScales[3] =  1.0/100.0;
        optimizerScales[4] =  1.0/100.0;
        optimizerScales[5] =  1.0/100.0;
        Superclass::m_Optimizer->SetScales( optimizerScales );
    }
    //  else if(std::string(Superclass::m_Transform->GetNameOfClass()) == "AffineTransform")
    //  {
    //      Superclass::m_Optimizer->MinimizeOn();
    //      Superclass::m_Optimizer->SetMaximumStepLength( 0.2 );
    //      Superclass::m_Optimizer->SetMinimumStepLength( 0.0001 );
    //      Superclass::m_Optimizer->SetNumberOfIterations( Superclass::m_Iterations );
    //      Superclass::m_Optimizer->SetGradientMagnitudeTolerance(0.00001);

    //      OptimizerScalesType optimizerScales( Superclass::m_Transform->GetNumberOfParameters() );

    //      optimizerScales[0] =  1.0;
    //      optimizerScales[1] =  1.0;
    //      optimizerScales[2] =  1.0;
    //      optimizerScales[3] =  1.0;
    //      optimizerScales[4] =  1.0;
    //      optimizerScales[5] =  1.0;
    //      optimizerScales[6] =  1.0;
    //      optimizerScales[7] =  1.0;
    //      optimizerScales[8] =  1.0;
    //      optimizerScales[9] =  1/1000.0;
    //      optimizerScales[10] =  1/1000.0;
    //      optimizerScales[11] =  1/1000.0;
    //      Superclass::m_Optimizer->SetScales( optimizerScales );


    //  }
    else
    {
        throw(std::string("Wrong type of Transform ! Only VersorRigid3D are accepted."));
    }


    Superclass::m_Observer = btk::CommandIterationUpdate::New();

    if (Superclass::m_EnableObserver)
    {
        Superclass::m_Optimizer -> AddObserver( itk::IterationEvent(), Superclass::m_Observer );
    }



    // Connect components
    this->SetTransform( Superclass::m_Transform );
    this->SetMetric( Superclass::m_Metric );
    this->SetOptimizer( Superclass::m_Optimizer );
    this->SetInterpolator( Superclass::m_Interpolator );


    Superclass::Initialize();

}

/*
 * PrintSelf
 */
template < typename ImageType >
void
RigidRegistration<ImageType>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf( os, indent );
}


} // end namespace mialsrtk


#endif
