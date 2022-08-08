#ifndef _mialsrtkSliceBySliceRigidRegistration_txx
#define _mialsrtkSliceBySliceRigidRegistration_txx

#include "mialsrtkSliceBySliceRigidRegistration.h"

namespace mialsrtk
{

/*
 * Constructor
 */
template < typename ImageType >
SliceBySliceRigidRegistration<ImageType>
::SliceBySliceRigidRegistration()
{
  m_Interpolator = 0;
  m_ImageMask = 0;
  m_Transform = 0;
  m_Iterations = 2000; // As set by default in BTK: 200
  m_GradientMagnitudeTolerance=1e-4; // As set by default in ITK and  BTK
  m_MinStepLength = 0.0001; // As set by default in BTK: 0.001
  m_MaxStepLength = 0.2; // As set by default in BTK: 0.1
  m_RelaxationFactor = 0.5; // As set by default in BTK: 0.8
}

/*
 * Initialize by setting the interconnects between components.
 */
template < typename ImageType >
void
SliceBySliceRigidRegistration<ImageType>
::Initialize() throw (itk::ExceptionObject)
{
  // Configure registration

  //std::cout << "m_FixedImage" << m_FixedImage << std::endl;
  //std::cout << "m_MovingImage" << m_MovingImage << std::endl;
  //std::cout << "m_Iterations" << m_Iterations << std::endl;
  //std::cout << "m_Transform" << m_Transform << std::endl;

  m_Registration = RegistrationType::New();
  m_Registration -> SetFixedImage(  m_FixedImage  );
  m_Registration -> SetMovingImage( m_MovingImage );
  m_Registration -> InitializeWithTransform();
  m_Registration -> SetEnableObserver( false );

  m_Registration -> SetIterations( m_Iterations );
  m_Registration -> SetGradientMagnitudeTolerance( m_GradientMagnitudeTolerance );
  m_Registration -> SetMinStepLength( m_MinStepLength );
  m_Registration -> SetMaxStepLength( m_MaxStepLength );
  m_Registration -> SetRelaxationFactor( m_RelaxationFactor );

  // TODO We have to decide after checking the results which one is the
  // the default behavior

  if ( !m_ImageMask )
  {
//    std::cout << "image mask IS NOT defined" << std::endl;
    m_ROI  = m_FixedImage -> GetLargestPossibleRegion().GetSize();
  } else
    {
//      std::cout << "image mask IS defined" << std::endl;
      typename MaskType::Pointer mask = MaskType::New();
      mask -> SetImage( m_ImageMask );
      m_Registration -> SetFixedImageMask( m_ImageMask );
      m_ROI = mask -> GetAxisAlignedBoundingBoxRegion();
    }
    
//  if ( !m_Transform)
//  {
//    m_Transform = SliceBySliceTransformType::New();
//    m_Transform -> SetImage( m_FixedImage );
//    m_Transform -> Initialize();
//  }

    //std::cout << "registration init done..." << std::endl;

}

/*
 * Starts the Registration Process
 */
template < typename ImageType >
void
SliceBySliceRigidRegistration<ImageType>
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

    //std::cout <<"After init."  << std::endl;

    unsigned int k1 =  m_ROI.GetIndex()[2];//First slice Id
    unsigned int k2 =  k1 + m_ROI.GetSize()[2] -1;

    //First globally register  all even and odd slices separately
    int numberOfslices = k2 - k1 + 1;

    typename ImageType::IndexType indexEvenImage = m_ROI.GetIndex();
    typename ImageType::IndexType indexOddImage = m_ROI.GetIndex();

    typename ImageType::SizeType sizeEvenImage = m_ROI.GetSize();
    typename ImageType::SizeType sizeOddImage = m_ROI.GetSize();

    const bool verbose = false;
    /*
    if(numberOfslices % 2 == 0)
    {
        sizeEvenImage[2] = numberOfslices / 2;
        sizeOddImage[2] = numberOfslices / 2;
    }
    else
    {
        sizeEvenImage[2] = round(numberOfslices / 2) + 1;
        sizeOddImage[2] = round(numberOfslices / 2);
    }
    */

    if(k1 % 2 == 0)
    {
        sizeOddImage[2] += 1;
    }
    else
    {
        indexEvenImage[2] += 1;
    }

    for ( unsigned int i = k1; i <= k2; i++ )
    { 
      if (verbose){
          std::cout << "Registering slice " << i << std::endl;
      }
      //TODO: outlier rejection scheme
      // We could store MSE between the registered slice and the slice in the HR volume 

      // Fixed region for slice i

      RegionType fixedImageRegion;
      IndexType  fixedImageRegionIndex;
      SizeType   fixedImageRegionSize;

      fixedImageRegionIndex = m_ROI.GetIndex();
      fixedImageRegionIndex[2] = i;

      fixedImageRegionSize = m_ROI.GetSize();
      fixedImageRegionSize[2] = 1;

      fixedImageRegion.SetIndex(fixedImageRegionIndex);
      fixedImageRegion.SetSize(fixedImageRegionSize);

      ParametersType initialRigidParameters( 6 );
      ParametersType finalParameters;

      m_Registration -> SetFixedImageRegion( fixedImageRegion );
      m_Registration -> SetInitialTransformParameters( m_Transform -> GetSliceTransform(i) -> GetParameters() );
      m_Registration -> SetTransformCenter( m_Transform -> GetSliceTransform(i) -> GetCenter() );

//      std::cout << "Initial registration parameters = " << m_Transform -> GetSliceTransform(i) -> GetParameters() << std::endl;

      try
        {
          //std::cout << "Before Update()" << std::endl;
        //m_Registration -> StartRegistration();// FIXME : in ITK4 StartRegistration() is replaced by Update()
        m_Registration->Update();
        //std::cout << "After Update()" << std::endl;
        }
      catch( itk::ExceptionObject & err )
        {
          std::cout << "Exception caught update registration" << std::endl;
        // TODO: Always check in case of problems: the following lines have been commented
        // on purpose (if the registration fails we prefer keep the old parameters)

  //      std::cerr << "ExceptionObject caught !" << std::endl;
  //      std::cerr << err << std::endl;
  //      return EXIT_FAILURE;
        }

      //std::cout << "Getting final parameters" << std::endl;  

      finalParameters = m_Registration -> GetLastTransformParameters();

      //std::cout << "Final rigid parameters = " << finalParameters << std::endl;

      m_Transform -> SetSliceParameters( i, finalParameters );

    } // end for in z

}

/*
 * PrintSelf
 */
template < typename ImageType >
void
SliceBySliceRigidRegistration<ImageType>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}


} // end namespace mialsrtk


#endif
