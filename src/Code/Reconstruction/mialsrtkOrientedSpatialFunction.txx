/*=========================================================================

Copyright (c) 2017 Medical Image Analysis Laboratory (MIAL), Lausanne
  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
==========================================================================*/

#ifndef __mialsrtkOrientedSpatialFunction_txx
#define __mialsrtkOrientedSpatialFunction_txx

#include "mialsrtkOrientedSpatialFunction.h"

namespace mialsrtk
{

template <typename TOutput, unsigned int VImageDimension, typename TInput>
OrientedSpatialFunction<TOutput, VImageDimension, TInput>
::OrientedSpatialFunction()
{
  m_Direction.set_size(3,3);

  m_Center.set_size(3);
  m_Center.fill(0.0);

  m_Spacing.set_size(3);
  m_Spacing.fill(1.0);

  // Gaussian setup (ITK object)
  m_Gaussian = GaussianFunctionType::New();
  m_Gaussian -> SetNormalized(false);

  ArrayType mean;
  mean[0] = 0; mean[1] = 0; mean[2] = 0;
  m_Gaussian -> SetMean( mean );

  ArrayType sigma;

  //Compute sigma of the Gaussian PSF 
  sigma[0] = 1.2 * (m_Spacing[0]/2.3548);
  sigma[1] = 1.2 * (m_Spacing[1]/2.3548);
  sigma[2] = 1.0 * (m_Spacing[2]/2.3548);

  m_Gaussian -> SetSigma( sigma );

  // End Gaussian setup

  m_PSF = GAUSSIAN;

}

template <typename TOutput, unsigned int VImageDimension, typename TInput>
OrientedSpatialFunction<TOutput, VImageDimension, TInput>
::~OrientedSpatialFunction()
{

}

template <typename TOutput, unsigned int VImageDimension, typename TInput>
typename OrientedSpatialFunction<TOutput, VImageDimension, TInput>::OutputType
OrientedSpatialFunction<TOutput, VImageDimension, TInput>
::Evaluate(const TInput& position) const
{

  vnl_vector<double> diff = position.GetVnlVector() - m_Center;
  PointType diffPoint;

  //Dot product between image direction and point vector (in PSF space)
  double icoor = dot_product(diff,m_idir);
  double jcoor = dot_product(diff,m_jdir);
  double kcoor = dot_product(diff,m_kdir);

  diffPoint[0] = icoor;
  diffPoint[1] = jcoor;
  diffPoint[2] = kcoor;

  double value = 0.0;

  switch (m_PSF)
  {
  case BOXCAR:
    if ( ( fabs(icoor) <= 0.5 * m_Spacing[0] ) &&
           ( fabs(jcoor) <= 0.5 * m_Spacing[1] ) &&
           ( fabs(kcoor) <= 0.5 * m_Spacing[2]) )
          value = 1.0;
    break;
  case GAUSSIAN:
    value = m_Gaussian -> Evaluate( diffPoint );
    break;
  default:
    std::cout << "Unknown function" << std::endl;
    break;
  }

  return (TOutput) value;
}

template <typename TOutput, unsigned int VImageDimension, typename TInput>
void
OrientedSpatialFunction<TOutput, VImageDimension, TInput>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "Direction: " << m_Direction << std::endl;
  os << indent << "Center: " << m_Center << std::endl;
  os << indent << "Spacing: " << m_Spacing << std::endl;
}


} // end namespace mialsrtk

#endif
