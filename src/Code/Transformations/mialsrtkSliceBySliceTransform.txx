/*==========================================================================

  © Université de Strasbourg - Centre National de la Recherche Scientifique

  Date: 16/08/2011
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

#ifndef __SliceBySliceTransform_txx
#define __SliceBySliceTransform_txx

#include "itkNumericTraits.h"
#include "mialsrtkSliceBySliceTransform.h"
#include "vnl/algo/vnl_matrix_inverse.h"


namespace mialsrtk
{

template <class TScalarType,unsigned int NDimensions, typename TPixelType>
typename SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::OutputPointType
SliceBySliceTransform<TScalarType,NDimensions,TPixelType>
::TransformPoint(const InputPointType& p ) const
{
    OutputPointType Tp( p );
    typename ImageType::IndexType index;
    bool isInside = m_Image -> TransformPhysicalPointToIndex( Tp , index);

//    if(index[2] < 0 || index[2] > m_TransformList.size() -1 )
//    {
//        Tp = p;
//    }
    if(!isInside) //if point is not inside the image we don't transform it
    {
        Tp =p;
    }
    else
    {
        Tp = m_TransformList[ index[2] ] -> TransformPoint( Tp );
    }

    return Tp;
}

template <class TScalarType,unsigned int NDimensions, typename TPixelType>
typename SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::OutputVectorType
SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::
TransformVector(const InputVectorType& p) const
{
  OutputVectorType Tp( p );
  for ( TransformPointerListConstIterator it = this->m_TransformList.begin();
      it != this->m_TransformList.end(); ++it )
    {
    Tp = (*it)->TransformVector( Tp );
    }
  return Tp;
}

template <class TScalarType,unsigned int NDimensions, typename TPixelType>
typename SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::OutputVnlVectorType
SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::
TransformVector(const InputVnlVectorType &p ) const
{
  OutputVnlVectorType Tp( p );
  for ( TransformPointerListConstIterator it = this->m_TransformList.begin();
      it != this->m_TransformList.end(); ++it )
    {
    Tp = (*it)->TransformVector( Tp );
    }
  return Tp;
}

template <class TScalarType,unsigned int NDimensions, typename TPixelType>
typename SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::OutputCovariantVectorType
SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::
TransformCovariantVector( const InputCovariantVectorType &p ) const
{
  OutputCovariantVectorType Tp( p );
  for ( TransformPointerListConstIterator it = this->m_TransformList.begin();
      it != this->m_TransformList.end(); ++it )
    {
    Tp = (*it)->TransformCovariantVector( Tp );
    }
  return Tp;
}

template <class TScalarType,unsigned int NDimensions, typename TPixelType>
const typename SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::JacobianType&
SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::
GetJacobian(const InputPointType &p ) const
{

  this->m_Jacobian.SetSize( NDimensions, this->GetNumberOfParameters() );
  this->m_Jacobian.Fill(0.0);

  typename ImageType::IndexType index;
  m_Image -> TransformPhysicalPointToIndex( p , index);
  //JacobianType jacobian = m_TransformList[ index[2] ] -> GetJacobian(p); // FIXME : in ITK GetJacobian is replaced by ComputeJacobianWithRespectToParameters (const InputPointType &p, JacobianType &jacobian)
  JacobianType jacobian;
  m_TransformList[ index[2] ]->ComputeJacobianWithRespectToParameters( p, jacobian ) ;
  unsigned int offset = index[2]*m_TransformList[0]->GetNumberOfParameters();

  for(unsigned int i = 0; i < NDimensions; i++)
    for(unsigned int j = 0; j < m_TransformList[0]->GetNumberOfParameters(); j++)
    {
      this->m_Jacobian[i][j] = jacobian[i][j+offset];
    }

  return this->m_Jacobian;

}
template <class TScalarType,unsigned int NDimensions, typename TPixelType>
void SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::
ComputeJacobianWithRespectToParameters( const InputPointType & p, JacobianType &_jacobian ) const
{

  _jacobian.SetSize( NDimensions, this->GetNumberOfParameters() );
  _jacobian.Fill(0.0);

  typename ImageType::IndexType index;
  m_Image -> TransformPhysicalPointToIndex( p , index);

  JacobianType jacobian;
  m_TransformList[ index[2] ]->ComputeJacobianWithRespectToParameters( p, jacobian ) ;
  unsigned int offset = index[2]*m_TransformList[0]->GetNumberOfParameters();

  for(unsigned int i = 0; i < NDimensions; i++)
    for(unsigned int j = 0; j < m_TransformList[0]->GetNumberOfParameters(); j++)
    {
      _jacobian[i][j] = jacobian[i][j+offset];
    }
  this->m_Jacobian = _jacobian;

}
template <class TScalarType,unsigned int NDimensions, typename TPixelType>
void
SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::
SetImage( ImageType * image)
{
  m_Image = image;

}


template <class TScalarType,unsigned int NDimensions, typename TPixelType>
void
SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::
Initialize()
{

  typename ImageType::SizeType size = m_Image -> GetLargestPossibleRegion().GetSize();

  m_NumberOfSlices = size[2];
  m_TransformList.resize(m_NumberOfSlices);

  TransformPointer tmpTransform = TransformType::New();
  m_ParametersPerSlice = tmpTransform -> GetNumberOfParameters();
  this -> m_Parameters.SetSize( m_NumberOfSlices * m_ParametersPerSlice );

  InputPointType centerPoint;
  ContinuousIndexType centerIndex;

  centerIndex[0] = (size[0]-1)/2.0;
  centerIndex[1] = (size[1]-1)/2.0;

  for(unsigned int i=0; i<m_NumberOfSlices; i++)
  {
    centerIndex[2] =  i;
    m_Image -> TransformContinuousIndexToPhysicalPoint(centerIndex,centerPoint);

    m_TransformList[i] = TransformType::New();
    m_TransformList[i] -> SetIdentity();
    m_TransformList[i] -> SetCenter(centerPoint);
  }

  this -> Modified();
}

template <class TScalarType,unsigned int NDimensions, typename TPixelType>
void
SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::
Initialize( TransformBase * t )
{
  typename ImageType::SizeType size = m_Image -> GetLargestPossibleRegion().GetSize();

  m_NumberOfSlices = size[2];
  m_TransformList.resize(m_NumberOfSlices);

  TransformPointer tmpTransform = TransformType::New();
  m_ParametersPerSlice = tmpTransform -> GetNumberOfParameters();
  this -> m_Parameters.SetSize( m_NumberOfSlices * m_ParametersPerSlice );

  InputPointType centerPoint;
  ContinuousIndexType centerIndex;

  centerIndex[0] = (size[0]-1)/2.0;
  centerIndex[1] = (size[1]-1)/2.0;

  for(unsigned int i=0; i<m_NumberOfSlices; i++)
  {
    centerIndex[2] =  i;
    m_Image -> TransformContinuousIndexToPhysicalPoint(centerIndex,centerPoint);

    m_TransformList[i] = TransformType::New();
    m_TransformList[i] -> SetIdentity();
    //FIXME : if we set the center the transformation is not exactly the same as a global transform
    //I don't know the reason why
//    m_TransformList[i] -> SetCenter(centerPoint);
//    t -> SetCenter( m_TransformList[i] -> GetCenter() );

    m_TransformList[i] -> SetParameters( t -> GetParameters() );
  }

  this -> Modified();
}

template <class TScalarType,unsigned int NDimensions, typename TPixelType>
const typename SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::ParametersType&
SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::GetParameters(void) const
{
  for(unsigned int i=0; i<m_NumberOfSlices; i++)
  {
    ParametersType param = m_TransformList[i] -> GetParameters();

    for (unsigned int p=0; p<m_ParametersPerSlice; p++)
    {
      this->m_Parameters[p + i*m_ParametersPerSlice] = param[p];
    }

  }
  return this->m_Parameters;

}

template <class TScalarType,unsigned int NDimensions, typename TPixelType>
void
SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::
SetParameters( const ParametersType & parameters )
{

  this -> m_Parameters = parameters;

  for(unsigned int i=0; i<m_NumberOfSlices; i++)
  {
    ParametersType param(m_ParametersPerSlice);

    for (unsigned int p=0; p<m_ParametersPerSlice; p++)
      param[p] = this->m_Parameters[p + i*m_ParametersPerSlice];

    m_TransformList[i] -> SetParameters( param );

  }

  this -> Modified();
}

template <class TScalarType,unsigned int NDimensions, typename TPixelType>
void
SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::
SetFixedParameters( const ParametersType & fp )
{
  this -> m_FixedParameters = fp;

  m_NumberOfSlices = this -> m_FixedParameters[0];

  if (m_TransformList.size() == 0 )
  {

    m_TransformList.resize(m_NumberOfSlices);

    TransformPointer tmpTransform = TransformType::New();
    m_ParametersPerSlice = tmpTransform -> GetNumberOfParameters();
    this -> m_Parameters.SetSize( m_NumberOfSlices * m_ParametersPerSlice );

    for(unsigned int i=0; i<m_NumberOfSlices; i++)
    {
      m_TransformList[i] = TransformType::New();
      m_TransformList[i] -> SetIdentity();
    }

  }

  InputPointType c;
  typedef typename ParametersType::ValueType ParameterValueType;

  for ( unsigned int j = 0; j<m_NumberOfSlices; j++)
  {
    for ( unsigned int i = 0; i < NDimensions; i++ )
      c[i] = this -> m_FixedParameters[j*NDimensions + i + 1];

    m_TransformList[j] -> SetCenter ( c );
  }

  this -> Modified();

}

template <class TScalarType,unsigned int NDimensions, typename TPixelType>
const typename SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::ParametersType&
SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::
GetFixedParameters(void) const
{
  this->m_FixedParameters.SetSize ( NDimensions * m_NumberOfSlices + 1 );

  this->m_FixedParameters[0] = m_NumberOfSlices;

  for ( unsigned int j = 0; j < m_NumberOfSlices; j++ )
  {
    for ( unsigned int i = 0; i < NDimensions; i++ )
      {
      this->m_FixedParameters[j*NDimensions + i + 1] = m_TransformList[j]-> GetCenter()[i];
      }
  }
  return this->m_FixedParameters;
}

template <class TScalarType,unsigned int NDimensions, typename TPixelType>
void
SliceBySliceTransform<TScalarType,NDimensions,TPixelType>::
GetInverse(Self* inverse) const
{
    inverse->SetFixedParameters(this->GetFixedParameters());
    inverse->SetImage(this->m_Image.GetPointer());

    for(int i = 0; i< m_NumberOfSlices; i++)
    {
        typename TransformType::Pointer t = TransformType::New();
        m_TransformList[i]->GetInverse(t);
        inverse->SetSliceParameters(i,t->GetParameters());
    }

}

} // namespace

#endif
