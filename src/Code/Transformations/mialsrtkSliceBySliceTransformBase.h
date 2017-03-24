/*==========================================================================

  © Université de Strasbourg - Centre National de la Recherche Scientifique

  Date: 16/08/2011
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

#ifndef __mialsrtk_SLICEBYSLICETRANSFORMBASE_H__
#define __mialsrtk_SLICEBYSLICETRANSFORMBASE_H__

#include "itkTransform.h"
#include "itkMatrixOffsetTransformBase.h"
#include "itkImage.h"
#include "itkContinuousIndex.h"
#include "list"

namespace mialsrtk
{
/**
 * @class SliceBySliceTransformBase
 * @brief Base class for slice-by-slice transformation, inheritate from an itk::Transform to be generic
 * @ingroup Transformations
 * @author Sebastien Tourbier
 */
template < class TScalarType, unsigned int NDimensions=3, typename TPixelType = float >
class SliceBySliceTransformBase : public itk::MatrixOffsetTransformBase<TScalarType,NDimensions, NDimensions> //public itk::Transform<TScalarType,NDimensions,NDimensions>
{
public:
    /** Standard class typedefs. */
    typedef SliceBySliceTransformBase  Self;
    //typedef itk::Transform<TScalarType,NDimensions,NDimensions> Superclass;
    typedef itk::MatrixOffsetTransformBase<TScalarType,NDimensions, NDimensions> Superclass;

    typedef itk::MatrixOffsetTransformBase<TScalarType,NDimensions,NDimensions> TransformType;
    typedef typename TransformType::Pointer TransformPointerType;

    typedef itk::Image< TPixelType,NDimensions > ImageType;
    typedef typename ImageType::Pointer ImagePointerType;

    typedef itk::ContinuousIndex<double, NDimensions > ContinuousIndexType;

    typedef itk::SmartPointer< Self >   Pointer;
    typedef itk::SmartPointer< const Self >  ConstPointer;

    //typedef itk::SmartPointer<Superclass> TransformPointer;
    typedef std::vector<TransformPointerType> TransformPointerList;
    typedef typename TransformPointerList::const_iterator TransformPointerListConstIterator;
    typedef std::vector<itk::TransformBase::Pointer> TransformBasePointerList;

    typedef typename Superclass::InputPointType InputPointType;
    typedef typename Superclass::OutputPointType OutputPointType;

    typedef typename Superclass::InputVectorType InputVectorType;
    typedef typename Superclass::OutputVectorType OutputVectorType;

    typedef typename Superclass::InputVnlVectorType InputVnlVectorType;
    typedef typename Superclass::OutputVnlVectorType OutputVnlVectorType;

    typedef typename Superclass::InputCovariantVectorType InputCovariantVectorType;
    typedef typename Superclass::OutputCovariantVectorType OutputCovariantVectorType;

    typedef typename Superclass::JacobianType JacobianType;

    typedef typename Superclass::ParametersType ParametersType;

    /** New method for creating an object using a factory. */
   // itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro( SliceBySliceTransformBase, itk::MatrixOffsetTransformBase );

    /** Method to transform a point. */
    virtual OutputPointType TransformPoint(const InputPointType& p ) const = 0;

    /**  Method to transform a vector. */
    virtual OutputVectorType TransformVector(const InputVectorType& p) const = 0;

    /**  Method to transform a vnl_vector. */
    virtual OutputVnlVectorType TransformVector(const InputVnlVectorType &p ) const = 0;

    /**  Method to transform a CovariantVector. */
    virtual OutputCovariantVectorType TransformCovariantVector( const InputCovariantVectorType &p ) const = 0;

    /** Compute Jacobian (product of component Jacobians). Not thread safe!! */
    virtual const JacobianType & GetJacobian(const InputPointType &p ) const = 0;

    /** NOT IMPLEMENTED: */
    virtual void ComputeJacobianWithRespectToParameters( const InputPointType  &p, JacobianType & jacobian) const = 0;


    /** NOT IMPLEMENTED: */
    virtual void ComputeJacobianWithRespectToPosition(const InputPointType & p, JacobianType &j ) const = 0;

    /** Set the image where the transformation is defined. */
    virtual void SetImage( ImageType * image) = 0;


    /** Get the Transformation Parameters. */
    virtual const ParametersType& GetParameters(void) const = 0;

    virtual void SetParameters( const ParametersType & parameters ) = 0;

    /** Print self */
    void PrintSelf(std::ostream &os, itk::Indent indent) const
    {
      Superclass::PrintSelf(os,indent);
    }

    /** Get the transformation associated to a slice. */
    virtual TransformType * GetSliceTransform( unsigned int i ) const = 0;

    /** Set the transformation parameters for a slice. */
   virtual  void SetSliceParameters( unsigned int i, const ParametersType & parameters ) = 0;

    /** Get the parameters for a slice. */
    virtual ParametersType GetSliceParameters(unsigned int i) = 0;

    /** Initialize with the identity. */
    virtual void Initialize() = 0;

   // /** Initialize with a transform. */
    //virtual void Initialize(TransformType* t) = 0;
//    {
//        typename ImageType::SizeType size = this->m_Image -> GetLargestPossibleRegion().GetSize();

//        m_NumberOfSlices = size[2];
//        m_TransformList.resize(m_NumberOfSlices);

//        TransformPointerType tmpTransform = TransformType::New();
//        m_ParametersPerSlice = tmpTransform -> GetNumberOfParameters();
//        this -> m_Parameters.SetSize( m_NumberOfSlices * m_ParametersPerSlice );

//        InputPointType centerPoint;
//        ContinuousIndexType centerIndex;

//        centerIndex[0] = (size[0]-1)/2.0;
//        centerIndex[1] = (size[1]-1)/2.0;

//        for(unsigned int i=0; i<m_NumberOfSlices; i++)
//        {
//            centerIndex[2] =  i;
//            this->m_Image -> TransformContinuousIndexToPhysicalPoint(centerIndex,centerPoint);

//            m_TransformList[i] = TransformType::New();
//            m_TransformList[i] -> SetIdentity();
//            m_TransformList[i] -> SetCenter(centerPoint);
//            t -> SetCenter( m_TransformList[i] -> GetCenter() );

//            m_TransformList[i] -> SetParameters( t -> GetParameters() );
//        }

//        this -> Modified();
//    }

    /** Set the Fixed Parameters. */
    virtual void SetFixedParameters( const ParametersType & fp ) = 0;

    /** Get the Fixed Parameters. */
    virtual const ParametersType & GetFixedParameters(void) const = 0;

    virtual unsigned int GetNumberOfSlices() = 0;

   // virtual void GetInverse(Self *) const = 0;



  protected:
      /** Default constructor. Otherwise we get a run time warning from itkTransform. */
    SliceBySliceTransformBase() : Superclass( 0 ) {}

    virtual ~SliceBySliceTransformBase(){}

  private:
    /** List of transforms. */
//    TransformPointerList       m_TransformList;
//    ImagePointerType           m_Image;
//    unsigned int 			   m_NumberOfSlices;
//    unsigned int 		       m_ParametersPerSlice;


};
}



#endif
