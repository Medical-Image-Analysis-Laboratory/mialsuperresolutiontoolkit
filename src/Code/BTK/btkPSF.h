/*==========================================================================
  
  © Université de Strasbourg - Centre National de la Recherche Scientifique
  
  Date: 27/05/2013
  Author(s):Marc Schweitzer (marc.schweitzer(at)unistra.fr)
  
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

#ifndef BTKPSF_H
#define BTKPSF_H

#include "itkObject.h"
#include "itkObjectFactory.h"
#include "itkFixedArray.h"
#include "itkPoint.h"
#include "itkMatrix.h"
#include "itkVector.h"
#include "itkSize.h"
#include "itkImage.h"

#include "btkMacro.h"

#include "vnl/vnl_vector.h"
#include "vnl/vnl_math.h"

namespace btk
{
/**
 * @class PSF
 * @brief PSF is a base class for PSF images (gaussian, boxcar...)
 * ! PSF is a pure virtual class, it can not be instancied !
 * @ingroup Maths
 * @author Marc Schweitzer
 */
class PSF: public itk::Object
{
    public:
    /** Typedefs */
        typedef btk::PSF                        Self;
        typedef itk::Object                     Superclass;

        typedef itk::SmartPointer< Self >       Pointer;
        typedef itk::SmartPointer< const Self > ConstPointer;
        typedef double                          OutputType;
        typedef itk::Point< double, 3 >         InputType;

        /** Direction type. */
        typedef itk::Matrix< double,3,3 >  DirectionType;

        /** Point type */
        typedef itk::Point< double,3 >  PointType;

        /** Spacing type */
        typedef itk::Vector< double,3 >  SpacingType;

        typedef itk::Image< float, 3 >  ImageType;
        typedef ImageType::IndexType    IndexType;

        typedef itk::Size < 3 >         SizeType;

        /** Method for creation through the object factory. */
        //itkNewMacro(Self);

        /** Run-time type information (and related methods). */
        itkTypeMacro(btk::PSF, itk::Object);

        /** Construct the PSF image (this method should be redefined in subclasses) */
        virtual void ConstructImage() = 0;

        /**
         * @brief SetCenter
         * @param center
         */
        virtual void SetCenter(PointType center)
        {
            m_Center = center.GetVnlVector();
            PointType origin, newOrigin,oldCenter,dist;
            IndexType centerOfImage;
            SizeType sizeOfImage;
            origin = m_PsfImage->GetOrigin();
            //sizeOfImage = m_PsfImage->GetLargestPossibleRegion().GetSize();
//            centerOfImage[0] = (sizeOfImage[0]-1)/2.0;
//            centerOfImage[1] = (sizeOfImage[1]-1)/2.0;
//            centerOfImage[2] = (sizeOfImage[2]-1)/2.0;

            centerOfImage[0] = (m_Size[0]-1)/2.0;
            centerOfImage[1] = (m_Size[1]-1)/2.0;
            centerOfImage[2] = (m_Size[2]-1)/2.0;



            m_PsfImage->TransformIndexToPhysicalPoint(centerOfImage,oldCenter);
            //std::cout<<"origin : "<<origin<<" center : "<<oldCenter<<std::endl;

            dist[0] = center[0] - oldCenter[0];
            dist[1] = center[1] - oldCenter[1];
            dist[2] = center[2] - oldCenter[2];

            newOrigin[0] = origin[0] + dist[0];
            newOrigin[1] = origin[1] + dist[1];
            newOrigin[2] = origin[2] + dist[2];

            //std::cout<<"new origin : "<<newOrigin<<" new center : "<<center<<std::endl;

            m_PsfImage->SetOrigin(newOrigin);

        }

        /**
         * @brief SetDirection
         * @param direction
         */
        virtual void SetDirection(DirectionType direction)
        {
            m_Direction = direction.GetVnlMatrix();
            m_idir = m_Direction.get_column(0);
            m_jdir = m_Direction.get_column(1);
            m_kdir = m_Direction.get_column(2);

        }
        /**
         * @brief SetSpacing
         * @param spacing
         */
        virtual void SetSpacing(SpacingType spacing)
        {
          m_Spacing = spacing.GetVnlVector();
        }
        /**
         * @brief SetSize
         * @param _size
         */
        virtual void SetSize(SizeType _size)
        {
            m_Size = _size;
        }
        /**
         * @brief SetLrSpacing : set the low resolution spacing
         * @param _spc
         */
        void SetLrSpacing(SpacingType& _spc)
        {
            m_LrSpacing = _spc;
        }

        /**
         * @brief btkGetMacro
         */
        btkGetMacro(PsfImage,ImageType::Pointer);

    protected:
        /** Constructor */
        PSF();
        /** Destructor */
        virtual ~PSF(){}
        void PrintSelf(std::ostream& os, itk::Indent indent) const;

        vnl_matrix< double > m_Direction;
        vnl_vector< double > m_Center;
        vnl_vector< double > m_Spacing;
        SizeType             m_Size;


        vnl_vector<double> m_idir;
        vnl_vector<double> m_jdir;
        vnl_vector<double> m_kdir;

        ImageType::Pointer m_PsfImage;

        SpacingType m_LrSpacing;





};

}
#endif // BTKPSF_H
