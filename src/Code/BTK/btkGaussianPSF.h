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

#ifndef BTKGAUSSIANPSF_H
#define BTKGAUSSIANPSF_H

#include "btkPSF.h"

#include "itkFixedArray.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkContinuousIndex.h"

namespace btk
{

/**
 * @class GaussianPSF
 * @brief Gaussian PSF represent a Gaussian PSF, the support is an image.
 *
 * Sigma is computed with the spacing and the size of the image.
 * A common use of this class is :
 * - first construct the image with the correct parameters (center, size, spacing...)
 * - Get the Psf Image previously constructed
 * - iterate over the image to get the values
 *
 * @todo An Evaluate Method for compute the value of a point
 * (for example iterate over the image and break when the iterated point == point we wanted, then return the value of this point )
 * @author Marc Schweitzer
 * @ingroup Maths
 */
class GaussianPSF : public btk::PSF
{
    public:
        typedef btk::GaussianPSF                Self;
        typedef btk::PSF                        Superclass;



        typedef itk::SmartPointer< Self >       Pointer;
        typedef itk::SmartPointer< const Self > ConstPointer;

        typedef Superclass::OutputType          OutputType;
        typedef Superclass::InputType           InputType;
        typedef Superclass::DirectionType       DirectionType;
        typedef Superclass::SpacingType         SpacingType;
        typedef Superclass::PointType           PointType;

        typedef itk::Image < float, 3 >         ImageType;
        typedef itk::ImageRegionIteratorWithIndex< ImageType > itkIteratorWithIndex;
        typedef itk::ContinuousIndex<double,3>     itkContinuousIndex;

        typedef Superclass::SizeType            SizeType;

        /** Array type */
        typedef itk::FixedArray< double,3 > ArrayType;


        /** Method for creation through the object factory. */
        itkNewMacro(Self);

        /** Run-time type information (and related methods). */
        itkTypeMacro(btk::GaussianPSF, btk::PSF);

        /** Set Spacing */
        virtual void SetSpacing(SpacingType _spacing)
        {
          m_Spacing = _spacing.GetVnlVector();
          this->UpdateSigma();
        }
        /** Construct the Gaussian PSF image */
        virtual void ConstructImage();

        /** Compute the Sigma (FWHM is used) */
        virtual void UpdateSigma()
        {
            m_Sigma[0] = m_Spacing[0] * m_Size[0] / 2.3548;
            m_Sigma[1] = m_Spacing[1] * m_Size[1] / 2.3548;
            m_Sigma[2] = m_Spacing[2] * m_Size[2] / 2.3548;
        }


        /** Set Size of the image */
        void SetSize(SizeType _size)
        {
            m_Size =  _size;
            this->UpdateSigma();
        }

    protected:
        /** Constructor */
        GaussianPSF();
        /** Destructor */
        virtual ~GaussianPSF(){}
        /** Print */
        void PrintSelf(std::ostream& os, itk::Indent indent) const;


    private:

        ArrayType m_Sigma; /** Sigma */



};

}//end namespace
#endif // BTKGAUSSIANPSF_H
