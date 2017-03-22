/*==========================================================================
  
  © Université de Strasbourg - Centre National de la Recherche Scientifique
  
  Date: 28/05/2013
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

#ifndef BTKBOXCARPSF_H
#define BTKBOXCARPSF_H



#include "btkPSF.h"
#include "btkMacro.h"

#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkContinuousIndex.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"



namespace btk
{
/** @class BoxCarPSF
 * @brief BoxCarPSF is used for represent a BoxCar PSF, the support of the PSF is an image.
 *
 * A common use of this class is :
 * - first construct the image with the correct parameters (center, size, spacing...)
 * - Get the Psf Image previously constructed
 * - iterate over the image to get the values
 *
 * NOTE : The Evaluate Method is currently not correct.
 * TODO : An Evaluate Method for compute the value of a point
 * (for example iterate over the image and break when the iterated point == point we wanted, then return the value of this point )
 *@author Marc Schweitzer
 *@ingroup Maths
 */
class BoxCarPSF : public btk::PSF
{
    public:
        /** Typedefs */
        typedef btk::BoxCarPSF                      Self;
        typedef btk::PSF                            Superclass;

        typedef itk::SmartPointer< Self >           Pointer;
        typedef itk::SmartPointer< const Self >     ConstPointer;

        typedef Superclass::OutputType              OutputType;
        typedef Superclass::InputType               InputType;
        typedef Superclass::DirectionType           DirectionType;
        typedef Superclass::SpacingType             SpacingType;
        typedef Superclass::PointType               PointType;
        typedef Superclass::SizeType                SizeType;

        typedef itk::Image < float, 3 >         ImageType;
        typedef itk::ImageRegionIteratorWithIndex< ImageType > itkIteratorWithIndex;
        typedef itk::ContinuousIndex<double,3>     itkContinuousIndex;
        typedef itk::BSplineInterpolateImageFunction<ImageType, double, double>  itkBSplineInterpolator;
        typedef itk::LinearInterpolateImageFunction< ImageType,double> itkLinearInterpolator;
        /** Method for creation through the object factory. */
        itkNewMacro(Self);

        /** Run-time type information (and related methods). */
        itkTypeMacro(btk::BoxCarPSF, btk::PSF);

        /** Construct Image */
        virtual void ConstructImage();


    protected:
        /** Constructor */
        BoxCarPSF();
        /** Destructor */
        virtual ~BoxCarPSF(){}
        /** Print */
        void PrintSelf(std::ostream& os, itk::Indent indent) const;


};

}//end namespace

#endif // BTKBOXCARPSF_H
