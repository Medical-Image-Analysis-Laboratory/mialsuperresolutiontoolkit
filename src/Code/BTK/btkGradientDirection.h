/*==========================================================================
  
  © Université de Strasbourg - Centre National de la Recherche Scientifique
  
  Date: 05/07/2012
  Author(s): Julien Pontabry (pontabry@unistra.fr)
  
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

#ifndef BTK_GRADIENT_DIRECTION_H
#define BTK_GRADIENT_DIRECTION_H

// ITK includes
#include "itkSmartPointer.h"
#include "itkVector.h"
#include "vnl/vnl_vector_fixed.h"

// Local includes
#include "btkMacro.h"
#include "btkSphericalDirection.h"


namespace btk
{

/**
 * @brief Represent a gradient direction corresponding to a gradient image of a diffusion sequence.
 * @author Julien Pontabry
 * @ingroup Diffusion
 */
class GradientDirection : public itk::Vector< double,3 >
{
    public:
        typedef GradientDirection      Self;
        typedef itk::Vector< double,3 > Superclass;

        /**
         * @brief Construct a null gradient direction.
         */
        GradientDirection();

        /**
         * @brief Copy constructor for superclass.
         * @param d Object of superclass type to copy.
         */
        GradientDirection(const Superclass &d);

        /**
         * @brief Constructor.
         * @param x X coordinate.
         * @param y Y coordinate.
         * @param z Z coordinate.
         */
        GradientDirection(double x, double y, double z);

        /**
         * @brief Constructor.
         * @param theta Elevation (in spherical coordinates).
         * @param phi Azimuth (in spherical coordinates).
         */
        GradientDirection(double theta, double phi);

        /**
         * @brief Get the gradient direction in a vnl fixed structure for vector.
         * @return A vnl fixed vector representing the gradient direction.
         */
        virtual vnl_vector_fixed< double,3 > GetVnlVectorFixed();

        /**
         * @brief Update the cartesian coordinates using the spherical ones.
         */
        void UpdateCartesianCoordinates();

        /**
         * @brief Update the sperical coordintes using the cartesian ones.
         */
        void UpdateSphericalCoordinates();

        /**
         * @brief Get if the vector is null.
         * @return True if the vector is null, false otherwise.
         */
        bool IsNull() const;

        btkGetMacro(SphericalDirection, btk::SphericalDirection);

    protected:
        /** Gradient direction in spherical coordinates system. */
        btk::SphericalDirection m_SphericalDirection;
};

} // namespace btk

#endif // BTK_GRADIENT_DIRECTION_H
