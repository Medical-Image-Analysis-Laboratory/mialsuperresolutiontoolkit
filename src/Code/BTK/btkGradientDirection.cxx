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

#include "btkGradientDirection.h"


// STL includes
#include "cmath"


namespace btk
{

GradientDirection::GradientDirection(double x, double y, double z)
{
    (*this)[0] = x;
    (*this)[1] = y;
    (*this)[2] = z;

    this->UpdateSphericalCoordinates();
}

//----------------------------------------------------------------------------------------

GradientDirection::GradientDirection(const Superclass &d)
{
    (*this)[0] = d[0];
    (*this)[1] = d[1];
    (*this)[2] = d[2];

    this->UpdateSphericalCoordinates();
}

//----------------------------------------------------------------------------------------

GradientDirection::GradientDirection()
{
    // NOTE : this seems not working
    Self::Self(0,0,0);
}

//----------------------------------------------------------------------------------------

GradientDirection::GradientDirection(double theta, double phi) : m_SphericalDirection(theta, phi, 1.0f)
{
    Self::UpdateCartesianCoordinates();
}

//----------------------------------------------------------------------------------------

vnl_vector_fixed< double,3 > GradientDirection::GetVnlVectorFixed()
{
    vnl_vector_fixed< double,3 > v;
    v[0] = (*this)[0];
    v[1] = (*this)[1];
    v[2] = (*this)[2];

    return v;
}

//----------------------------------------------------------------------------------------

void GradientDirection::UpdateCartesianCoordinates()
{
    float theta = this->m_SphericalDirection[0];
    float phi   = this->m_SphericalDirection[1];
    float rau   = this->m_SphericalDirection[2];

    float cosTheta = std::cos(theta);
    float sinTheta = std::sin(theta);
    float   cosPhi = std::cos(phi);
    float   sinPhi = std::sin(phi);

    (*this)[0] = rau * sinTheta * cosPhi;
    (*this)[1] = rau * sinTheta * sinPhi;
    (*this)[2] = rau * cosTheta;
}

//----------------------------------------------------------------------------------------

void GradientDirection::UpdateSphericalCoordinates()
{
    float rau = std::sqrt(
                   (*this)[0] * (*this)[0] +
                   (*this)[1] * (*this)[1] +
                   (*this)[2] * (*this)[2]
                );

    float theta = 0, phi = 0;

    if((*this)[0] != 0 || (*this)[0] != 0) // Not colinear to Z-axis.
    {
        theta = std::acos( (*this)[2] / rau );
        phi   = std::acos(
                    (*this)[0] /
                    std::sqrt(
                        (*this)[0] * (*this)[0] +
                        (*this)[1] * (*this)[1]
                    )
                );

        if((*this)[1] < 0)
            phi = 2.0 * M_PI - phi;
    }

    this->m_SphericalDirection = btk::SphericalDirection(theta, phi, rau);
}

//----------------------------------------------------------------------------------------

bool GradientDirection::IsNull() const
{
    return ( (*this)[0] == 0 && (*this)[1] == 0 && (*this)[2] == 0 );
}

} // namespace btk
