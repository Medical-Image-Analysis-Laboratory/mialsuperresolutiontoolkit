/*==========================================================================
  
  © Université de Strasbourg - Centre National de la Recherche Scientifique
  
  Date:  02/10/2012
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

#ifndef BTK_IOIMAGEHELPER_H
#define BTK_IOIMAGEHELPER_H

/* ITK */
#include "itkImageIOBase.h"
#include "itkImageIOFactory.h"

/* BTK */
#include "btkMacro.h"
/* STL */
#include "string"

namespace btk
{
/**
 * @class IOImageHelper
 * @brief Helper class for Image IO (for example read the pixel type of a unknown image)
 * @author Marc Schweitzer
 * @ingroup Tools
 */
class IOImageHelper
{

    public:
        /** @brief itk::ImageIOBase ComponentType, it is a enumeration of different type */
        typedef itk::ImageIOBase::IOComponentType ScalarPixelType;

        /** @brief enumeration of type used in btk */
        enum ScalarType
        {
            Float = 0,
            Short,
            UShort,
            Char,
            UChar,
            Double,
            Int,
            UInt
        };

        /**
         * @brief Get the pixel type of a unknown image
         * @param inputFile Input file name of the image to read (string)
         * @return a ScalarType (enumeration) (Float, Double...)
         */
        static ScalarType GetComponentTypeOfImageFile(const std::string & inputFile);




};

}


#endif // BTK_IOIMAGEHELPER_H
