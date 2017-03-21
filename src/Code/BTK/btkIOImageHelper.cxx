/*==========================================================================
  
  © Université de Strasbourg - Centre National de la Recherche Scientifique
  
  Date: 02/10/2012
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
#include "btkIOImageHelper.h"

namespace btk
{
IOImageHelper::ScalarType
IOImageHelper::GetComponentTypeOfImageFile(const std::string &_inputFile)
{
    ScalarType Type;

    itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(_inputFile.c_str(), itk::ImageIOFactory::ReadMode);

    imageIO->SetFileName(_inputFile);
    imageIO->ReadImageInformation();
    ScalarPixelType pixelType = imageIO->GetComponentType();

    switch (pixelType)
    {
        case itk::ImageIOBase::SHORT:
            Type = Short;
            break;
        case itk::ImageIOBase::USHORT:
            Type = UShort;
            break;
        case itk::ImageIOBase::FLOAT:
            Type = Float;
            break;
        case itk::ImageIOBase::DOUBLE:
            Type = Double;
            break;
        case itk::ImageIOBase::CHAR:
            Type = Char;
            break;
        case itk::ImageIOBase::UCHAR:
            Type = UChar;
            break;
        case itk::ImageIOBase::INT:
            Type = Int;
            break;
        case itk::ImageIOBase::UINT:
            Type = UInt;
            break;

        default:
            btkException("Pixel Type not supported. Exiting.");
            break;
    }

    return Type;
}
}
