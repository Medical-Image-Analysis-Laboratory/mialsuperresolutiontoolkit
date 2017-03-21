/*==========================================================================

  © Université de Strasbourg - Centre National de la Recherche Scientifique

  Date: 16/04/2012
  Author(s): Marc Schweitzer (marc.schweitzer(at)unistra.fr)
             Julien Pontabry (pontabry@unistra.fr)

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

#ifndef BTK_FILE_HELPER_H
#define BTK_FILE_HELPER_H

#include "iostream"
#include "vector"
#include "fstream"

namespace btk
{
/**
 * @class FileHelper
 * @brief The FileHelper is a helper class for handle file (file exist, print ...)
 * @ingroup Tools
 * @author Julien Pontabry, Marc Schweitzer
 */
class FileHelper
{
    public:

        /**
         * @brief FileExist
         * @param file
         * @return boolean if file exist
         */
        static bool FileExist(const std::string &file);

        /**
         * @brief FileExistPrintIt
         * @param file
         * @return boolean
         */
        static bool FileExistPrintIt(const std::string &file);

        /**
         * @brief FilesExist
         * @param files
         * @param result (vector of boolean)
         */
        static void FilesExist(std::vector< std::string > &files, std::vector< bool > &result);

        /**
         * @brief FilesExistPrintIt
         * @param files
         * @param result (vector of boolean)
         */
        static void FilesExistPrintIt(std::vector< std::string > &files, std::vector< bool > &result);

        /**
         * @brief Extract the radix filename.
         * Exemple: With filename equal to "xxx.yy.zz" or "xxx.yy", this function returns "xxx".
         * @param filename Filename
         * @return Radix of the filename
         * @ingroup InputOutput
         */
        static std::string GetRadixOf(std::string filename);

        /**
         * @brief Extract the extension of the filename.
         * Exemple: With filename equal to "xxx.yy.zz" or "xxx.yy", this function returns ".yy.zz" or ".yy".
         * @param filename Filename
         * @return Extension of the filename
         * @ingroup InputOutput
         */
        static std::string GetExtensionOf(std::string filename);

    private:

        /**
         * @brief Get the position of the first point of the extension
         * @param filename Filename
         * @return Positopn of the first point of the extension
         * @ingroup InputOutput
         */
        static unsigned int GetExtensionPosition(std::string filename);
};

} // namespace btk


#endif // BTK_FILE_HELPER_H
