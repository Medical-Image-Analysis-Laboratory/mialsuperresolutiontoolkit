/*==========================================================================
  
  © Université de Strasbourg - Centre National de la Recherche Scientifique
  
  Date: 28/08/2013
  Author(s): Julien Pontabry (pontabry (at)unistra.fr )
  
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

#ifndef __BTK_COMMAND_PROGRESS_UPDATE_H__
#define __BTK_COMMAND_PROGRESS_UPDATE_H__

// ITK includes
#include "itkObject.h"
#include "itkEventObject.h"
#include "itkProcessObject.h"
#include "itkCommand.h"


namespace btk
{

/**
 * @brief Define a command tool for a commandline progress bar.
 * @author Julien Pontabry
 * @ingroup InputOutput
 */
class CommandProgressUpdate : public itk::Command
{
    public:
        itkNewMacro(CommandProgressUpdate);

        /**
         * @brief Displays the progress bar on the standard output (std::out).
         * @param caller Object who invoked the itk::ProgressEvent()
         * @param event Event which is invoked
         */
        void Execute(itk::Object *caller, const itk::EventObject &event)
        {
            Execute((const itk::Object *)caller, event);
        }

        /**
         * @brief Displays the progress bar on the standard output (std::out).
         * @param caller Object who invoked the itk::ProgressEvent()
         * @param event Event which is invoked
         */
        void Execute(const itk::Object *caller, const itk::EventObject &event)
        {
            if(!m_Finished)
            {
                unsigned int progress = static_cast< const itk::ProcessObject *>(caller)->GetProgress()*100.0;
                std::cout << "\tProgress: " << progress << "%\r" << std::flush;

                if(progress == 100)
                {
                    std::cout << std::endl;
                    m_Finished = true;
                }
            }
        }

    protected:
        CommandProgressUpdate() : m_Finished(false), itk::Command()
        {
            // ----
        }

    private:
        bool m_Finished;
};

} // namespace btk

#endif // __BTK_COMMAND_PROGRESS_UPDATE_H__
