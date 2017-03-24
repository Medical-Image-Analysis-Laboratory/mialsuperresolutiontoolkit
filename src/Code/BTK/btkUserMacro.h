/*==========================================================================

  © Université de Strasbourg - Centre National de la Recherche Scientifique

  Date: 27/01/2011
  Author(s): Estanislao Oubel (oubel@unistra.fr)

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

#ifndef __btkUserMacro_h
#define __btkUserMacro_h

#include "itkWin32Header.h"
#include "itkConfigure.h"

#include <string>

/** Set built-in type.  Creates member Set"name"() (e.g., SetVisibility()); */
#define itkSetMacroNoDeb(name,type) \
  virtual void Set##name (const type _arg) \
  { \
    if (this->m_##name != _arg) \
      { \
      this->m_##name = _arg; \
      this->Modified(); \
      } \
  }

/** Get built-in type.  Creates member Get"name"() (e.g., GetVisibility()); */
#define itkGetMacroNoDeb(name,type) \
  virtual type Get##name () \
  { \
    return this->m_##name; \
  }

/** Set built-in type.  Creates member Set"name"() (e.g., SetVisibility()); */
#define UserSetMacro(name,type) \
  virtual void Set##name (int i, const type _arg) \
  { \
    itkDebugMacro("setting " #name " to " << _arg); \
    if (this->m_##name[i] != _arg) \
      { \
      this->m_##name[i] = _arg; \
      this->Modified(); \
      } \
  }

/** Set pointer to object; uses Object reference counting methodology.
 * Creates method Set"name"() (e.g., SetPoints()). Note that using
 * smart pointers requires using real pointers when setting input,
 * but returning smart pointers on output. */
#define UserSetObjectMacro(name,type) \
  virtual void Set##name (int i, type* _arg ) \
  { \
    if (this->m_##name[i] != _arg) \
      { \
      this->m_##name[i] = _arg; \
      this->Modified(); \
      } \
  }

/** Get a smart pointer to an object.  Creates the member
 * Get"name"() (e.g., GetPoints()). */
#define UserGetObjectMacro(name,type) \
  virtual type * Get##name (int i) \
  { \
    itkDebugMacro("returning " #name " address " << this->m_##name[i] ); \
    return this->m_##name[i].GetPointer(); \
  }

/** Set const pointer to object; uses Object reference counting methodology.
 * Creates method Set"name"() (e.g., SetPoints()). Note that using
 * smart pointers requires using real pointers when setting input,
 * but returning smart pointers on output. */
#define UserSetConstObjectMacro(name,type) \
  virtual void Set##name (int i, const type* _arg) \
  { \
    itkDebugMacro("setting " << #name " to " << _arg ); \
    if (this->m_##name[i] != _arg) \
      { \
      this->m_##name[i] = _arg; \
      this->Modified(); \
      } \
  }


/** Get a smart const pointer to an object.  Creates the member
 * Get"name"() (e.g., GetPoints()). */
#define UserGetConstObjectMacro(name,type) \
  virtual const type * Get##name (int i) const \
  { \
    itkDebugMacro("returning " #name " address " << this->m_##name[i] ); \
    return this->m_##name[i].GetPointer(); \
  }

/** Get a const reference to a smart pointer to an object.
 * Creates the member Get"name"() (e.g., GetPoints()). */
#define UserGetConstReferenceObjectMacro(name,type) \
  virtual const typename type::Pointer & Get##name (int i) const \
  { \
    itkDebugMacro("returning " #name " address " << this->m_##name[i] ); \
    return this->m_##name[i]; \
  }

#endif
