/*==========================================================================

  © Université de Strasbourg - Centre National de la Recherche Scientifique

  Date: 24/01/2011
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

#ifndef __mialsrtk_IMAGEREGISTRATIONFILTER_TXX__
#define __mialsrtk_IMAGEREGISTRATIONFILTER_TXX__

#include "mialsrtkImageRegistrationFilter.h"

namespace mialsrtk
{

  /*
   * Constructor
   */
  template < class TImage, class TMask, unsigned int Dimension >
  ImageRegistrationFilter<TImage,TMask,Dimension>
  ::ImageRegistrationFilter()
  {
    m_ItMax = 10;
    m_Epsilon = 1e-4;
    m_Margin = 0.0;
    //m_noreg = false;
  }

  template < class TImage, class TMask, unsigned int Dimension >
  void
  ImageRegistrationFilter<TImage,TMask,Dimension>
  ::AddImage(ImageType* image)
  {
    m_Images.push_back(image);

    // Add transforms for this image
    m_Transforms.resize( m_Transforms.size() + 1 );
    SizeType imageSize = image -> GetLargestPossibleRegion().GetSize();
    m_Transforms[m_Transforms.size()-1].resize( imageSize[2]);
  }

  template < class TImage, class TMask, unsigned int Dimension >
  void
  ImageRegistrationFilter<TImage,TMask,Dimension>
  ::AddMask(MaskType* mask)
  {
    m_Masks.push_back(mask);
  }

  template < class TImage, class TMask, unsigned int Dimension >
  void
  ImageRegistrationFilter<TImage,TMask,Dimension>
  ::SetTransform(TransformType* transform)
  {
    m_Transforms.push_back(transform);
  }

  template < class TImage, class TMask, unsigned int Dimension >
  void
  ImageRegistrationFilter<TImage,TMask,Dimension>
  ::SetMaxIterations(unsigned int iters)
  {
    m_ItMax = iters;
  }

  template < class TImage, class TMask, unsigned int Dimension >
  void
  ImageRegistrationFilter<TImage,TMask,Dimension>
  ::SetEpsilon(double epsilon)
  {
    m_Epsilon = epsilon;
  }

  template < class TImage, class TMask, unsigned int Dimension >
  void
  ImageRegistrationFilter<TImage,TMask,Dimension>
  ::SetMargin(double margin)
  {
    m_Margin = margin;
  }

}

#endif /* mialsrtkImageRegistrationFilter_txx */
