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

#include "btkGaussianPSF.h"

namespace btk
{

GaussianPSF::GaussianPSF():Superclass::PSF()
{
    m_PsfImage = ImageType::New();

    m_Sigma[0] = m_Spacing[0] * m_Size[0] / 2.3548;
    m_Sigma[1] = m_Spacing[1] * m_Size[1] / 2.3548;
    m_Sigma[2] = m_Spacing[2] * m_Size[2] / 2.3548;

}
//-------------------------------------------------------------------------------------------------
void GaussianPSF::ConstructImage()
{
    ImageType::RegionType region;


    ImageType::IndexType index;

    index[0] = 0;
    index[1] = 0;
    index[2] = 0;

    region.SetSize(m_Size);
    region.SetIndex(index);

    ImageType::SpacingType spc;
    spc[0] = m_Spacing[0];
    spc[1] = m_Spacing[1];
    spc[2] = m_Spacing[2];

    m_PsfImage->SetRegions(region);
    m_PsfImage->SetSpacing(spc);
    m_PsfImage->Allocate();
    m_PsfImage->FillBuffer(0.0);
    double sum = 0;
    double sum2 = 0;

    ImageType::IndexType hrIndex;

    itkIteratorWithIndex itPSF(m_PsfImage,m_PsfImage->GetLargestPossibleRegion());
    itkContinuousIndex hrIndexCenter;
    hrIndexCenter[0] = (m_Size[0]-1)/2.0;
    hrIndexCenter[1] = (m_Size[1]-1)/2.0;
    hrIndexCenter[2] = (m_Size[2]-1)/2.0;

    PointType hrPointCenter;
    m_PsfImage->TransformContinuousIndexToPhysicalPoint(hrIndexCenter,hrPointCenter);

    PointType hrOrigin;
    hrOrigin[0] =  hrPointCenter[0];
    hrOrigin[1] =  hrPointCenter[1];
    hrOrigin[2] =  hrPointCenter[2];
    m_PsfImage->SetOrigin(hrOrigin);

    for(itPSF.GoToBegin(); !itPSF.IsAtEnd(); ++itPSF)
    {
        hrIndex = itPSF.GetIndex();
        float x = hrIndex[0]- hrIndexCenter[0];
        float y = hrIndex[1]- hrIndexCenter[1];
        float z = hrIndex[2]- hrIndexCenter[2];
        // Compute the Gaussian value
        float value = (x*x)/(2*m_Sigma[0]*m_Sigma[0]) + (y*y)/(2*m_Sigma[1]*m_Sigma[1]) + (z*z)/(2*m_Sigma[2]*m_Sigma[2]);
        value = exp(-value);

        itPSF.Set(value);

        sum += itPSF.Get();
    }

    //Normalization
    if(sum>0)
    {
        for(itPSF.GoToBegin(); !itPSF.IsAtEnd(); ++itPSF)
        {
            if(itPSF.Get() > 0)
            {
                itPSF.Set( itPSF.Get() / sum );
            }
        }
    }

}
//-------------------------------------------------------------------------------------------------
void GaussianPSF::PrintSelf(std::ostream &os, itk::Indent indent) const
{
    Superclass::PrintSelf(os,indent);

    os << indent << "Direction: " << m_Direction << std::endl;
    os << indent << "Center: " << m_Center << std::endl;
    os << indent << "Spacing: " << m_Spacing << std::endl;
}

}// end namespace
