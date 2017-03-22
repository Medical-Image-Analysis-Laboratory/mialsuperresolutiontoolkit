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

#include "btkBoxCarPSF.h"

namespace btk
{

BoxCarPSF::BoxCarPSF(): Superclass::PSF()
{

    m_PsfImage = ImageType::New();
}
//-------------------------------------------------------------------------------------------------
void BoxCarPSF::ConstructImage()
{


    ImageType::Pointer lrImage = ImageType::New();
    ImageType::RegionType region, HrRegion;


    ImageType::SizeType size;
    size[0] = 3;
    size[1] = 3;
    size[2] = 3;

    ImageType::IndexType index;

    //size = _size;

    index[0] = 0;
    index[1] = 0;
    index[2] = 0;

    //region.SetSize(m_Size);
    region.SetSize(size);
    region.SetIndex(index);



    lrImage->SetRegions(region);
    lrImage->SetSpacing(m_LrSpacing);
    lrImage->Allocate();
    lrImage->FillBuffer(0.0);
    ImageType::IndexType lrIndexCenter;
    lrIndexCenter[0] = (size[0]-1)/2.0;
    lrIndexCenter[1] = (size[1]-1)/2.0;
    lrIndexCenter[2] = (size[2]-1)/2.0;
    ImageType::PointType lrPointCenter;
    lrImage->TransformIndexToPhysicalPoint(lrIndexCenter,lrPointCenter);
    lrImage->SetPixel(lrIndexCenter,1.0);

    ImageType::SpacingType spc;
    spc[0] = m_Spacing[0];
    spc[1] = m_Spacing[1];
    spc[2] = m_Spacing[2];
    HrRegion.SetSize(m_Size);
    HrRegion.SetIndex(index);


    m_PsfImage->SetRegions(HrRegion);
    m_PsfImage->SetSpacing(spc);
    m_PsfImage->Allocate();
    m_PsfImage->FillBuffer(0.0);

    double sum = 0;

    ImageType::IndexType hrIndex;

    itkIteratorWithIndex itPSF(m_PsfImage,m_PsfImage->GetLargestPossibleRegion());
    itkContinuousIndex hrIndexCenter;
    //ImageType::IndexType hrIndexCenter;
    hrIndexCenter[0] = (m_Size[0]-1)/2;
    hrIndexCenter[1] = (m_Size[1]-1)/2;
    hrIndexCenter[2] = (m_Size[2]-1)/2;
    //m_PsfImage->SetPixel(hrIndexCenter,1.0);

    PointType hrPointCenter;
    m_PsfImage->TransformContinuousIndexToPhysicalPoint(hrIndexCenter,hrPointCenter);

    PointType hrOrigin;
    hrOrigin[0] = lrPointCenter[0] - hrPointCenter[0];
    hrOrigin[1] = lrPointCenter[1] - hrPointCenter[1];
    hrOrigin[2] = lrPointCenter[2] - hrPointCenter[2];
    m_PsfImage->SetOrigin(hrOrigin);

    itkBSplineInterpolator::Pointer bsInterpolator = itkBSplineInterpolator::New();
    bsInterpolator->SetSplineOrder(0);
    bsInterpolator->SetInputImage(lrImage);
    unsigned int nbSamples = 10;
    for(itPSF.GoToBegin(); !itPSF.IsAtEnd(); ++itPSF)
    {
        sum = 0.0;
        hrIndex = itPSF.GetIndex();
        PointType hrPoint;
        m_PsfImage->TransformIndexToPhysicalPoint(hrIndex,hrPoint);
        //Continuous coordinate in LR image
        itkContinuousIndex hrContIndex;
        itkContinuousIndex lrContIndex;
        lrImage->TransformPhysicalPointToContinuousIndex(hrPoint,lrContIndex);
        int x,y,z;

        for(z=0; z<nbSamples; z++)
          for(y=0; y<nbSamples; y++)
            for(x=0; x<nbSamples; x++)
            {

              hrContIndex[0] = hrIndex[0] - 0.5 + 1.0*x/nbSamples;
              hrContIndex[1] = hrIndex[1] - 0.5 + 1.0*y/nbSamples;
              hrContIndex[2] = hrIndex[2] - 0.5 + 1.0*z/nbSamples;

              //Coordinate in physical space
              m_PsfImage->TransformContinuousIndexToPhysicalPoint(hrContIndex,hrPoint);

              //Continuous coordinate in LR image
              lrImage->TransformPhysicalPointToContinuousIndex(hrPoint,lrContIndex);

              //std::cout<<lrContIndex<<std::endl;

              sum += bsInterpolator->EvaluateAtContinuousIndex(lrContIndex);
            }


        itPSF.Set(sum);


    }
    //Normalization of the PSF
    sum = 0;
    for(itPSF.GoToBegin(); !itPSF.IsAtEnd(); ++itPSF) sum += itPSF.Get();
    if(sum>0)
      for(itPSF.GoToBegin(); !itPSF.IsAtEnd(); ++itPSF)
      {
        itPSF.Set( itPSF.Get() / sum );
      }

    //for memory saving, we limit the number of non-null voxels
    for(itPSF.GoToBegin(); !itPSF.IsAtEnd(); ++itPSF)
    {
      if(itPSF.Get() < 0.01)
        itPSF.Set(0);
    }
    sum = 0;
    for(itPSF.GoToBegin(); !itPSF.IsAtEnd(); ++itPSF) sum += itPSF.Get();
    if(sum>0)
      for(itPSF.GoToBegin(); !itPSF.IsAtEnd(); ++itPSF)
      {
          if(itPSF.Get() != 0)
          {
              itPSF.Set( itPSF.Get() / sum );
          }

      }


    hrOrigin[0] = 0;
    hrOrigin[1] = 0;
    hrOrigin[2] = 0;
    m_PsfImage->SetOrigin(hrOrigin);




}
//-------------------------------------------------------------------------------------------------
void BoxCarPSF::PrintSelf(std::ostream &os, itk::Indent indent) const
{
    Superclass::PrintSelf(os,indent);

    os << indent << "Direction: " << m_Direction << std::endl;
    os << indent << "Center: " << m_Center << std::endl;
    os << indent << "Spacing: " << m_Spacing << std::endl;
}
//-------------------------------------------------------------------------------------------------

}
