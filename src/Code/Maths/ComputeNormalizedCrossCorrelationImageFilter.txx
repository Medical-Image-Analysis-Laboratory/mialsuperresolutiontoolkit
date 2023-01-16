/*=========================================================================

Copyright (c) 2017 Medical Image Analysis Laboratory (MIAL), Lausanne
  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
==========================================================================*/

#ifndef _ITK_ComputeNormalizedCrossCorrelation_TXX_
#define _ITK_ComputeNormalizedCrossCorrelation_TXX_

#define MIN(a,b) ((a) < (b) ? (a) : (b))

#include "ComputeNormalizedCrossCorrelationImageFilter.h"

#include <cmath> //sqrt() fabs()

#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIterator.h"
#include "itkConstNeighborhoodIterator.h"

namespace itk
{

template<  typename TInputImage, typename TOutputImage >
void
ComputeNormalizedCrossCorrelationImageFilter< TInputImage,TOutputImage >
::Initialize()
{
    this->AllocateOutputs();
    this->GetOutput()->FillBuffer(0);

    // XXX GORTHI: Added!
    if(!m_ComputationMask)
        this->CreateComputationMaskFromInputs();

    this->InitializeComputationRegionFromMask();
}


template<  typename TInputImage, typename TOutputImage >
void
ComputeNormalizedCrossCorrelationImageFilter< TInputImage,TOutputImage >
::GenerateData()
{
    //0 find the number of labels and allocate the output vector image
    this->Initialize();

    if(m_ComputeMetricAtSingleVoxel == false)
    {
        this->InitializeSplitRegionsFromMask();
        this->ThreadOnRegions();
    }
    else // ==> m_ComputeMetricAtSingleVoxel == true
    {
        typename ConstNeighborhoodIterator<TInputImage>::RadiusType radiusS;
        typename ConstNeighborhoodIterator<TInputImage>::OffsetType offset;

        radiusS[0]=m_BlockHalfSizeX;
        radiusS[1]=m_BlockHalfSizeY;
        radiusS[2]=m_BlockHalfSizeZ;

        ConstNeighborhoodIterator<TInputImage>  target = ConstNeighborhoodIterator<TInputImage>(radiusS,this->GetInput(0), this->GetInput(0)->GetLargestPossibleRegion());
        ConstNeighborhoodIterator<TInputImage>  atlas  = ConstNeighborhoodIterator<TInputImage>(radiusS,this->GetInput(1), this->GetInput(1)->GetLargestPossibleRegion());

        unsigned int N = target.Size();

        double corr=0.0;
        std::vector<double> targetImage;
        std::vector<double> atlasImage;

        double targetVal, atlasVal;
        double meanTarget  = 0.0;
        double meanAtlas   = 0.0;

        target.SetLocation(m_VoxelIndex);
        atlas.SetLocation(m_VoxelIndex);

        for(int xx = (-1.0*m_BlockHalfSizeX); xx < (int)m_BlockHalfSizeX+1; ++xx)
        {
            offset[0] = xx;
            for(int yy = (-1.0*m_BlockHalfSizeY); yy < (int)m_BlockHalfSizeY+1; ++yy)
            {
                offset[1] = yy;
                for(int zz = (-1.0*m_BlockHalfSizeZ); zz < (int)m_BlockHalfSizeZ+1; ++zz)
                {
                    offset[2] = zz;

                    targetVal  = target.GetPixel(offset);
                    atlasVal   = atlas.GetPixel(offset);

                    meanTarget += targetVal;
                    meanAtlas  += atlasVal;

                    targetImage.push_back(targetVal);
                    atlasImage.push_back(atlasVal);
                }
            }
        }
        meanTarget /= N;
        meanAtlas  /= N;

        double nr          = 0.0;
        double sigmaTarget = 0.0;
        double sigmaAtlas  = 0.0;
        
        for(unsigned int num = 0; num < N; ++num)
        {
            nr          += ( (targetImage[num]-meanTarget) * (atlasImage[num]-meanAtlas) );
            sigmaTarget += ( (targetImage[num]-meanTarget) * (targetImage[num]-meanTarget) );
            sigmaAtlas  += ( (atlasImage[num]-meanAtlas)   * (atlasImage[num]-meanAtlas) );
        }

        sigmaTarget = sqrt(sigmaTarget/N);
        sigmaAtlas  = sqrt(sigmaAtlas/N);

        // XXX GORTHI Modified
        //  if( (sigmaTarget > m_SigmaThreshold) && (sigmaAtlas > m_SigmaThreshold))
        if( (sigmaTarget != 0) && (sigmaAtlas != 0))
        {
            corr = nr / (N*sigmaTarget*sigmaAtlas);                
        } else
        {
            // XXX GORTHI Modified
            if( fabs(sigmaTarget-sigmaAtlas) < 1e-6 )
                corr = 1.0; // maximum correlation
            else
                corr = 0.0; // no correlation
            //corr = 1.0;
        }
        m_metricValueAtCentralVoxel = corr;
    }
}


template<  typename TInputImage, typename TOutputImage >
void
ComputeNormalizedCrossCorrelationImageFilter< TInputImage,TOutputImage >
::ThreadOnRegions()
{
    itk::MultiThreader::Pointer threaderEstep = itk::MultiThreader::New();

    ThreadStruct *tmpStr = new ThreadStruct;
    tmpStr->Filter = this;
    unsigned int actualNumberOfThreads = MIN((unsigned int)this->GetNumberOfThreads(),this->GetActualNumberOfThreads());
    threaderEstep->SetNumberOfThreads(actualNumberOfThreads);
    threaderEstep->SetSingleMethod(this->CallThreadOnRegions,tmpStr);
    threaderEstep->SingleMethodExecute();

    delete tmpStr;
}


template<  typename TInputImage, typename TOutputImage >
ITK_THREAD_RETURN_TYPE
ComputeNormalizedCrossCorrelationImageFilter< TInputImage,TOutputImage >
::CallThreadOnRegions(void *arg)
{
    MultiThreader::ThreadInfoStruct *threadArgs = (MultiThreader::ThreadInfoStruct *)arg;

    unsigned int nbThread = threadArgs->ThreadID;
    unsigned int nbProcs = threadArgs->NumberOfThreads;

    ThreadStruct *tmpStr = (ThreadStruct *)threadArgs->UserData;
    InputImageRegionType threadRegion;
    unsigned int total = tmpStr->Filter->SplitRequestedRegion(nbThread,nbProcs,threadRegion);

    if (nbThread < total)
        tmpStr->Filter->ThreadOnRegions(threadRegion);
    return NULL;
}


template<  typename TInputImage, typename TOutputImage >
void
ComputeNormalizedCrossCorrelationImageFilter< TInputImage,TOutputImage >
::ThreadOnRegions(OutputImageRegionType &region)
{

    typename ConstNeighborhoodIterator<TInputImage>::RadiusType radiusS;
    typename ConstNeighborhoodIterator<TInputImage>::OffsetType offset;

    radiusS[0]=m_BlockHalfSizeX;
    radiusS[1]=m_BlockHalfSizeY;
    radiusS[2]=m_BlockHalfSizeZ;

    ConstNeighborhoodIterator<TInputImage>  target = ConstNeighborhoodIterator<TInputImage>(radiusS,this->GetInput(0), region);
    ConstNeighborhoodIterator<TInputImage>  atlas  = ConstNeighborhoodIterator<TInputImage>(radiusS,this->GetInput(1), region);

    unsigned int N = target.Size();

    ImageRegionConstIterator<MaskImageType> mask   = ImageRegionConstIterator<MaskImageType>(m_ComputationMask, region);
    ImageRegionIterator<TOutputImage>       output = ImageRegionIterator<TOutputImage>( this->GetOutput(), region);

    while (!output.IsAtEnd())
    {
        double corr=-10.0;
        if (mask.Get()!=0)
        {
            std::vector<double> targetImage;
            std::vector<double> atlasImage;

            double targetVal, atlasVal;
            double meanTarget  = 0.0;
            double meanAtlas   = 0.0;

            for(int xx = (-1.0*m_BlockHalfSizeX); xx < (int)m_BlockHalfSizeX+1; ++xx)
            {
                offset[0] = xx;
                for(int yy = (-1.0*m_BlockHalfSizeY); yy < (int)m_BlockHalfSizeY+1; ++yy)
                {
                    offset[1] = yy;
                    for(int zz = (-1.0*m_BlockHalfSizeZ); zz < (int)m_BlockHalfSizeZ+1; ++zz)
                    {
                        offset[2] = zz;

                        targetVal  = target.GetPixel(offset);
                        atlasVal   = atlas.GetPixel(offset);

                        meanTarget += targetVal;
                        meanAtlas  += atlasVal;

                        targetImage.push_back(targetVal);
                        atlasImage.push_back(atlasVal);
                    }
                }
            }
            meanTarget /= N;
            meanAtlas  /= N;

            double nr          = 0.0;
            double sigmaTarget = 0.0;
            double sigmaAtlas  = 0.0;
            
            for(unsigned int num = 0; num < N; ++num)
            {
                nr          += ( (targetImage[num]-meanTarget) * (atlasImage[num]-meanAtlas) );
                sigmaTarget += ( (targetImage[num]-meanTarget) * (targetImage[num]-meanTarget) );
                sigmaAtlas  += ( (atlasImage[num]-meanAtlas)   * (atlasImage[num]-meanAtlas) );
            }

            sigmaTarget = sqrt(sigmaTarget/N);
            sigmaAtlas  = sqrt(sigmaAtlas/N);

            // XXX GORTHI Modified
            //  if( (sigmaTarget > m_SigmaThreshold) && (sigmaAtlas > m_SigmaThreshold))
            if( (sigmaTarget != 0) && (sigmaAtlas != 0))
            {
                corr = nr / (N*sigmaTarget*sigmaAtlas);                
            } else
            {
                // XXX GORTHI Modified
                //  if( fabs(sigmaTarget-sigmaAtlas) < 1e-6 )
                //      corr = 1.0; // maximum correlation
                //  else
                //      corr = 0.0; // no correlation
                corr = 1.0;
            }
        }
        output.Set(corr);
        ++atlas;
        ++target;
        ++output;
        ++mask;
    }
}


template<  typename TInputImage, typename TOutputImage >
void
ComputeNormalizedCrossCorrelationImageFilter< TInputImage,TOutputImage >
::ThreadOnNumbers()
{
    itk::MultiThreader::Pointer threader = itk::MultiThreader::New();

    ThreadStruct *tmpStr = new ThreadStruct;
    tmpStr->Filter = this;

    unsigned int actualNumberOfThreads = MIN((unsigned int)this->GetNumberOfThreads(),this->GetNumberOfInputs());

    threader->SetNumberOfThreads(actualNumberOfThreads);
    threader->SetSingleMethod(this->CallThreadOnNumbers,tmpStr);
    threader->SingleMethodExecute();

    delete tmpStr;
}


template<  typename TInputImage, typename TOutputImage >
ITK_THREAD_RETURN_TYPE
ComputeNormalizedCrossCorrelationImageFilter< TInputImage,TOutputImage >
::CallThreadOnNumbers(void *arg)
{
    MultiThreader::ThreadInfoStruct *threadArgs = (MultiThreader::ThreadInfoStruct *)arg;

    unsigned int nbThread = threadArgs->ThreadID;
    unsigned int nbProcs = threadArgs->NumberOfThreads;

    ThreadStruct *tmpStr = (ThreadStruct *)threadArgs->UserData;
    unsigned int nbN = tmpStr->Filter->GetNumberOfInputs();

    unsigned int minN = (unsigned int)floor((double)nbThread*nbN/nbProcs);
    unsigned int maxN = (unsigned int)floor((double)(nbThread + 1.0)*nbN/nbProcs);

    maxN = MIN(nbN,maxN);
    tmpStr->Filter->ThreadOnNumbers(minN,maxN);

    return NULL;
}


template<  typename TInputImage, typename TOutputImage >
void
ComputeNormalizedCrossCorrelationImageFilter< TInputImage,TOutputImage >
::ThreadOnNumbers(unsigned int minN, unsigned int maxN)
{

}


template<  typename TInputImage, typename TOutputImage >
void
ComputeNormalizedCrossCorrelationImageFilter< TInputImage,TOutputImage >
::InitializeSplitRegionsFromMask()
{
    if (!m_ComputationMask)
        this->CreateComputationMaskFromInputs();

    typedef ImageRegionIteratorWithIndex< MaskImageType > MaskRegionIteratorType;

    MaskRegionIteratorType maskItr(m_ComputationMask,this->GetOutput()->GetRequestedRegion());
    maskItr.GoToBegin();

    bool isImage3D = (this->GetOutput()->GetRequestedRegion().GetSize()[2] > 1);

    if (this->GetNumberOfThreads() == 1)
    {
        m_lowerLimits.clear();
        m_upperLimits.clear();

        m_lowerLimits.push_back(this->GetOutput()->GetRequestedRegion().GetIndex()[1 + isImage3D]);
        m_upperLimits.push_back(this->GetOutput()->GetRequestedRegion().GetSize()[1 + isImage3D] + this->GetOutput()->GetRequestedRegion().GetIndex()[1 + isImage3D] - 1);

        return;
    }

    if ((unsigned int)this->GetNumberOfThreads() > this->GetOutput()->GetRequestedRegion().GetSize()[1 + isImage3D])
    {
        m_lowerLimits.clear();
        m_upperLimits.clear();

        for (unsigned int i = 0;i < this->GetOutput()->GetRequestedRegion().GetSize()[1 + isImage3D];++i)
        {
            m_lowerLimits.push_back(i);
            m_upperLimits.push_back(i);
        }

        return;
    }

    unsigned int nbPts = 0;
    while (!maskItr.IsAtEnd())
    {
        if (maskItr.Get() != 0)
            nbPts++;
        ++maskItr;
    }

    unsigned int approxNumPtsPerThread = (unsigned int)floor(nbPts/((double)this->GetNumberOfThreads()));

    maskItr.GoToBegin();

    m_lowerLimits.clear();
    m_upperLimits.clear();

    m_lowerLimits.push_back(this->GetOutput()->GetRequestedRegion().GetIndex()[1 + isImage3D]);
    unsigned int fromPrevious = 0;

    for (unsigned int j = 0;j < (unsigned int)this->GetNumberOfThreads()-1;++j)
    {
        unsigned int nbPtsThread = fromPrevious;
        unsigned int borneSup = m_lowerLimits[j];
        unsigned int lastSliceNbPts = 0;
        if (fromPrevious != 0)
            borneSup++;

        while ((nbPtsThread < approxNumPtsPerThread)&&(!maskItr.IsAtEnd()))
        {
            lastSliceNbPts = 0;
            // Add one slice
            unsigned int tmpVal = maskItr.GetIndex()[1 + isImage3D];
            while ((tmpVal == borneSup)&&(!maskItr.IsAtEnd()))
            {
                if (maskItr.Get() != 0)
                    lastSliceNbPts++;

                ++maskItr;
                tmpVal = maskItr.GetIndex()[1 + isImage3D];
            }
            nbPtsThread += lastSliceNbPts;
            borneSup++;
        }

        if (j != (unsigned int)this->GetNumberOfThreads() - 1)
        {
            unsigned int nbPtsSliceBefore = nbPtsThread - lastSliceNbPts;

            if ((nbPtsSliceBefore>0) &&(approxNumPtsPerThread - nbPtsSliceBefore < nbPtsThread - approxNumPtsPerThread))
            {
                borneSup--;
                fromPrevious = lastSliceNbPts;
                nbPtsThread = nbPtsThread - lastSliceNbPts;
            }
            else
                fromPrevious = 0;
        }

        if (borneSup<(this->GetOutput()->GetRequestedRegion().GetSize()[1 + isImage3D] + this->GetOutput()->GetRequestedRegion().GetIndex()[1 + isImage3D]))
        {
            m_upperLimits.push_back(borneSup - 1);
            m_lowerLimits.push_back(borneSup);
        }
        else
            break;

    }
    m_upperLimits.push_back(this->GetOutput()->GetRequestedRegion().GetSize()[1 + isImage3D] + this->GetOutput()->GetRequestedRegion().GetIndex()[1 + isImage3D] - 1);
}


template<  typename TInputImage, typename TOutputImage >
int
ComputeNormalizedCrossCorrelationImageFilter< TInputImage,TOutputImage >
::SplitRequestedRegion(int i, int num, OutputImageRegionType& splitRegion)
{
    splitRegion = this->GetOutput()->GetRequestedRegion();
    bool isImage3D = (this->GetOutput()->GetRequestedRegion().GetSize()[2] > 1);

    splitRegion.SetIndex(1 + isImage3D,m_lowerLimits[i]);
    splitRegion.SetSize(1 + isImage3D,m_upperLimits[i] - m_lowerLimits[i] + 1);

    return m_upperLimits.size();
}


template<  typename TInputImage, typename TOutputImage >
void
ComputeNormalizedCrossCorrelationImageFilter< TInputImage,TOutputImage >
::CreateComputationMaskFromInputs()
{
    if (m_ComputationMask)
        return;

    if (this -> GetVerbose())
        std::cout << "No computation mask specified... Using the whole image..." << std::endl;

    if (!m_ComputationMask)
    {
        m_ComputationMask = MaskImageType::New();
        m_ComputationMask->Initialize();
        InputImageRegionType region = this->GetInput(0)->GetLargestPossibleRegion();
        m_ComputationMask->SetRegions(region);
        m_ComputationMask->SetSpacing(this->GetInput(0)->GetSpacing());
        m_ComputationMask->SetOrigin(this->GetInput(0)->GetOrigin());
        m_ComputationMask->SetDirection(this->GetInput(0)->GetDirection());
        m_ComputationMask->Allocate();
        m_ComputationMask->FillBuffer(1);
    }
}


template <  typename TInputImage, typename TOutputImage >
void
ComputeNormalizedCrossCorrelationImageFilter<TInputImage, TOutputImage>
::GenerateInputRequestedRegion()
{
    // this filter requires the all of the input image to be in the buffer
    int numberOfInputs = this->GetNumberOfInputs();
    for (int i=0; i<numberOfInputs; i++)
    {
        InputImagePointer inputPtr = const_cast< TInputImage * >(this->GetInput(i));
        if ( inputPtr )
            inputPtr->SetRequestedRegionToLargestPossibleRegion();
    }
}


template<  typename TInputImage, typename TOutputImage >
void
ComputeNormalizedCrossCorrelationImageFilter< TInputImage,TOutputImage >
::InitializeComputationRegionFromMask()
{
    typedef ImageRegionIteratorWithIndex< MaskImageType > MaskRegionIteratorType;
    MaskRegionIteratorType maskItr(m_ComputationMask,m_ComputationMask->GetLargestPossibleRegion());
    maskItr.GoToBegin();
    MaskImageType::IndexType minPos, maxPos;
    for (unsigned int i = 0;i < m_ComputationMask->GetImageDimension();++i)
    {
        minPos[i] = m_ComputationMask->GetLargestPossibleRegion().GetIndex()[i] + m_ComputationMask->GetLargestPossibleRegion().GetSize()[i];
        maxPos[i] = 0;
    }

    while (!maskItr.IsAtEnd())
    {
        if (maskItr.Get() != 0)
        {
            MaskImageType::IndexType tmpInd = maskItr.GetIndex();

            for (unsigned int i = 0;i < m_ComputationMask->GetImageDimension();++i)
            {
                if (minPos[i] > tmpInd[i])
                    minPos[i] = tmpInd[i];

                if (maxPos[i] < tmpInd[i])
                    maxPos[i] = tmpInd[i];
            }
        }
        ++maskItr;
    }

    m_ComputationRegion.SetIndex(minPos);
    MaskImageType::SizeType tmpSize;
    for (unsigned int i = 0;i < m_ComputationMask->GetImageDimension();++i)
        tmpSize[i] = maxPos[i] - minPos[i] + 1;
    m_ComputationRegion.SetSize(tmpSize);
    this->GetOutput()->SetRequestedRegion(m_ComputationRegion);
}


template<  typename TInputImage, typename TOutputImage >
double
ComputeNormalizedCrossCorrelationImageFilter< TInputImage,TOutputImage >
::GetNormalizedCrossCorrelationValue()
{
    if(m_ComputeMetricAtSingleVoxel == true)
        return m_metricValueAtCentralVoxel;

    unsigned long numVoxels = 0;
    m_nccValue = 0.0;

    itk::ImageRegionConstIterator<TOutputImage>  outputItr(this->GetOutput(), this->GetOutput()->GetLargestPossibleRegion());
    itk::ImageRegionConstIterator<MaskImageType> maskItr(m_ComputationMask, m_ComputationMask->GetLargestPossibleRegion());

    for(outputItr.GoToBegin(), maskItr.GoToBegin(); !outputItr.IsAtEnd(); ++outputItr,  ++maskItr)
    {
        if(maskItr.Get() != 0)
        {
            m_nccValue += outputItr.Get();
            ++numVoxels;
        }
    }

    m_nccValue /= numVoxels;
    return m_nccValue;
}


template<  typename TInputImage, typename TOutputImage >
void
ComputeNormalizedCrossCorrelationImageFilter< TInputImage,TOutputImage >
::SetCentralVoxelIndex(InputImageIndexType indx)
{
    m_ComputeMetricAtSingleVoxel = true;
    m_VoxelIndex = indx;
}


} // end namespace itk
#endif
