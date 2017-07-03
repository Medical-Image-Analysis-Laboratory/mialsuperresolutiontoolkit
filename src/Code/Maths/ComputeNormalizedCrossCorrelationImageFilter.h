/*=========================================================================

Copyright (c) 2017 Medical Image Analysis Laboratory (MIAL), Lausanne
  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
==========================================================================*/
#ifndef _ITK_ComputeNormalizedCrossCorrelation_H_
#define _ITK_ComputeNormalizedCrossCorrelation_H_

#include <iostream>
#include <vector>

#include "itkImageToImageFilter.h"
#include "itkImage.h"

namespace itk
{
template <typename TInputImage, typename TOutputImage>
        class ITK_EXPORT ComputeNormalizedCrossCorrelationImageFilter :
                public ImageToImageFilter< TInputImage, TOutputImage >
{
    public:
    /** Standard class typedefs. */
    typedef ComputeNormalizedCrossCorrelationImageFilter Self;
    typedef ImageToImageFilter< TInputImage, TOutputImage > Superclass;
    typedef SmartPointer<Self> Pointer;
    typedef SmartPointer<const Self>  ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods) */
    itkTypeMacro(ComputeNormalizedCrossCorrelationImageFilter, ImageToImageFilter);

    /** typedef support */
    typedef TInputImage  InputImageType;
    typedef typename InputImageType::Pointer InputImagePointer;
    typedef typename Superclass::InputImageRegionType   InputImageRegionType;

    typedef typename TInputImage::IndexType             InputImageIndexType;
    typedef typename Superclass::OutputImageRegionType  OutputImageRegionType;

    /** ComputeNormalizedCrossCorrelation typedefs */
    typedef itk::Image <float, 3> MaskImageType;
    typedef MaskImageType::RegionType MaskRegionType;
    typedef MaskImageType::Pointer MaskImagePointer;

    //Thread functions which will be called
    void ThreadOnNumbers();
    void ThreadOnNumbers(unsigned int minExp, unsigned int maxExp);
    void ThreadOnRegions();
    void ThreadOnRegions(OutputImageRegionType &region);

    //Set/Get functions
    itkSetMacro(ComputationMask, MaskImagePointer);
    itkGetMacro(ComputationMask, MaskImagePointer);
    itkSetMacro(Verbose, bool);

    unsigned int GetActualNumberOfThreads()
    {
        return m_lowerLimits.size();
    }

    itkSetMacro(BlockHalfSizeX, int);
    itkGetMacro(BlockHalfSizeX, int);
    itkSetMacro(BlockHalfSizeY, int);
    itkGetMacro(BlockHalfSizeY, int);
    itkSetMacro(BlockHalfSizeZ, int);
    itkGetMacro(BlockHalfSizeZ, int);

    itkSetMacro(SigmaThreshold, double);
    itkGetMacro(SigmaThreshold, double);

    double GetNormalizedCrossCorrelationValue();

    void SetCentralVoxelIndex(InputImageIndexType indx);

    protected:
    ComputeNormalizedCrossCorrelationImageFilter()
    : Superclass()
    {
        m_ComputationMask=NULL;
        m_Verbose = true;
        m_BlockHalfSizeX=2;
        m_BlockHalfSizeY=2;
        m_BlockHalfSizeZ=2;
        m_lowerLimits.clear();
        m_upperLimits.clear();
        m_nccValue = 0.0;
        m_SigmaThreshold = 1e-3;
        m_ComputeMetricAtSingleVoxel = false;
        m_metricValueAtCentralVoxel = 0.0;
    }

    virtual ~ComputeNormalizedCrossCorrelationImageFilter() {}

    //ComputeNormalizedCrossCorrelation protected functions
    void CreateComputationMaskFromInputs();
    void InitializeComputationRegionFromMask();
    void InitializeSplitRegionsFromMask();
    void Initialize();
    int SplitRequestedRegion(int i, int num, OutputImageRegionType& splitRegion);

    struct ThreadStruct
    {
        Pointer Filter;
    };

    // Does the splitting and calls some Threads on a sub region
    static ITK_THREAD_RETURN_TYPE CallThreadOnRegions( void *arg );
    // Does the splitting and calls some Threads on a numbers
    static ITK_THREAD_RETURN_TYPE CallThreadOnNumbers( void *arg );

    //Redefine virtual functions
    void GenerateData();
    void GenerateInputRequestedRegion();

    private:
    ComputeNormalizedCrossCorrelationImageFilter(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    bool   m_Verbose;
    int    m_BlockHalfSizeX;
    int    m_BlockHalfSizeY;
    int    m_BlockHalfSizeZ;
    double m_nccValue;

    //For overriding the split image region with something more intelligent taking into account the mask
    std::vector < unsigned int > m_lowerLimits, m_upperLimits;

    double m_SigmaThreshold;

    MaskImagePointer    m_ComputationMask;
    MaskRegionType      m_ComputationRegion;
    InputImageIndexType m_VoxelIndex;
    bool                m_ComputeMetricAtSingleVoxel;
    double              m_metricValueAtCentralVoxel;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "ComputeNormalizedCrossCorrelationImageFilter.txx"
#endif

#endif
