#ifndef __BTKNLMTOOL_TXX__
#define __BTKNLMTOOL_TXX__

#include "btkNLMTool.h"

namespace btk
{
template <typename T>
void NLMTool<T>::SetInput(itkTPointer inputImage)
{
  m_inputImage = inputImage;
  //compute characteristics of the input image
  m_region  = m_inputImage->GetLargestPossibleRegion();
  m_size    = m_region.GetSize();
  m_spacing = m_inputImage->GetSpacing();

  //duplicate the input image into the output image to keep all header information
  typename itkTDuplicator::Pointer duplicator = itkTDuplicator::New();
  duplicator->SetInputImage( inputImage );
  duplicator->Update();
  m_outputImage = duplicator->GetOutput();

  m_useTheReferenceImage = false;
  m_useGlobalSmoothing = true;

  //duplicate the input image into the rangeBandwidth image to keep all header information
  typename itkTDuplicator::Pointer duplicator2 = itkTDuplicator::New();
  duplicator2->SetInputImage( inputImage );
  duplicator2->Update();
  m_rangeBandwidthImage = duplicator2->GetOutput();
  m_rangeBandwidthImage->FillBuffer(0.0);
}

template <typename T>
void NLMTool<T>::SetReferenceImage(itkTPointer refImage)
{
  m_refImage = refImage;
  m_useTheReferenceImage = true;
}

template <typename T>
void NLMTool<T>::SetDefaultParameters()
{
  //SetPaddingValue(0);
  SetPatchSize(1);
  SetSpatialBandwidth(5);
  SetSmoothing(1);
  SetCentralPointStrategy(-1);
  SetBlockwiseStrategy(2);
  SetOptimizationStrategy(1);
  SetLowerThresholds(0.95, 0.5);
}

template <typename T>
void NLMTool<T>::SetPatchSize(int h)
{
  std::cout<<"Computing patch size (taking into account possible image anisotropy)"<<std::endl;
  float minVoxSz = m_spacing[0];

  if(m_spacing[1] < minVoxSz)
  {
      minVoxSz = m_spacing[1];
  }

  if(m_spacing[2] < minVoxSz)
  {
      minVoxSz = m_spacing[2];
  }

  m_halfPatchSize[0] = (int)(0.5 + h * minVoxSz / m_spacing[0]);
  m_halfPatchSize[1] = (int)(0.5 + h * minVoxSz / m_spacing[1]);
  m_halfPatchSize[2] = (int)(0.5 + h * minVoxSz / m_spacing[2]);
  std::cout<<"half patchSize : "<<m_halfPatchSize[0]<<" "<<m_halfPatchSize[1]<<" "<<m_halfPatchSize[2]<<std::endl;

  m_fullPatchSize[0] = 2 * m_halfPatchSize[0] + 1;
  m_fullPatchSize[1] = 2 * m_halfPatchSize[1] + 1;
  m_fullPatchSize[2] = 2 * m_halfPatchSize[2] + 1;
}

template <typename T>
void NLMTool<T>::SetSpatialBandwidth(int s)
{
  std::cout<<"Computing spatial bandwidth (taking into account possible image anisotropy)"<<std::endl;
  float minVoxSz = m_spacing[0];
  if(m_spacing[1] < minVoxSz)
  {
      minVoxSz = m_spacing[1];
  }
  if(m_spacing[2] < minVoxSz)
  {
      minVoxSz = m_spacing[2];
  }

  m_halfSpatialBandwidth[0] = (int)(0.5 + s * minVoxSz / m_spacing[0]);
  m_halfSpatialBandwidth[1] = (int)(0.5 + s * minVoxSz / m_spacing[1]);
  m_halfSpatialBandwidth[2] = (int)(0.5 + s * minVoxSz / m_spacing[2]);
  std::cout<<"half spatialBandwidth : "<<m_halfSpatialBandwidth[0]<<" "<<m_halfSpatialBandwidth[1]<<" "<<m_halfSpatialBandwidth[2]<<std::endl;

  m_fullSpatialBandwidth[0] = 2 * m_halfSpatialBandwidth[0] + 1;
  m_fullSpatialBandwidth[1] = 2 * m_halfSpatialBandwidth[1] + 1;
  m_fullSpatialBandwidth[2] = 2 * m_halfSpatialBandwidth[2] + 1;
}

template <typename T>
void NLMTool<T>::SetPaddingValue(float padding)
{
  m_padding = padding;

  std::cout<<"Creating the mask image using the padding value ("<<padding<<")"<<std::endl;
  typename itkTDuplicator::Pointer duplicator = itkTDuplicator::New();
  duplicator->SetInputImage( m_inputImage );
  duplicator->Update();
  m_maskImage = duplicator->GetOutput();

  double count = 0;
  itkTIterator iterator( m_maskImage, m_maskImage->GetRequestedRegion() );
  for(iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    if(iterator.Get() <= padding)
    {
      iterator.Set(0);
    }
    else
    {
      iterator.Set(1);
      count++;
    }
  }
  std::cout<<"Percentage of points to be processed : "<<(int)(count / m_maskImage->GetLargestPossibleRegion().GetNumberOfPixels() * 100.0) <<std::endl;

}

template <typename T>
void NLMTool<T>::SetMaskImage(itkTPointer maskImage)
{
  m_maskImage = maskImage;

  //check the size and spacing of the mask wrt the input image
  typename itkTImage::SpacingType spacing;
  typename itkTImage::SizeType    size;
  typename itkTImage::RegionType  region;

  region  = m_maskImage->GetLargestPossibleRegion();
  size    = m_region.GetSize();
  spacing = m_maskImage->GetSpacing();

  typename itkTImage::IndexType q;
  for(unsigned int i=0; i!= q.GetIndexDimension(); i++)
  {
    if( (size[i] != m_size[i]) || (spacing[i] != m_spacing[i]) )
    {
      std::cout<<"*************************************************************************************"<<std::endl;
      std::cout<<"WARNING : the size or the spacing of the mask image are incorrect wrt the input image"<<std::endl;
      std::cout<<"*************************************************************************************"<<std::endl;
      break;
    }
  }

  double count = 0;
  itkTIterator iterator( m_maskImage, m_maskImage->GetRequestedRegion() );
  for(iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    if(iterator.Get() > 0)
    {
      count++;
    }
  }
  std::cout<<"Percentage of points to be processed : "<<(int)(count / m_maskImage->GetLargestPossibleRegion().GetNumberOfPixels() * 100.0) <<std::endl;


}

template <typename T>
void NLMTool<T>::SetCentralPointStrategy(int s)
{
  m_centralPointStrategy = s;
}

template <typename T>
void NLMTool<T>::SetBlockwiseStrategy(int b)
{
  m_blockwise = b;
}

template <typename T>
void NLMTool<T>::SetOptimizationStrategy(int o)
{
    m_optimized = o;

    if(o==1)
    {
        std::cout<<"Optimized mode. Computing Mean and Variance images"<<std::endl;
        m_meanImage = itkTImage::New();
        m_meanImage->SetRegions(m_inputImage->GetLargestPossibleRegion());
        m_meanImage->SetSpacing( m_inputImage->GetSpacing() );
        m_meanImage->SetOrigin( m_inputImage->GetOrigin() );
        m_meanImage->SetDirection( m_inputImage->GetDirection() );
        m_meanImage->Allocate();
        m_meanImage->FillBuffer(0);

        m_varianceImage = itkTImage::New();
        m_varianceImage->SetRegions(m_inputImage->GetLargestPossibleRegion());
        m_varianceImage->SetSpacing( m_inputImage->GetSpacing() );
        m_varianceImage->SetOrigin( m_inputImage->GetOrigin() );
        m_varianceImage->SetDirection( m_inputImage->GetDirection() );
        m_varianceImage->Allocate();
        m_varianceImage->FillBuffer(0);

        int x,y,z;
        #pragma omp parallel for private(x,y,z) schedule(dynamic)

        for(z=0; z < (int)m_size[2]; z++)
        {
            for(y=0; y < (int)m_size[1]; y++)
            {
                for(x=0; x < (int)m_size[0]; x++)
                {
                    typename itkTImage::IndexType p;
                    p[0] = x;
                    p[1] = y;
                    p[2] = z;

                    if( m_maskImage->GetPixel(p) > 0 )
                    {
                        itkTPointer patch = itkTImage::New();
                        CreatePatch(patch);
                        GetPatch(p,patch);
                        itkTIterator itp( patch, patch->GetRequestedRegion() );
                        double m = 0;
                        double m2= 0;
                        for(itp.GoToBegin(); !itp.IsAtEnd(); ++itp)
                        {
                            m += itp.Get();
                            m2+= (itp.Get() * itp.Get());
                        }
                        int n = patch->GetLargestPossibleRegion().GetNumberOfPixels();
                        float mean = m / n;
                        float variance = (m2 / n) - (mean * mean) ;

                        m_meanImage->SetPixel( p, mean );
                        m_varianceImage->SetPixel( p, variance );
                    }
                }
            }
        }

    }
}

template <typename T>
void NLMTool<T>::SetLowerThresholds(float m, float v)
{
  m_lowerMeanThreshold = m;
  m_lowerVarianceThreshold = v;
}


template <typename T>
double NLMTool<T>::ComputePseudoResidual(typename itkTImage::IndexType & pixelIndex)
{
  double ei = 0;

  itkTPointer tmpImage;
  if(m_useTheReferenceImage == false)
  {
    tmpImage = m_inputImage;
  }
  else
  {
    tmpImage = m_refImage;
  }

  int x = pixelIndex[0];
  int y = pixelIndex[1];
  int z = pixelIndex[2];

  double value = tmpImage->GetPixel(pixelIndex);
    //TODO : Maybe use a convolution instead !
    pixelIndex[0] = x+1;	  pixelIndex[1] = y;	  pixelIndex[2] = z;
    ei = tmpImage->GetPixel(pixelIndex);
    pixelIndex[0] = x-1;	  pixelIndex[1] = y;	  pixelIndex[2] = z;
    ei += tmpImage->GetPixel(pixelIndex);
    pixelIndex[0] = x;	    pixelIndex[1] = y+1;	pixelIndex[2] = z;
    ei += tmpImage->GetPixel(pixelIndex);
    pixelIndex[0] = x;  	  pixelIndex[1] = y-1;	pixelIndex[2] = z;
    ei += tmpImage->GetPixel(pixelIndex);
    pixelIndex[0] = x;  	  pixelIndex[1] = y;	  pixelIndex[2] = z+1;
    ei += tmpImage->GetPixel(pixelIndex);
    pixelIndex[0] = x;  	  pixelIndex[1] = y;	  pixelIndex[2] = z-1;
    ei += tmpImage->GetPixel(pixelIndex);

    ei = sqrt(6.0/7.0)*(value -ei/6.0);
  return ei;
}

template <typename T>
double NLMTool<T>::ComputePseudoResidualSafely(typename itkTImage::IndexType & pixelIndex)
{
  double tmp = 0;

  if( (pixelIndex[0] > 1) && (pixelIndex[1] > 1)
          && (pixelIndex[2] > 1) && (pixelIndex[0] < (unsigned int)m_size[0]-1 )
          && (pixelIndex[1] < (unsigned int)m_size[1]-1) && (pixelIndex[2] < (unsigned int)m_size[2]-1) )
  {
    tmp = ComputePseudoResidual(pixelIndex);
  }
  return tmp;
}

template <typename T>
float NLMTool<T>::MADEstimation(std::vector<float> & vecei, float & beta)
{
  //Estimation of sigma with MAD
  std::sort(vecei.begin(), vecei.end());
  float med = vecei[(int)(vecei.size()/2)];
  std::cout<<"Median in MAD estimation : "<<med<<std::endl;
  for(unsigned int i=0; i<vecei.size(); i++)
  {
    vecei[i] = fabs(vecei[i] - med);
  }
  std::sort(vecei.begin(), vecei.end());

  double sigma2 = 1.4826 * vecei[(int)(vecei.size()/2)];
  std::cout<<"sigma : "<<sigma2<<std::endl;
  sigma2 = sigma2 * sigma2;
  float NLMsmooth = 2 * beta * sigma2 * (2*m_halfPatchSize[0]+1) * (2*m_halfPatchSize[1]+1) * (2*m_halfPatchSize[2]+1);
  return NLMsmooth;
}

template <typename T>
void NLMTool<T>::SetSmoothing(float beta)
{
    //this function should be rewritten using a convolution-based approach
    std::cout<<"Computing the global range bandwidth (corresponding to the smoothing parameter for the NLM algorithm)."<<std::endl;
    int x,y,z;

    std::vector<float> vecei;
    //since we have use to use a neighborhood around the current voxel, we neglect the border to avoid slow tests.

    #pragma omp parallel for private(x,y,z) schedule(dynamic)
    for(z=1;z<(int)m_size[2]-1;z++)
    {
        for(y=1;y<(int)m_size[1]-1;y++)
        {
            for(x=1;x<(int)m_size[0]-1;x++)
            {
                typename itkTImage::IndexType pixelIndex;
                pixelIndex[0] = x;
                pixelIndex[1] = y;
                pixelIndex[2] = z;

                if( m_maskImage->GetPixel(pixelIndex) > 0)
                {
                    double ei = ComputePseudoResidual(pixelIndex);

                    #pragma omp critical
                    if(fabs(ei>0))
                    {
                        vecei.push_back(fabs(ei));
                    }
                }
            }
        }
    }
    float NLMsmooth = MADEstimation(vecei, beta);
    std::cout<<"Global smoothing parameter h : "<<sqrt(NLMsmooth)<<std::endl;
    m_rangeBandwidthImage->FillBuffer(NLMsmooth);

}
template <typename T>
void NLMTool<T>::SetLocalSmoothing(float beta)
{
    std::cout<<"Computing the range bandwidth locally, for every voxel."<<std::endl;
    int x,y,z;


#pragma omp parallel for private(x,y,z) schedule(dynamic)
    for(z=0;z<(int)m_size[2];z++)
    {
        for(y=0;y<(int)m_size[1];y++)
        {
            for(x=0;x<(int)m_size[0];x++)
            {

                typename itkTImage::IndexType pixelIndex;
                double ei = 0;
                pixelIndex[0] = x;
                pixelIndex[1] = y;
                pixelIndex[2] = z;

                if( m_maskImage->GetPixel(pixelIndex) > 0)
                {

                    typename itkTImage::RegionType searchRegion;
                    ComputeSearchRegion(pixelIndex,searchRegion);

                    itkTIteratorWithIndex itRegion(m_inputImage, searchRegion);
                    typename itkTImage::IndexType neighbourPixelIndex;

                    std::vector<float> vecei;

                    for(itRegion.GoToBegin(); !itRegion.IsAtEnd(); ++itRegion)
                    {
                        neighbourPixelIndex = itRegion.GetIndex();

                        bool goForIt = true;
                        if(m_optimized == 1)
                        {
                            goForIt = CheckSpeed(pixelIndex, neighbourPixelIndex);
                        }

                        if(goForIt == true)
                        {
                            if( m_maskImage->GetPixel(neighbourPixelIndex) > 0)
                            {
                                ei = ComputePseudoResidualSafely(neighbourPixelIndex);
//                                if(fabs(ei>0))
//                                {
                                    vecei.push_back(fabs(ei));
//                                }
                            }


                        }

                    }
                    if(vecei.size()>0)
                    {
                        float NLMsmooth = MADEstimation(vecei, beta);
                        m_rangeBandwidthImage->SetPixel(pixelIndex, NLMsmooth);

                    }
                }

            }
        }
    }

}

template <typename T>
typename NLMTool<T>::itkTPointer
NLMTool<T>::GetOutput()
{
  return m_outputImage;
}

template <typename T>
void NLMTool<T>::ComputeOutput()
{
  std::cout<<"Compute the denoised image using NLM algorithm"<<std::endl;
  if(m_useTheReferenceImage == true)
  {
    std::cout<<"Use of a reference image"<<std::endl;
  }

  itkTPointer denoisedImage = itkTImage::New();
  typename itkTDuplicator::Pointer duplicator = itkTDuplicator::New();
  duplicator->SetInputImage( m_inputImage );
  duplicator->Update();
  denoisedImage = duplicator->GetOutput();
  denoisedImage->FillBuffer(0);
  /*
  denoisedImage->SetRegions(m_inputImage->GetLargestPossibleRegion());
  denoisedImage->SetSpacing( m_inputImage->GetSpacing() );
  denoisedImage->SetOrigin( m_inputImage->GetOrigin() );
  denoisedImage->SetDirection( m_inputImage->GetDirection() );
  denoisedImage->Allocate();
  denoisedImage->FillBuffer(0);
  */

  itkTIterator denoisedIt( denoisedImage, denoisedImage->GetLargestPossibleRegion() );
  itkTIterator outputIt( m_outputImage, m_outputImage->GetLargestPossibleRegion());

  int x,y,z;

  if(m_blockwise == 0)
  {
    std::cout<<"pointwise denoising"<<std::endl;
    typename itkTImage::IndexType q;
    for(unsigned int i=0; i!= q.GetIndexDimension(); i++)
    {
      q[i] = m_halfPatchSize[i];
    }
      #pragma omp parallel for private(x,y,z) schedule(dynamic)
      for(z=0; z < (int)m_size[2]; z++)
      {
        for(y=0; y < (int)m_size[1]; y++)
        {
          for(x=0; x < (int)m_size[0]; x++)
          {
            typename itkTImage::IndexType p;
            p[0] = x;
            p[1] = y;
            p[2] = z;

            if( m_maskImage->GetPixel(p) > 0 )
            {
              itkTPointer patch = itkTImage::New();
              double sum = 0;
              if(m_useTheReferenceImage==false)
              {
                  sum = GetDenoisedPatch(p, patch);
              }
              else
              {
                  sum = GetDenoisedPatchUsingTheReferenceImage(p, patch);
              }

              denoisedImage->SetPixel( p, patch->GetPixel(q) );
            }
          }
        }
      }
  }
  if(m_blockwise >= 1)
  {
    itkTPointer weightImage = itkTImage::New();
    weightImage->SetRegions(m_inputImage->GetLargestPossibleRegion());
    weightImage->SetSpacing( m_inputImage->GetSpacing() );
    weightImage->SetOrigin( m_inputImage->GetOrigin() );
    weightImage->SetDirection( m_inputImage->GetDirection() );
    weightImage->Allocate();
    weightImage->FillBuffer(0);

    if(m_blockwise == 1)
    {
      std::cout<<"blockwise denoising"<<std::endl;
      #pragma omp parallel for private(x,y,z) schedule(dynamic)
      for(z=0; z < (int)m_size[2]; z++)
      {
        for(y=0; y < (int)m_size[1]; y++)
        {
          for(x=0; x < (int)m_size[0]; x++)
          {
            typename itkTImage::IndexType p;
            p[0] = x;
            p[1] = y;
            p[2] = z;

            if( m_maskImage->GetPixel(p) > 0 )
            {
              itkTPointer patch = itkTImage::New();
              double sum = 0;
              if(m_useTheReferenceImage==false)
              {
                  sum = GetDenoisedPatch(p, patch);
              }
              else
              {
                  sum = GetDenoisedPatchUsingTheReferenceImage(p, patch);
              }

              double weight = 1.0;
              #pragma omp critical
              AddPatchToImage(p, patch, denoisedImage, weightImage, weight);
            }
          }
        }
      }
    }
    if(m_blockwise == 2)
    {
        std::cout<<"fast blockwise denoising"<<std::endl;
        //TODO: Simplify this, there is to much for-if-for !
        #pragma omp parallel for private(x,y,z) schedule(dynamic)
        for(z=0; z < (int)m_size[2]; z++)
        {
            if( z%(m_halfPatchSize[2]+1) == 0)
            {
                for(y=0; y < (int)m_size[1]; y++)
                {
                    if( y%(m_halfPatchSize[1]+1) == 0)
                    {
                        for(x=0; x < (int)m_size[0]; x++)
                        {
                            if( x%(m_halfPatchSize[0]+1) == 0 )
                            {
                                typename itkTImage::IndexType p;
                                p[0] = x;
                                p[1] = y;
                                p[2] = z;

                                if( m_maskImage->GetPixel(p) > 0 )
                                {
                                    itkTPointer patch = itkTImage::New();
                                    double sum = 0;
                                    if(m_useTheReferenceImage==false)
                                    {
                                        sum = GetDenoisedPatch(p, patch);
                                    }
                                    else
                                    {
                                        sum = GetDenoisedPatchUsingTheReferenceImage(p, patch);
                                    }

                                    double weight = 1.0;
                                    #pragma omp critical
                                    AddPatchToImage(p, patch, denoisedImage, weightImage, weight);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    itkTIterator weightIt( weightImage, weightImage->GetLargestPossibleRegion() );
    //weight normalization
    for ( denoisedIt.GoToBegin(), weightIt.GoToBegin(); !denoisedIt.IsAtEnd(); ++denoisedIt, ++weightIt)
    {
      if( weightIt.Get() > 0 )
      {
        denoisedIt.Set( denoisedIt.Get() / weightIt.Get() );
      }
    }

  }

  //TODO:This should be modified by just copy the two images.
  //Convert data from denoisedImage to m_outputImage
  for ( denoisedIt.GoToBegin(), outputIt.GoToBegin(); !denoisedIt.IsAtEnd(); ++denoisedIt, ++outputIt)
  {
    //outputIt.Set( (T)rint(denoisedIt.Get()) );
    outputIt.Set( (T)(denoisedIt.Get()) );
  }

}



template <typename T>
void NLMTool<T>::PrintInfo()
{
  //Image information (size, spacing etc.)
  std::cout<<"size of the input image: "<<m_size[0]<<" "<<m_size[1]<<" "<<m_size[2]<<std::endl;
  std::cout<<"spacing of the input image: "<<m_spacing[0]<<" "<<m_spacing[1]<<" "<<m_spacing[2]<<std::endl;

  //Image information (size, spacing etc.)
  typename itkTImage::SpacingType spacing;
  typename itkTImage::SizeType    size;
  typename itkTImage::RegionType  region;

  region  = m_outputImage->GetLargestPossibleRegion();
  size    = region.GetSize();
  std::cout<<"size of the output image: "<<size[0]<<" "<<size[1]<<" "<<size[2]<<std::endl;
  spacing = m_outputImage->GetSpacing();
  std::cout<<"spacing of the output image: "<<spacing[0]<<" "<<spacing[1]<<" "<<spacing[2]<<std::endl;

}

template <typename T>
void NLMTool<T>::CreatePatch(itkTPointer & patch)
{
  //resize and allocate the patch and set the estimate to 0
  typename itkTImage::SizeType size = m_fullPatchSize;
  typename itkTImage::RegionType region;
  region.SetSize(size);
  patch->SetRegions(region);
  patch->Allocate();
  patch->FillBuffer(0);
}

template <typename T>
void NLMTool<T>::GetPatch(typename itkTImage::IndexType p, itkTPointer & patch)
{
  //this function is equivalent to a cropping function
  //WARNING : no check about the size of the input patch !
  patch->FillBuffer(0);
  typename itkTImage::RegionType imageRegion;
  typename itkTImage::RegionType patchRegion;
  ComputePatchRegion(p,imageRegion,patchRegion);

  itkTConstIterator inputIt( m_inputImage, imageRegion);
  itkTIterator outputIt( patch, patchRegion);

  for ( inputIt.GoToBegin(), outputIt.GoToBegin(); !inputIt.IsAtEnd(); ++inputIt, ++outputIt)
  {
    outputIt.Set( inputIt.Get() );
  }


  //Old school style :)
  /*
  typename itkTImage::IndexType pixelIndex;
  typename itkTImage::IndexType patchIndex;

  for(int pz=-(int)m_halfPatchSize[2];pz<=(int)m_halfPatchSize[2];pz++){
    pixelIndex[2] = p[2] + pz;
    patchIndex[2] = pz + m_halfPatchSize[2];
    if( (pixelIndex[2]>=0) && (pixelIndex[2]<(int)m_size[2]) ){
      for(int py=-(int)m_halfPatchSize[1];py<=(int)m_halfPatchSize[1];py++){
        pixelIndex[1] = p[1] + py;
        patchIndex[1] = py + m_halfPatchSize[1];
        if( (pixelIndex[1]>=0) && (pixelIndex[1]<(int)m_size[1]) ){
          for(int px=-(int)m_halfPatchSize[0];px<=(int)m_halfPatchSize[0];px++){
            pixelIndex[0] = p[0] + px;
            patchIndex[0] = px + m_halfPatchSize[0];
            if( (pixelIndex[0]>=0) && (pixelIndex[0]<(int)m_size[0]) ){
              patch->SetPixel(patchIndex, m_inputImage->GetPixel(pixelIndex));
            }
          }
        }
      }
    }
  }
  */

}

template <typename T>
void NLMTool<T>::GetPatchFromReferenceImage(typename itkTImage::IndexType p, itkTPointer & patch)
{
  //this function is equivalent to a cropping function
  //WARNING : no check about the size of the input patch !
  patch->FillBuffer(0);
  typename itkTImage::RegionType imageRegion;
  typename itkTImage::RegionType patchRegion;
  ComputePatchRegion(p,imageRegion,patchRegion);

  itkTConstIterator refIt( m_refImage, imageRegion);
  itkTIterator outputIt( patch, patchRegion);

  for ( refIt.GoToBegin(), outputIt.GoToBegin(); !refIt.IsAtEnd(); ++refIt, ++outputIt)
  {
    outputIt.Set( refIt.Get() );
  }
}

template <typename T>
void NLMTool<T>::AddPatchToImage(typename itkTImage::IndexType p, itkTPointer & patch, itkTPointer & image, itkTPointer & weightImage, double weight)
{
  //this function add a patch value to the denoised image
  typename itkTImage::RegionType imageRegion;
  typename itkTImage::RegionType patchRegion;
  ComputePatchRegion(p,imageRegion,patchRegion);

  itkTIterator imageIt( image, imageRegion);
  itkTIterator weightIt( weightImage, imageRegion);
  itkTIterator patchIt( patch, patchRegion);

  for ( imageIt.GoToBegin(), patchIt.GoToBegin(), weightIt.GoToBegin(); !imageIt.IsAtEnd(); ++imageIt, ++patchIt, ++weightIt)
  {
    imageIt.Set( imageIt.Get() + patchIt.Get() );
    weightIt.Set( weightIt.Get() + weight );
  }

}

template <typename T>
double NLMTool<T>::PatchDistance(itkTPointer & p,itkTPointer & q)
{
  double diff=0;
  double dist = 0;
  itkTConstIterator itp( p, p->GetRequestedRegion() );
  itkTConstIterator itq( q, q->GetRequestedRegion() );

  for(itp.GoToBegin(), itq.GoToBegin(); !itp.IsAtEnd(); ++itp, ++itq)
  {
    diff = itp.Get() - itq.Get();
    dist += diff*diff;
  }
  return dist;
}

template <typename T>
double NLMTool<T>::GetDenoisedPatch(typename itkTImage::IndexType p, itkTPointer & patch)
{
  double wmax = 0; //maximum weight of patches
  double sum  = 0; //sum of weights (used for normalization purpose)
  double rangeBandwidth = m_rangeBandwidthImage->GetPixel(p);

  //create the patch and set the estimate to 0
  CreatePatch(patch);
  itkTIterator patchIt(patch, patch->GetRequestedRegion());

  //get the value of the patch around the current pixel
  itkTPointer centralPatch = itkTImage::New();
  CreatePatch(centralPatch);
  GetPatch(p,centralPatch);
  itkTIterator centralPatchIt(centralPatch, centralPatch->GetRequestedRegion());

  //set the search region around the current pixel
  typename itkTImage::RegionType searchRegion;
  ComputeSearchRegion(p,searchRegion);

  //create the patch for pixels in the neighbourhood of the current pixel
  itkTPointer neighbourPatch = itkTImage::New();
  CreatePatch(neighbourPatch);
  itkTIterator neighbourPatchIt(neighbourPatch, neighbourPatch->GetRequestedRegion());

  //go through the neighbourhood with a region iterator
  itkTIteratorWithIndex it( m_inputImage, searchRegion);
  typename itkTImage::IndexType neighbourPixelIndex;

  for(it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    neighbourPixelIndex = it.GetIndex();

    bool goForIt = true;
    if(m_optimized == 1)
    {
      goForIt = CheckSpeed(p, neighbourPixelIndex);
    }

    if(goForIt == true)
    {

      GetPatch(neighbourPixelIndex, neighbourPatch);
      double weight = exp( - PatchDistance(centralPatch, neighbourPatch) / rangeBandwidth);

      if(weight>wmax)
      {
          if( (p[0] != neighbourPixelIndex[0]) && (p[1] != neighbourPixelIndex[1]) && (p[2] != neighbourPixelIndex[2]) )//has to be modify
          {
              wmax = weight;
          }
      }

      sum += weight;

      //Add this patch to the current estimate using the computed weight
      for(patchIt.GoToBegin(), neighbourPatchIt.GoToBegin(); !patchIt.IsAtEnd(); ++patchIt, ++neighbourPatchIt)
      {
        patchIt.Set( patchIt.Get() + neighbourPatchIt.Get() * weight );
      }
    }
  }


  //consider now the special case of the central patch
  switch(m_centralPointStrategy)
  {
  case 0:                                        //remove the central patch to the estimated patch
      for(patchIt.GoToBegin(), centralPatchIt.GoToBegin(); !patchIt.IsAtEnd(); ++patchIt, ++centralPatchIt)
      {
          patchIt.Set( patchIt.Get() -1.0 * centralPatchIt.Get() );
      }
      sum -= 1.0;
      break;
  case 1:
      break;                                 //nothing to do
  case -1:
      for(patchIt.GoToBegin(), centralPatchIt.GoToBegin(); !patchIt.IsAtEnd(); ++patchIt, ++centralPatchIt)
      {
          patchIt.Set( patchIt.Get()  + (wmax -1.0) * centralPatchIt.Get() );
      }
      sum += (wmax - 1.0);
      break;
  default:
      //as in case -1
      for(patchIt.GoToBegin(), centralPatchIt.GoToBegin(); !patchIt.IsAtEnd(); ++patchIt, ++centralPatchIt)
      {
          patchIt.Set( patchIt.Get()  + (wmax -1.0) * centralPatchIt.Get() );
      }
      sum += (wmax - 1.0);
      break;
  }

  if(sum>0.0001)
  {
    //Normalization of the denoised patch
    for(patchIt.GoToBegin(); !patchIt.IsAtEnd(); ++patchIt)
    {
      patchIt.Set( patchIt.Get() / sum );
    }
  }
  else
  {
    //copy the central patch to the denoised patch
    for(patchIt.GoToBegin(), centralPatchIt.GoToBegin(); !patchIt.IsAtEnd(); ++patchIt, ++centralPatchIt)
    {
      patchIt.Set( centralPatchIt.Get() );
    }
  }

  return sum;
}

template <typename T>
double NLMTool<T>::GetDenoisedPatchUsingTheReferenceImage(typename itkTImage::IndexType p, itkTPointer & patch)
{
    double wmax = 0; //maximum weight of patches
    double sum  = 0; //sum of weights (used for normalization purpose)
    double rangeBandwidth = m_rangeBandwidthImage->GetPixel(p);

    //create the patch and set the estimate to 0
    CreatePatch(patch);
    itkTIterator patchIt(patch, patch->GetRequestedRegion());

    //get the value of the patch around the current pixel
    itkTPointer centralPatch = itkTImage::New();
    CreatePatch(centralPatch);
    GetPatch(p,centralPatch);
    itkTIterator centralPatchIt(centralPatch, centralPatch->GetRequestedRegion());

    //get the value of the patch around the current pixel in the reference image
    itkTPointer centralPatchReferenceImage = itkTImage::New();
    CreatePatch(centralPatchReferenceImage);
    GetPatchFromReferenceImage(p,centralPatchReferenceImage);

    //set the search region around the current pixel
    typename itkTImage::RegionType searchRegion;
    ComputeSearchRegion(p,searchRegion);

    //create the patch for pixels in the neighbourhood of the current pixel
    itkTPointer neighbourPatch = itkTImage::New();
    CreatePatch(neighbourPatch);
    itkTIterator neighbourPatchIt(neighbourPatch, neighbourPatch->GetRequestedRegion());

    //create the patch for pixels in the neighbourhood of the current pixel in the reference image
    itkTPointer neighbourPatchReferenceImage = itkTImage::New();
    CreatePatch(neighbourPatchReferenceImage);

    //go through the neighbourhood with a region iterator
    itkTIteratorWithIndex it( m_inputImage, searchRegion);
    typename itkTImage::IndexType neighbourPixelIndex;

    for(it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
        neighbourPixelIndex = it.GetIndex();

        bool goForIt = true;
        if(m_optimized == 1)
            goForIt = CheckSpeed(p, neighbourPixelIndex);

        if(goForIt == true)
        {

            GetPatchFromReferenceImage(neighbourPixelIndex, neighbourPatchReferenceImage);
            GetPatch(neighbourPixelIndex, neighbourPatch);

            double weight = exp( - PatchDistance(centralPatchReferenceImage, neighbourPatchReferenceImage) / rangeBandwidth);

            if(weight>wmax)
            {
                if( (p[0] != neighbourPixelIndex[0]) && (p[1] != neighbourPixelIndex[1]) && (p[2] != neighbourPixelIndex[2]) ) //has to be modify
                {
                    wmax = weight;
                }
            }

            sum += weight;

            //Add this patch to the current estimate using the computed weight
            for(patchIt.GoToBegin(), neighbourPatchIt.GoToBegin(); !patchIt.IsAtEnd(); ++patchIt, ++neighbourPatchIt)
            {
                patchIt.Set( patchIt.Get() + neighbourPatchIt.Get() * weight );
            }
        }
    }


    //consider now the special case of the central patch
    switch(m_centralPointStrategy)
    {
    case 0:                                        //remove the central patch to the estimated patch
        for(patchIt.GoToBegin(), centralPatchIt.GoToBegin(); !patchIt.IsAtEnd(); ++patchIt, ++centralPatchIt)
        {
            patchIt.Set( patchIt.Get() -1.0 * centralPatchIt.Get() );
        }
        sum -= 1.0;
        break;

    case 1:
        //nothing to do
        break;

    case -1:
        for(patchIt.GoToBegin(), centralPatchIt.GoToBegin(); !patchIt.IsAtEnd(); ++patchIt, ++centralPatchIt)
        {
            patchIt.Set( patchIt.Get()  + (wmax -1.0) * centralPatchIt.Get() );
        }
        sum += (wmax - 1.0);
        break;

    default:
        //as in case -1
        for(patchIt.GoToBegin(), centralPatchIt.GoToBegin(); !patchIt.IsAtEnd(); ++patchIt, ++centralPatchIt)
        {
            patchIt.Set( patchIt.Get()  + (wmax -1.0) * centralPatchIt.Get() );
        }
        sum += (wmax - 1.0);
        break;
    }

    if(sum>0.0001)
    {
        //Normalization of the denoised patch
        for(patchIt.GoToBegin(); !patchIt.IsAtEnd(); ++patchIt)
        {
            patchIt.Set( patchIt.Get() / sum );
        }
    }
    else
    {
        //copy the central patch to the denoised patch
        for(patchIt.GoToBegin(), centralPatchIt.GoToBegin(); !patchIt.IsAtEnd(); ++patchIt, ++centralPatchIt)
        {
            patchIt.Set( centralPatchIt.Get() );
        }
    }

    return sum;
}


template <typename T>
bool NLMTool<T>::CheckSpeed(typename itkTImage::IndexType p, typename itkTImage::IndexType q)
{
    bool goForIt = true;

    float mSpeed = 0;
    if(m_meanImage->GetPixel(q) ==0)
    {
        if(m_meanImage->GetPixel(p) == 0)
        {
            mSpeed = 1;
        }
        else
        {
            mSpeed = 0;
        }
    }
    else
    {
        mSpeed = m_meanImage->GetPixel(p) / m_meanImage->GetPixel(q);
    }

    if( (mSpeed < m_lowerMeanThreshold) || (mSpeed > 1/m_lowerMeanThreshold) )
    {
        goForIt = false;
    }

    float vSpeed = 0;
    if(m_varianceImage->GetPixel(q) ==0)
    {
        if(m_varianceImage->GetPixel(p) == 0)
        {
            vSpeed = 1;
        }
        else
        {
            vSpeed = 0;
        }
    }
    else
    {
        vSpeed = m_varianceImage->GetPixel(p) / m_varianceImage->GetPixel(q);
    }

    if( (vSpeed < m_lowerVarianceThreshold) || (vSpeed > 1/m_lowerVarianceThreshold) )
    {
        goForIt = false;
    }

    return goForIt;
}

template <typename T>
void NLMTool<T>::ComputeSearchRegion(typename itkTImage::IndexType p, typename itkTImage::RegionType & region)
{
  //create an appropiate search region around the current pixel p
  typename itkTImage::RegionType::IndexType start;
  typename itkTImage::RegionType::SizeType size;

  for(unsigned int i=0; i!= p.GetIndexDimension(); i++)
  {
    start[i] = p[i] - m_halfSpatialBandwidth[i];
    size[i] = m_fullSpatialBandwidth[i];

    if(start[i] < 0)
    {                  //if the starting index is outside the image (<0)
      size[i] += start[i];
      start[i] = 0;
    }
    else if(start[i] >= (int)m_size[i])
    {
      //if the starting index is outside the image (>image size)
      start[i] = m_size[i]-1;
      size[i] = 0;
    }

    int d = (start[i] + size[i]) - m_size[i]; //if the region is not fully inside the image
    if(d>0)
    {
//      size[i] = size[i] - d;
      if(static_cast< int >(size[i]) - d < 0)
      {
          size[i] = 0;
      }
      else
        size[i] = size[i] - d;
    }
  }
  region.SetSize( size );
  region.SetIndex( start );
}

template <typename T>
void NLMTool<T>::ComputePatchRegion(typename itkTImage::IndexType p, typename itkTImage::RegionType & imageRegion, typename itkTImage::RegionType & patchRegion)
{
  //create an appropriate patch region around the current pixel p, and consider an offset for pixels close to boundaries.
  //the patch region is defined in the image coordinate system.
  //the offset allows to do the link between the patch coordinate system and the one of the image.
  typename itkTImage::RegionType::IndexType start;
  typename itkTImage::RegionType::IndexType offset;
  typename itkTImage::RegionType::SizeType size;

  for(unsigned int i=0; i!= p.GetIndexDimension(); i++)
  {
    start[i] = p[i] - m_halfPatchSize[i];
    size[i] = m_fullPatchSize[i];
    offset[i] = 0;

    if(start[i] < 0)
    {                  //if the starting index is outside the image (<0)
      size[i] += start[i];
      offset[i] = -start[i];
      start[i] = 0;
    }
    else if(start[i] >= (int)m_size[i])
    {
      //if the starting index is outside the image (>image size)
      start[i] = m_size[i]-1;
      size[i] = 0;
    }

    int d = (start[i] + size[i]) - m_size[i]; //if the region is not fully inside the image
    if(d>0)
    {
//      size[i] = size[i] - d;
      if(static_cast< int >(size[i]) - d < 0)
      {
          size[i] = 0;
      }
      else
        size[i] = size[i] - d;
    }
  }
  imageRegion.SetSize( size );
  imageRegion.SetIndex( start );
  patchRegion.SetSize( size );
  patchRegion.SetIndex( offset );
}

}
#endif // btkNLMTool_TXX
