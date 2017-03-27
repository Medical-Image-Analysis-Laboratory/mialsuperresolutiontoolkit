/*
Copyright or © or Copr. Université de Strasbourg - Centre National de la Recherche Scientifique

25 january 2011
rousseau@unistra.fr

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
*/

/*
 This program implements a denoising method based on the work of Coupé et al. described in :
 Coupé, P., Yger, P., Prima, S., Hellier, P., Kervrann, C., Barillot, C., 2008. 
 An optimized blockwise nonlocal means denoising filter for 3-D magnetic resonance images.
 IEEE Transactions on Medical Imaging 27 (4), 425–441.
*/

#ifndef btkNLMTool_H
#define btkNLMTool_H


#include "itkImage.h"
#include "itkImageDuplicator.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkCastImageFilter.h"

#include "string"
#include "iomanip"
#include "sstream"
#include "fstream"

//WARNING:
//the possible types for T are float or double. Other types may introduce error during the computations 
namespace btk
{
/**
 * @class NLM Denoising
 * @brief This program implements a denoising method based on the work of Coupé et al. described in :
 Coupé, P., Yger, P., Prima, S., Hellier, P., Kervrann, C., Barillot, C., 2008.
 An optimized blockwise nonlocal means denoising filter for 3-D magnetic resonance images.
 IEEE Transactions on Medical Imaging 27 (4), 425–441.
 * @author François Rousseau
 * @ingroup Denoising
 */
template <typename TPixelType>
class NLMTool
{
 public:
    /**
     * @brief Image type.
     */
  typedef typename itk::Image< TPixelType, 3> itkTImage;
    /**
     * @brief Pointer to image type
     */
  typedef typename itkTImage::Pointer itkTPointer;
    /**
     * @brief itk DuplicatorFilter type
     */
  typedef typename itk::ImageDuplicator< itkTImage > itkTDuplicator;
    /**
     * @brief itk ImageRegionIterator type
     */
  typedef typename itk::ImageRegionIterator< itkTImage > itkTIterator;
    /**
     * @brief itk ImageregionConstIterator type
     */
  typedef typename itk::ImageRegionConstIterator< itkTImage > itkTConstIterator;
    /**
     * @brief itk ImageregionIteratorWithIndex type
     */
  typedef typename itk::ImageRegionIteratorWithIndex< itkTImage > itkTIteratorWithIndex;

    /**
     * @brief Set Input Image
     * @param inputImage Image to set in Input
     */
  void SetInput(itkTPointer inputImage);
  /**
   * @brief Set Reference Image
   * @param inputImage Image to set in reference
   */
  void SetReferenceImage(itkTPointer refImage);
  /**
   * @brief Todo
   * @param Todo
   */
  void SetDefaultParameters();
  /**
   * @brief Todo
   * @param Todo
   */
  void SetPatchSize(int h);
  /**
   * @brief Todo
   * @param Todo
   */
  void SetSpatialBandwidth(int s);
  /**
   * @brief Todo
   * @param Todo
   */
  void SetPaddingValue(float padding);
  /**
   * @brief Todo
   * @param Todo
   */
  void SetMaskImage(itkTPointer maskImage);
  /**
   * @brief Todo
   * @param Todo
   */
  void SetCentralPointStrategy(int s);
  /**
   * @brief Todo
   * @param Todo
   */
  void SetBlockwiseStrategy(int b);
  /**
   * @brief Todo
   * @param Todo
   */
  void SetOptimizationStrategy(int o);
  /**
   * @brief Todo
   * @param Todo
   */
  void SetLowerThresholds(float m, float v);
  /**
   * @brief Todo
   * @param Todo
   */
  double ComputePseudoResidual(typename itkTImage::IndexType & pixelIndex);
  /**
   * @brief Todo
   * @param Todo
   */
  double ComputePseudoResidualSafely(typename itkTImage::IndexType & pixelIndex);
  /**
   * @brief Todo
   * @param Todo
   */
  float MADEstimation(std::vector<float> & vecei, float & beta);
  /**
   * @brief Todo
   * @param Todo
   */
  void SetSmoothing(float beta); //set the smoothing parameter and compute the corrected smoothing value (m_rangeBandwidth)
  /**
   * @brief Todo
   * @param Todo
   */
  void SetLocalSmoothing(float beta); //set the smoothing parameter for every voxel and compute the corrected smoothing value (m_rangeBandwidth)
  /**
   * @brief Todo
   * @param Todo
   */
   itkTPointer GetOutput();
  /**
   * @brief Todo
   * @param Todo
   */
  void ComputeOutput();
  /**
   * @brief Todo
   * @param Todo
   */
  void PrintInfo();
  /**
   * @brief Todo
   * @param Todo
   */
  void CreatePatch(itkTPointer & patch);
  /**
   * @brief Todo
   * @param Todo
   */
  void GetPatch(typename itkTImage::IndexType p, itkTPointer & patch);
  /**
   * @brief Todo
   * @param Todo
   */
  void GetPatchFromReferenceImage(typename itkTImage::IndexType p, itkTPointer & patch);
  /**
   * @brief Todo
   * @param Todo
   */
  void AddPatchToImage(typename itkTImage::IndexType p, itkTPointer & patch, itkTPointer & image, itkTPointer & weightImage, double weight);
  /**
   * @brief Todo
   * @param Todo
   */
  double PatchDistance(itkTPointer & p,itkTPointer & q);
  /**
   * @brief Todo
   * @param Todo
   */

  double GetDenoisedPatch(typename itkTImage::IndexType p, itkTPointer & patch);
  /**
   * @brief Todo
   * @param Todo
   */
  double GetDenoisedPatchUsingTheReferenceImage(typename itkTImage::IndexType p, itkTPointer & patch);
  /**
   * @brief Todo
   * @param Todo
   */
  void ComputeSearchRegion(typename itkTImage::IndexType p, typename itkTImage::RegionType & region);
  /**
   * @brief Todo
   * @param Todo
   */
  void ComputePatchRegion(typename itkTImage::IndexType p, typename itkTImage::RegionType & imageRegion, typename itkTImage::RegionType & patchRegion);
  /**
   * @brief Todo
   * @param Todo
   */
  bool CheckSpeed(typename itkTImage::IndexType p, typename itkTImage::IndexType q);

protected:

  itkTPointer m_inputImage;/**< Pointer to input Image */
  itkTPointer m_outputImage;/**< Pointer to outputImage */
  itkTPointer m_maskImage;/**< Pointer to mask Image */
  itkTPointer m_refImage;/**< Pointer to reference Image */
  itkTPointer m_meanImage;/**< Pointer to mean Image */
  itkTPointer m_varianceImage;/**< Pointer to variance Image */
  itkTPointer m_rangeBandwidthImage;/**< Pointer to range Bandwidth Image */

  //Image information (size, spacing etc.)
  typename itkTImage::SpacingType m_spacing; /**< spacing */
  typename itkTImage::SizeType    m_size;/**< size */
  typename itkTImage::RegionType  m_region;/**< region */

private :

  typename itkTImage::SizeType m_halfPatchSize;          /**< half of the patch size*/
  typename itkTImage::SizeType m_fullPatchSize;          /**< patch size  : 2 * halfPatchSize + 1*/
  typename itkTImage::SizeType m_halfSpatialBandwidth;   /**< equivalent to the half size of the volume search area in non-local means*/
  typename itkTImage::SizeType m_fullSpatialBandwidth;   /**< spatial bandwidth : 2 * halfSpatialBandwidth + 1*/

  float m_padding; /**< float value of padding */
  int   m_centralPointStrategy; /**< todo */
  int   m_blockwise;/**< todo */
  int   m_optimized;/**< todo */
  float m_lowerMeanThreshold;/**< todo */
  float m_lowerVarianceThreshold;/**< todo */
  bool  m_useTheReferenceImage;/**< todo */
  bool  m_useGlobalSmoothing;/**< todo */
};
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "btkNLMTool.txx"
#endif


#endif

