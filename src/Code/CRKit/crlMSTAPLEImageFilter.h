/*
 * Copyright (c) 2008-2011 Children's Hospital Boston.
 *
 * This software is licensed by the copyright holder under the terms of the
 * Open Software License version 3.0.
 * http://www.opensource.org/licenses/osl-3.0.php
 *
 * Attribution Notice.
 *
 * This research was carried out in the Computational Radiology Laboratory of
 * Children's Hospital, Boston and Harvard Medical School.
 * http://www.crl.med.harvard.edu
 * For more information contact: simon.warfield@childrens.harvard.edu
 *
 * This research work was made possible by Grant Number R01 RR021885 (Principal
 * Investigator: Simon K. Warfield, Ph.D.) to Children's Hospital, Boston
 * from the National Center for Research Resources (NCRR), a component of the
 * National Institutes of Health (NIH).
*/


#ifndef __crlMSTAPLEImageFilter_h
#define __crlMSTAPLEImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkImage.h>
#include <vector>

namespace crl
{
/** \class MSTAPLEImageFilter
 *
 * \brief The MSTAPLE filter implements the Simultaneous Truth and Performance
 * Level Estimation algorithm for generating ground truth volumes from a set of
 * segmentations. The segmentations may consist of binary labels,
 * or any finite number of categories.
 *
 * The STAPLE algorithm estimates a probabilistic reference standard for the
 * true segmentation of an image from a collection of segmentations by experts
 * or algorithms.  The reference standard is obtained by an optimal weighting
 * of the input segmentations, and the algorithm also identifies the optimal
 * weights.
 * The reference standard produced by this filter is a set of
 * floating point volumes of values between zero and one that indicate
 * probability of each pixel having a particular label.
 *
 * The STAPLE algorithm is described in
 *
 * S. Warfield, K. Zou, W. Wells, "Validation of image segmentation and expert
 * quality with an expectation-maximization algorithm" in MICCAI 2002: Fifth
 * International Conference on Medical Image Computing and Computer-Assisted
 * Intervention, Springer-Verlag, Heidelberg, Germany, 2002, pp. 298-306
 *
 * The multi-label version of the algorithm is described in
 * Warfield, Zou, Wells IEEE TMI 2004.
 *
 *
 * \par INPUTS
 * Input volumes to the STAPLE filter must be segmentations of the same image.
 * That is, there must be consistent set of label values and a consistent
 * geometry for each input image.
 *
 * Input volumes must all contain the same size RequestedRegions.
 *
 * \par OUTPUTS
 * The STAPLE filter produces a single output volume with a range of floating
 * point values from zero to one. IT IS VERY IMPORTANT TO INSTANTIATE THIS
 * FILTER WITH A FLOATING POINT OUTPUT TYPE (floats or doubles).
 * You may select the index of the component with the largest value if you
 * wish to produce an image of discrete labels as output.
 *
 * \par PARAMETERS
 * The STAPLE algorithm requires a number of inputs.  You may specify any
 * number of input volumes using the SetInput(i, p_i) method, where i ranges
 * from zero to N-1, N is the total number of input segmentations, and p_i is
 * the SmartPointer to the i-th segmentation.
 *
 * The STAPLE algorithm is an iterative Expectation-Maximization algorithm and
 * will converge on a solution after some number of iterations that cannot be
 * known a priori.
 * After updating the filter, the total elapsed iterations taken to converge on
 * the solution can be queried through GetElapsedIterations().  You may also
 * specify a MaximumNumberOfIterations, after which the algorithm will stop
 * iterating regardless of whether or not it has converged.  This
 * implementation of the STAPLE algorithm will find the solution to within
 * seven digits of precision unless it is stopped early.
 *
 * Once updated, the performance rates for each rater input volume can be
 * queried using GetPerformance(i), where i is the i-th input volume.
 *
 * \par REQUIRED PARAMETERS
 * The only required parameters for this filter are the input volumes.
 * All other parameters may be safely left to their default
 * values. Please see the paper cited above for more information on the STAPLE
 * algorithm and its parameters.  A proper understanding of the algorithm is
 * important for interpreting the results that it produces.
 *
 * \par EVENTS
 * This filter invokes IterationEvent() at each iteration of the E-M
 * algorithm. Setting the AbortGenerateData() flag will cause the algorithm to
 * halt after the current iteration and produce results just as if it had
 * converged. The algorithm makes no attempt to report its progress since the
 * number of iterations needed cannot be known in advance. */

template <typename TInputImage, typename TOutputImage>
class ITK_EXPORT MSTAPLEImageFilter :
    public itk::ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef MSTAPLEImageFilter Self;
  typedef itk::ImageToImageFilter< TInputImage, TOutputImage > Superclass;
  typedef itk::SmartPointer<Self> Pointer;
  typedef itk::SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods) */
  itkTypeMacro(MSTAPLEImageFilter, ImageToImageFilter);

  /** Extract some information from the image types.  Dimensionality
   * of the two images is assumed to be the same. */
  typedef typename TOutputImage::PixelType OutputPixelType;
  typedef typename TInputImage::PixelType InputPixelType;
  typedef typename itk::NumericTraits<InputPixelType>::RealType RealType;

  /** Extract some information from the image types.  Dimensionality
   * of the two images is assumed to be the same. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TOutputImage::ImageDimension);

  /** Image typedef support */
  typedef TInputImage  InputImageType;
  typedef TOutputImage OutputImageType;
  typedef typename InputImageType::Pointer InputImagePointer;
  typedef typename OutputImageType::Pointer OutputImagePointer;

  /** Superclass typedefs. */
  typedef typename Superclass::OutputImageRegionType OutputImageRegionType;

  /** After the filter is updated, this method returns a std::vector<double>
   * encoding the peformance parameter matrix corresponding to one of the
   * input segmentations.
   *  */
  const std::vector<double> &GetPerformance(unsigned int i) const
  {
    if (i > this->GetNumberOfInputs()) {
      itkExceptionMacro(<< "Array reference out of bounds.");
    }
    return (*m_Performance[i]);
  }

  /** To enable setting the default initial expert performance parameters
    * for each segmentation generator.
    */
  bool SetInitialExpertPerformance(std::vector<double> ondiagonal);

  /** To set the stationary prior for each tissue class.
    */
  bool SetStationaryPrior(std::vector<double> prior)
  {
    m_StationaryPriorSet = true;
    m_StationaryPrior = prior;
		return true;
  }

  // Set the initial expert performance.
  bool SetInitialExpertPerformanceParameters(std::vector<double> perf)
  {
    m_InitialExpertPerformanceSet = true;
    m_InitialExpertPerformance = perf;
    /* If this is not large enough to match the number of segmentations,
     * it is a problem.
     */
    return true;
  }

  /** Set/Get the maximum number of iterations after which the STAPLE algorithm
   *  will be considered to have converged.  In general this SHOULD NOT
   * be set and the algorithm should be allowed to converge on its own. */
  itkSetMacro(MaximumIterations, unsigned int);
  itkGetMacro(MaximumIterations, unsigned int);

  /** Set/Get the threshold for which a change in the relative magnitude of
   * the mean of the trace of expert performance parameters shall be so small
   * as to trigger termination of the estimation procedure.
   */
  itkSetMacro(RelativeConvergenceThreshold, double);
  itkGetMacro(RelativeConvergenceThreshold, double);

  /** Set/Get the weight for the stationary weight
   */
  itkSetMacro(StationaryPriorWeight, double);
  itkGetMacro(StationaryPriorWeight, double);

  /** Get the number of elapsed iterations of the iterative E-M algorithm. */
  itkGetMacro(ElapsedIterations, unsigned int);

  /** Set/Get the nature of underflow protection to be used.
   * Underflow protection is computationally expensive but important when
   * a large number of raters are being used. The default value is 0 which
   * does straightforward calculations, 1 represents strong underflow
   * protection, and 2 is extreme underflow protection.
   */
  itkSetMacro(UnderflowProtection, int);
  itkGetMacro(UnderflowProtection, int);

  /** Set/Get boolean to use or not use (default) compression in output files.
   */
  itkSetMacro(UseWriteCompression, bool);
  itkGetMacro(UseWriteCompression, bool);

  /** Set/Get boolean to use or not use (default) assign consensus voxels.
   * All voxels that have been labelled the same by every input segmentation
   * are assigned that label, and skipped in the computation of 
   * segmentation performance.
   */
  itkSetMacro(AssignConsensusVoxels, bool);
  itkGetMacro(AssignConsensusVoxels, bool);

  /** Set/Get boolean to use or not use (default) MAP estimation.
   */
  itkSetMacro(MAPStaple, bool);
  itkGetMacro(MAPStaple, bool);

  /** Set/Get alpha parameter for MAP estimation.
   */
  itkSetMacro(MAPAlpha, double);
  itkGetMacro(MAPAlpha, double);

  /** Set/Get beta parameter for MAP estimation.
   */
  itkSetMacro(MAPBeta, double);
  itkGetMacro(MAPBeta, double);

  /** Set to true to start with estimating the reference standard.
    * This means we initialize with the performance parameters.
    */
  void SetStartAtEStep(bool go)
  {
    // If this is set to true, we start with estimating the reference standard.
    // This means we initialize with the performance parameters.
    // If it is set to false, we start with estimating performance parameters. 
    // This means we initialize with the reference standard estimate.
    m_StartAtEStep = go;
  }

  /** Set to true to start with estimating the performance parameters.
    * This means we initialize with the reference standard.
    */
  void SetStartAtMStep(bool go)
  {
    // If this is set to true, we start with estimating the performance
    // parameters.
    // This means we initialize with the reference standard.
    // If it is set to false, we start with estimating the reference standard. 
    // This means we initialize with the performance parameters.
    m_StartAtEStep = !go;
  }

  /** Add comment describing purpose of function. */
  void EstimateReferenceStandard();

  /** Add comment describing purpose of function. */
  void EstimatePerformanceParameters();

  // This implements the performance parameters utilizing a prior
  // probability for all elements of the performance matrix.
  void EstimatePerformanceParametersMAP();

  /** Add comment describing purpose of function. */
  double ExpertPerformanceTraceMean();

protected:
  MSTAPLEImageFilter()
  {
    m_MaxLabel = itk::NumericTraits<InputPixelType>::One;
    m_MaximumIterations = itk::NumericTraits<unsigned int>::max();
    m_ElapsedIterations = 0;
    m_RelativeConvergenceThreshold = 5e-07;
    m_UnderflowProtection = 0;
    m_StationaryPriorSet = false;
    m_StationaryPriorWeight = 0.01;
    m_InitialExpertPerformanceSet = false;
    m_MAPStaple = false;
    m_ROIImage = 0;
    m_LocalPriorImage = 0;
    m_StartAtEStep = false;
  }

  virtual ~MSTAPLEImageFilter() {}
  void GenerateData( );

  void PrintSelf(std::ostream&, itk::Indent) const;

  void PrintExpertPerformance();

  void PrintExpertPerformanceSummary();

  void PrintExpertPosteriors();

  void GenerateOutputInformation();

  void UpdateOutputInformation();


private:
  MSTAPLEImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  // InputPixelType m_MaxLabel;
  unsigned int m_MaxLabel;
  unsigned int m_ElapsedIterations;
  unsigned int m_MaximumIterations;
  double       m_RelativeConvergenceThreshold;
  double       m_StationaryPriorWeight;
  int          m_UnderflowProtection;
  bool         m_UseWriteCompression;
  bool         m_AssignConsensusVoxels;
  bool         m_MAPStaple;
  double       m_MAPAlpha;
  double       m_MAPBeta;
  bool         m_StartAtEStep;

  typename TInputImage::Pointer m_ROIImage;
  typename TOutputImage::Pointer m_LocalPriorImage;

  typedef std::vector<double> PerfType;

  std::vector< PerfType * > m_Performance;

  bool m_StationaryPriorSet;
  std::vector<double> m_StationaryPrior;

  bool m_InitialExpertPerformanceSet;
  std::vector<double> m_InitialExpertPerformance;

};

} // end namespace crl

#ifndef ITK_MANUAL_INSTANTIATION
#include "crlMSTAPLEImageFilter.txx"
#endif

#endif
