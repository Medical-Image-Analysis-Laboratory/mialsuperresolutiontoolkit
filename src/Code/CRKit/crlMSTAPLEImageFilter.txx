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

#ifndef _crlMSTAPLEImageFilter_txx
#define _crlMSTAPLEImageFilter_txx
#include "crlMSTAPLEImageFilter.h"

#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkStatisticsImageFilter.h"

// This brings in types needed for modifying the pipeline.
#include "itkProcessObject.h"

// Enable control of printed number precision
#include <iomanip>
// Import definition of DBL_MIN
#include <limits.h>
#include <float.h>

namespace crl
{

  template <typename TInputImage, typename TOutputImage>
  void
  MSTAPLEImageFilter<TInputImage, TOutputImage>
  ::PrintSelf(std::ostream& os, itk::Indent indent) const
  {
    Superclass::PrintSelf(os,indent);
    os << indent << "m_MaximumIterations = " << m_MaximumIterations << std::endl;
    os << indent << "m_MaxLabel = " << m_MaxLabel << std::endl;
    os << indent << "m_ElapsedIterations = " << m_ElapsedIterations << std::endl;
    os << indent << "m_RelativeConvergenceThreshold = " <<
      m_RelativeConvergenceThreshold << std::endl;
  }

  //
  // This is very similar to the itkProcessObject, but the internal pipeline
  // is triggered to propogate the requestion region and
  // generate the output information earlier than would normally happen.
  //
  template< typename TInputImage, typename TOutputImage >
  void
  MSTAPLEImageFilter< TInputImage, TOutputImage >
  ::UpdateOutputInformation()
  {
    unsigned long t1, t2;
    // typedef typename DataObject::Pointer DataObjectPointer;
    // std::vector<DataObjectPointer>::size_type idx;
    itk::DataObject *input;
    itk::DataObject *output;

    /**
     * Watch out for loops in the pipeline
     */
    if ( this->m_Updating )
      {
	/**
	 * Since we are in a loop, we will want to update. But if
	 * we don't modify this filter, then we will not execute
	 * because our OutputInformationMTime will be more recent than
	 * the MTime of our output.
	 */
	this->Modified();
	return;
      }

    /**
     * We now wish to set the PipelineMTime of each output DataObject to
     * the largest of this ProcessObject's MTime, all input DataObject's
     * PipelineMTime, and all input's MTime.  We begin with the MTime of
     * this ProcessObject.
     */
    t1 = this->GetMTime();

    /**
     * Loop through the inputs
     */
    for (unsigned int idx = 0; idx < this->GetNumberOfInputs(); ++idx)
      {
	if (this->GetInput(idx))
	  {
	    input = const_cast<TInputImage *>(this->GetInput(idx));

	    /**
	     * Propagate the UpdateOutputInformation call
	     */
	    this->m_Updating = true;
	    input->UpdateOutputInformation();
	    //
	    // This is here in this class so that the input data voxels are read
	    // in and valid for the pipeline by the time GenerateOutputInformation
	    // is called, so that GenerateOutputInformation can utilize the voxel
	    // values of the input.
	    //
	    input->PropagateRequestedRegion();
	    input->UpdateOutputData();
	    this->m_Updating = false;

	    /**
	     * What is the PipelineMTime of this input? Compare this against
	     * our current computation to find the largest one.
	     */
	    t2 = input->GetPipelineMTime();

	    if (t2 > t1)
	      {
		t1 = t2;
	      }

	    /**
	     * Pipeline MTime of the input does not include the MTime of the
	     * data object itself. Factor these mtimes into the next PipelineMTime
	     */
	    t2 = input->GetMTime();
	    if (t2 > t1)
	      {
		t1 = t2;
	      }
	  }
      }

    // Record the number of input files.
    int numberOfInputFiles = this->GetNumberOfInputs();
    if (numberOfInputFiles <= 0) {
      ::std::cerr << "Did not receive any input files." << ::std::endl;
      exit(1);
    }

    // Let's try reading in the images
    typedef typename itk::StatisticsImageFilter< TInputImage > FilterType;
    typename FilterType::Pointer filter;
    filter = FilterType::New();

    // The range of voxel values in the first input defines the
    // range of labels that are evaluated.
    typename TInputImage::PixelType pixelmin;
    typename TInputImage::PixelType pixelmax;
    filter->SetInput( this->GetInput(0) );
    filter->Update();
    pixelmin = filter->GetMinimum();
    pixelmax = filter->GetMaximum();

    for (unsigned int i = 0; i < this->GetNumberOfInputs(); i++) {
      filter->SetInput( this->GetInput(i) );
      filter->Update();
      if (pixelmin > filter->GetMinimum()) pixelmin = filter->GetMinimum();
      if (pixelmax < filter->GetMaximum()) pixelmax = filter->GetMaximum();
    }
    m_MaxLabel = pixelmax;

    /**
     * Call GenerateOutputInformation for subclass specific information.
     * Since UpdateOutputInformation propagates all the way up the pipeline,
     * we need to be careful here to call GenerateOutputInformation only if
     * necessary. Otherwise, we may cause this source to be modified which
     * will cause it to execute again on the next update.
     */
    if (t1 > this->m_OutputInformationMTime.GetMTime())
      {
	for (unsigned int idx = 0; idx < this->GetNumberOfOutputs(); ++idx)
	  {
	    output = this->GetOutput( idx );
	    if (output)
	      {
		output->SetPipelineMTime(t1);
	      }
	  }

	this->GenerateOutputInformation();

	/**
	 * Keep track of the last time GenerateOutputInformation() was called
	 */
	this->m_OutputInformationMTime.Modified();
      }

  }



  template< typename TInputImage, typename TOutputImage >
  void
  MSTAPLEImageFilter< TInputImage, TOutputImage >
  ::GenerateOutputInformation()
  {
    typename Superclass::OutputImagePointer      outputPtr = this->GetOutput();
    typename Superclass::InputImageConstPointer  inputPtr  = this->GetInput(0);

    if ( !outputPtr || !inputPtr) {
      return;
    }

    // Set the output image largest possible region.  Use a RegionCopier
    // so that the input and output images can be different dimensions.
    OutputImageRegionType outputLargestPossibleRegion;
    this->CallCopyInputRegionToOutputRegion(outputLargestPossibleRegion,
					    inputPtr->GetLargestPossibleRegion());
    typedef typename Superclass::OutputImageType::SizeType SizeType;
    SizeType size;
    size = outputLargestPossibleRegion.GetSize();

    size[TOutputImage::ImageDimension-1] = m_MaxLabel + 1;
    outputLargestPossibleRegion.SetSize(size);

    outputPtr->SetLargestPossibleRegion( outputLargestPossibleRegion );

    // Set the output spacing and origin
    const itk::ImageBase<Superclass::InputImageDimension> *phyData;

    phyData
      = dynamic_cast<const itk::ImageBase<Superclass::InputImageDimension>*>(this->GetInput());

    if (phyData)
      {
	// Copy what we can from the image from spacing and origin of the input
	// This logic needs to be augmented with logic that select which
	// dimensions to copy
	unsigned int i, j;
	const typename InputImageType::SpacingType&
	  inputSpacing = inputPtr->GetSpacing();
	const typename InputImageType::PointType&
	  inputOrigin = inputPtr->GetOrigin();
	const typename InputImageType::DirectionType&
	  inputDirection = inputPtr->GetDirection();

	typename OutputImageType::SpacingType outputSpacing;
	typename OutputImageType::PointType outputOrigin;
	typename OutputImageType::DirectionType outputDirection;

	// copy the input to the output and fill the rest of the
	// output with zeros.
	for (i=0; i < Superclass::InputImageDimension; ++i)
	  {
	    outputSpacing[i] = inputSpacing[i];
	    outputOrigin[i] = inputOrigin[i];
	    for (j=0; j < Superclass::OutputImageDimension; j++)
	      {
		if (j < Superclass::InputImageDimension)
		  {
		    outputDirection[j][i] = inputDirection[j][i];
		  }
		else
		  {
		    outputDirection[j][i] = 0.0;
		  }
	      }
	  }
	for (; i < Superclass::OutputImageDimension; ++i)
	  {
	    outputSpacing[i] = 1.0;
	    outputOrigin[i] = 0.0;
	    for (j=0; j < Superclass::OutputImageDimension; j++)
	      {
		if (j == i)
		  {
		    outputDirection[j][i] = 1.0;
		  }
		else
		  {
		    outputDirection[j][i] = 0.0;
		  }
	      }
	  }
	// set the spacing and origin
	outputPtr->SetSpacing( outputSpacing );
	outputPtr->SetOrigin( outputOrigin );
	outputPtr->SetDirection( outputDirection );
      }
    else
      {
	// pointer could not be cast back down
	itkExceptionMacro(<< "itk::MSTAPLEImageFilter::GenerateOutputInformation "
			  << "cannot cast input to "
			  << typeid(itk::ImageBase<Superclass::InputImageDimension>*).name() );
      }
  }


  template< typename TInputImage, typename TOutputImage >
  void
  MSTAPLEImageFilter< TInputImage, TOutputImage >
  ::EstimateReferenceStandard()
  {
    typedef typename TInputImage::RegionType InRegionType;
    typedef typename TInputImage::SizeType   InSizeType;
    typedef typename TInputImage::IndexType   InIndexType;
    typedef typename TInputImage::ConstPointer InConstPointer;
    typedef itk::ImageRegionConstIteratorWithIndex< TInputImage > InIteratorType;

    typedef typename TOutputImage::RegionType OutRegionType;
    typedef typename TOutputImage::SizeType   OutSizeType;
    typedef typename TOutputImage::IndexType  OutIndexType;
    typedef typename TOutputImage::Pointer    OutImagePointer;
    typedef itk::ImageRegionIteratorWithIndex< TOutputImage >
      OutRegionIteratorType;

    /*
     * Given the expert performance parameters and the segmentations,
     * estimate for each voxel Pr(Ti=t|Dij=d, \hat{\theta}).
     */
    // Here we go to some length to cope with potentially very very small
    // numbers due to computing the product of many probabilities.
    // However, there is an accuracy versus time tradeoff.

    // First compute f(Ti=t)\prod_j f(Dij | Ti=t, \theta) for each label t
    // Then normalize the weight for each label t by the sum over all t.

    // pow(0.1, 10) is an heuristic limit for the smallest representable
    // product of probabilities in a double precision number. It serves as a
    // suggested threshold for switching to calculation to preserve precision.
    //
    // The numerical problem that can arise is that intermediate calculations
    // of the contribution to the weight for a particular class can underflow
    // due to multiplying by many small numbers, when the true product should
    // not be zero. This is avoided by instead summing the logarithm of the
    // terms and then scaling so the largest intermediate value becomes 1.0.
    //
    // Very large numbers of experts can require more extreme steps to maintain
    // appropriate precision.
    //
    if (m_UnderflowProtection != 0) {
      std::cout << "Using underflow protection mode : " ;
      if (m_UnderflowProtection == 1) {
	std::cout << "strong" ;
      } else {
	std::cout << "extreme" ;
      }
      std::cout << std::endl;
    }

    OutIndexType outIndex;
    InIndexType inIndex;
    inIndex.Fill( static_cast<typename InIndexType::IndexValueType>(0.0) );
    OutIndexType outIndexWrite;
    outIndexWrite.Fill( 
                    static_cast<typename OutIndexType::IndexValueType>(0.0) );
    OutImagePointer W = this->GetOutput();
    // Set up an output region that walks over the N-1 dimensions of the output
    OutRegionType outRegion = W->GetLargestPossibleRegion();
    outRegion.SetSize(W->GetImageDimension()-1, 1);

    double *significands = (double *)malloc(sizeof(double)*(m_MaxLabel+1));
    assert(significands != 0);
    int *exponents = (int *)malloc(sizeof(int)*(m_MaxLabel+1));
    assert(exponents != 0);

    OutRegionIteratorType outItr(W, outRegion);
    outItr.GoToBegin();
    while (!outItr.IsAtEnd()) { // loop over space
      outIndex = outItr.GetIndex();
      for (unsigned int i = 0;i < this->GetInput(0)->GetImageDimension(); i++) {
        inIndex[i] = outIndex[i];
        outIndexWrite[i] = outIndex[i];
      }
      // Do no processing of voxels not in the ROI.
      if (m_ROIImage->GetPixel(inIndex) == 0) { ++outItr; continue; }
      outRegion.SetSize(W->GetImageDimension()-1, 1);
      long double sumweights = 0.0;
      long double maxcondprob = -1.0;
      long double condprob = 0.0;
      memset(significands, 0, sizeof(double)*(m_MaxLabel+1));
      memset(exponents, 0, sizeof(int)*(m_MaxLabel+1));
      // Loop over label values
      for (unsigned int i = 0; ( i < (m_MaxLabel+1)); i++) {
	condprob = 0.0;
        outIndexWrite[W->GetImageDimension()-1] = i;
        long double localPrior = m_StationaryPriorWeight*m_StationaryPrior[i] +
         (1-m_StationaryPriorWeight)*m_LocalPriorImage->GetPixel(outIndexWrite);
	if (m_UnderflowProtection == 2) {
	  // Extreme steps to avoid underflow
	  double tsignificand;
	  int texponent;
	  // tsignificand = frexp( m_StationaryPrior[i], &texponent);
	  tsignificand = frexp( localPrior, &texponent);
	  significands[i] = 1.0*tsignificand;
	  exponents[i] += texponent;
	  for (unsigned int j = 0; j < this->GetNumberOfInputs(); j++) {
	    int decision = this->GetInput(j)->GetPixel( inIndex );
	    // multiply by probability of decision when truth is label i
	    double parm = (*m_Performance[j])[i*(m_MaxLabel+1)+decision];
	    tsignificand = frexp( parm, &texponent);
	    significands[i] *= tsignificand;
	    exponents[i] += texponent;
	  }
	  // The calculated intermediate results are left in the significands
	  // and exponents arrays.
	} else if (m_UnderflowProtection == 1) {
	  // Compute the sum of the logarithm of the terms
	  condprob = logl(localPrior);
	  for (unsigned int j = 0; j < this->GetNumberOfInputs(); j++) {
	    int decision = this->GetInput(j)->GetPixel( inIndex );
	    // multiply by probability of decision when truth is label i
	    double parm = (*m_Performance[j])[i*(m_MaxLabel+1)+decision];
	    condprob += logl(parm);
	  }
	} else { // m_UnderflowProtection == 0 or otherwise
	  // Compute the product directly.
	  condprob = localPrior;
	  for (unsigned int j = 0; j < this->GetNumberOfInputs(); j++) {
	    int decision = this->GetInput(j)->GetPixel( inIndex );
	    // multiply by probability of decision when truth is label i
	    double parm = (*m_Performance[j])[i*(m_MaxLabel+1)+decision];
	    condprob *= parm;
	  }
	}
        W->SetPixel(outIndexWrite, condprob);
        sumweights += condprob;
        if (condprob > maxcondprob) maxcondprob = condprob;
      }
      if (m_UnderflowProtection == 2) {
	// Using the new representation for underflow protection
	// Identify the largest exponent
	// Add -1*max exponent to each exponent
	// sum the weights
	// Divide each by the sum
	// Assign the weights
	int largestexponent = exponents[0];
	for (unsigned int i = 0; i < (m_MaxLabel+1); i++) {
	  if (exponents[i] > largestexponent) {
	    largestexponent = exponents[i];
	  }
	}
	for (unsigned int i = 0; i < (m_MaxLabel+1); i++) {
	  exponents[i] -= largestexponent;
	}
	// Now that the weights have been normalized, let's sum them
	sumweights = 0.0;
	for (unsigned int i = 0; i < (m_MaxLabel+1); i++) {
	  // reuse significands array to store individual weights
	  significands[i] = ldexp(significands[i], exponents[i]);
	  sumweights += significands[i];
	}
	for (unsigned int i = 0; i < (m_MaxLabel+1); i++) {
	  outIndexWrite[W->GetImageDimension()-1] = i;
	  W->SetPixel( outIndexWrite, significands[i] );
	}
        // At the end of this calculation, sumweights is correct and
        // the output pixel array is set to the non-normalized weight values.
      } else if (m_UnderflowProtection == 1) {
        if ( (expl(maxcondprob) <= 0.0) ) {
          std::cerr << "maxcondprob is " << expl(maxcondprob) << std::endl;
        }
        // Scale the logarithm of the terms by dividing by the magnitude of
        // the largest term, which is achieved by exponentiating with the
        // log term and the normalization factor.
        //     We also calculate the correct sumweights value here.
        double normfactor = logl(1.0) - maxcondprob;
        double scaledterm = 0.0;
        sumweights = 0.0;
        for (unsigned int i = 0; i < (m_MaxLabel+1); i++) {
          outIndexWrite[W->GetImageDimension()-1] = i;
          scaledterm = exp(W->GetPixel( outIndexWrite ) + normfactor);
          W->SetPixel( outIndexWrite , scaledterm );
          sumweights += scaledterm;
        }
        if (sumweights <= 0.0) {
          std::cout << "Underflow has occurred - please recalculate " <<
	    "using extreme underflow protection." << std::endl;
          std::cout << "sum of weights is " << sumweights << std::endl;
          for (unsigned int j = 0;
	       j < this->GetInput(0)->GetImageDimension(); j++) {
            std::cout << "index " << j << " is " << outIndexWrite[j] << " ";
          }
          std::cout << std::endl;
          std::cout << "sumweights is now " << sumweights << std::endl;
          std::cout << "normfactor is " << normfactor << std::endl;
          std::cout << "log maxcondprob is " << maxcondprob << std::endl;
          std::cout << "maxcondprob is " << expl(maxcondprob) << std::endl;
        }
        // At the end of this calculation, sumweights is correct and
        // the output pixel array is set to the non-normalized weight values.
      }
      // Normalize the weights to sum from zero to 1.0
      for (unsigned int i = 0; i < (m_MaxLabel+1); i++) {
        outIndexWrite[W->GetImageDimension()-1] = i;
        W->SetPixel( outIndexWrite, W->GetPixel(outIndexWrite)/sumweights );
      }
      ++outItr;
    } // End loop over space

    free(significands);
    free(exponents);

  }


  template< typename TInputImage, typename TOutputImage >
  void
  MSTAPLEImageFilter< TInputImage, TOutputImage >
  ::EstimatePerformanceParameters()
  {
    if (m_MAPStaple) {
      EstimatePerformanceParametersMAP();
      return;
    }

    typedef typename TInputImage::RegionType InRegionType;
    typedef typename TInputImage::SizeType   InSizeType;
    typedef typename TInputImage::IndexType   InIndexType;
    typedef typename TInputImage::ConstPointer InConstPointer;
    typedef itk::ImageRegionConstIteratorWithIndex< TInputImage > InIteratorType;

    typedef typename TOutputImage::RegionType OutRegionType;
    typedef typename TOutputImage::SizeType   OutSizeType;
    typedef typename TOutputImage::IndexType  OutIndexType;

    // Implement Equation 24 Warfield,Zou,Wells IEEE TMI 2004
    for (unsigned int i = 0; i < this->GetNumberOfInputs(); i++) {
      for (unsigned int v = 0; v < m_Performance[i]->size(); v++) {
        (*m_Performance[i])[v] = 0.0;
      }
      InRegionType inRegion = this->GetInput(i)->GetLargestPossibleRegion();
      InIndexType inIdx;
      OutIndexType outIdx;
      outIdx.Fill( static_cast<typename OutIndexType::IndexValueType>(0.0) );
      InIteratorType inItr(this->GetInput(i), inRegion);
      inItr.GoToBegin();
      while (!inItr.IsAtEnd()) {
	inIdx = inItr.GetIndex();
        // Do no processing of voxels not in the ROI.
        if (m_ROIImage->GetPixel(inIdx) == 0) { ++inItr; continue; }
	for (unsigned int k = 0; k < 
                               this->GetInput(i)->GetImageDimension();k++) {
	  outIdx[k] = inIdx[k];
	}
	int decision = inItr.Value();
	// Compute the credit the segmenter gets for this decision
	for (unsigned int j = 0; j < (m_MaxLabel + 1); j++) {
	  // Pr(D = d | T = j)
	  outIdx[this->GetOutput()->GetImageDimension()-1] = j;
	  (*m_Performance[i])[j*(m_MaxLabel+1) + decision] +=
	    this->GetOutput()->GetPixel(outIdx);
	}
	++inItr;
      }
    }

#ifdef WIN32
		double* sum;
		sum = (double*)malloc(sizeof(double)*(m_MaxLabel+1));
		assert(sum);
#else
    double sum[m_MaxLabel+1];
#endif
    // Now normalize by sum over all decisions for each truth class at voxels
    // Ensure Pr(Dij = decision|T=t) sums to one over all decision labels.
    for (unsigned int i = 0; i < m_Performance.size(); i++) {
      for (unsigned int t = 0; t < (m_MaxLabel+1); t++) {
	//double sum = 0.0;
	sum[t] = 0.0;
	for (unsigned int decision = 0; decision < (m_MaxLabel+1); decision++) {
	  sum[t] += (*m_Performance[i])[t*(m_MaxLabel+1)+decision];
	}
	if (sum[t] == 0.0) continue;
	for (unsigned int decision = 0; decision < (m_MaxLabel+1); decision++) {
	  (*m_Performance[i])[t*(m_MaxLabel+1) + decision] /= sum[t];
	}
      }
    }

#ifdef WIN32
		free(sum);
#endif
  }

  // This implements the performance parameters utilizing a prior 
  // probability for all elements of the performance matrix.
  template< typename TInputImage, typename TOutputImage >
  void
  MSTAPLEImageFilter< TInputImage, TOutputImage >
  ::EstimatePerformanceParametersMAP()
  {
    typedef typename TInputImage::RegionType InRegionType;
    typedef typename TInputImage::SizeType   InSizeType;
    typedef typename TInputImage::IndexType   InIndexType;
    typedef typename TInputImage::ConstPointer InConstPointer;
    typedef itk::ImageRegionConstIteratorWithIndex< TInputImage> InIteratorType;

    typedef typename TOutputImage::RegionType OutRegionType;
    typedef typename TOutputImage::SizeType   OutSizeType;
    typedef typename TOutputImage::IndexType  OutIndexType;

    // wnnp represents partial sums over the image for a particular input.
    double* wnnp = 0;
    wnnp = (double*)malloc(sizeof(double)*(m_MaxLabel+1)*(m_MaxLabel+1));
    assert(wnnp);

    // A prior for every parameter, but we set them all the same with
    // just different values for the diagonal and non-diagonal terms
    double* alphann = 0;
    alphann = (double*)malloc(sizeof(double)*(m_MaxLabel+1)*(m_MaxLabel+1));
    assert(alphann);

    double* betann = 0;
    betann = (double*)malloc(sizeof(double)*(m_MaxLabel+1)*(m_MaxLabel+1));
    assert(betann);
    for (unsigned int n = 0; n <= m_MaxLabel; n++) {
      for (unsigned int np = 0; np <= m_MaxLabel; np++) {
        // Prefer solutions with smaller off-diagonal terms and
        // larger diagonal terms.
        if (n != np) {  // Order reversed for off diagonal terms.
          alphann[n*(m_MaxLabel+1) + np] = GetMAPBeta();
          betann[n*(m_MaxLabel+1) + np] = GetMAPAlpha();
        } else { // Parameters for diagonal terms.
          alphann[n*(m_MaxLabel+1) + np] = GetMAPAlpha();
          betann[n*(m_MaxLabel+1) + np] = GetMAPBeta();
        }
      }
    }
std::cout << "beta distribution used: " << 
   " alpha " << GetMAPAlpha() << " and beta " << GetMAPBeta() << std::endl;

    double relativeconvergence = 1e-05;
    int maxiterations = 10;

    double* oldval = 0;
    oldval = (double*)malloc(sizeof(double)*(m_MaxLabel+1)*(m_MaxLabel+1));
    assert(oldval);
    double* newval = 0;
    newval = (double*)malloc(sizeof(double)*(m_MaxLabel+1)*(m_MaxLabel+1));
    assert(newval);
    double relativeerror = 0.0;

    // Implement Equation 24 Warfield,Zou,Wells IEEE TMI 2004
    for (unsigned int i = 0; i < this->GetNumberOfInputs(); i++) {
      for (unsigned int v = 0; v < m_Performance[i]->size(); v++) {
        (*m_Performance[i])[v] = 0.0;
      }
      InRegionType inRegion = this->GetInput(i)->GetLargestPossibleRegion();
      InIndexType inIdx;
      OutIndexType outIdx;
      outIdx.Fill( static_cast<typename OutIndexType::IndexValueType>(0.0) );
      InIteratorType inItr(this->GetInput(i), inRegion);
      inItr.GoToBegin();
      while (!inItr.IsAtEnd()) {
	inIdx = inItr.GetIndex();
        // Do no processing of voxels not in the ROI.
        if (m_ROIImage->GetPixel(inIdx) == 0) { ++inItr; continue; }
	for (unsigned int k = 0; k < 
                        this->GetInput(i)->GetImageDimension();k++) {
	  outIdx[k] = inIdx[k];
	}
	int decision = inItr.Value();
	// Compute the credit the segmenter gets for this decision
	for (unsigned int j = 0; j < (m_MaxLabel + 1); j++) {
	  // Pr(D = d | T = j)
	  outIdx[this->GetOutput()->GetImageDimension()-1] = j;
	  (*m_Performance[i])[j*(m_MaxLabel+1) + decision] +=
	    this->GetOutput()->GetPixel(outIdx);
	}
	++inItr;
      }
    }

    // Compute the initial estimate of the performance parameters,
    // assuming betann = 0, and then utilize the iterative update 
    // to identify the fixed point MAP solution.

  // The estimate is computed for each rater in turn.
    for (unsigned int i = 0; i < m_Performance.size(); i++) {

      // We are now going to store the partial sums computed above for
      // this rater for later reuse.
      // wnnp = \sum_{i:Dij=n'} W_{ni}^{t}
      for (unsigned int d = 0; d < (m_MaxLabel+1); d++) {
        for (unsigned int t = 0; t < (m_MaxLabel+1); t++) {
	  wnnp[t*(m_MaxLabel+1)+d] = 
                  (*m_Performance[i])[t*(m_MaxLabel+1)+d];
        }
      }
      for (unsigned int t = 0; t < (m_MaxLabel+1); t++) {
        double sum = 0.0;
        // Compute each of the numerator terms.
        //   Form the sum of the above terms over all decisions.
        // to get the value of the denominator.
        for (unsigned int d = 0; d < (m_MaxLabel+1); d++) {
	  (*m_Performance[i])[t*(m_MaxLabel+1) + d] =
               wnnp[t*(m_MaxLabel+1)+d] + 
               alphann[t*(m_MaxLabel+1) + d] - 1.0 +
               betann[t*(m_MaxLabel+1) + d] - 1.0 ;
          sum += (*m_Performance[i])[t*(m_MaxLabel+1) + d];
        }
        if (sum == 0.0) continue;
        // Now store the initial performance estimate
	for (unsigned int d= 0; d< (m_MaxLabel+1); d++) {
	  (*m_Performance[i])[t*(m_MaxLabel+1) + d] /= sum;
std::cout << "First estimate at t " << t << " d " << d << " is " 
<< (*m_Performance[i])[t*(m_MaxLabel+1) + d] << std::endl;
	    oldval[t*(m_MaxLabel+1) + d] = 
                           (*m_Performance[i])[t*(m_MaxLabel+1) + d];
	}
      }
      // This provides an initial estimate. We now need to use a fixed point
      // iteration to find the rater specific performance estimate.
      maxiterations = 10;
      do {

    // Now we have generated an initial estimate for the performance parameters
    // using the specified prior parameters.
    //  We need to solve the iterative equation to account for all of the 
    // betann parameters that are != 1.

      // Reset the indicator of the relative convergence to be zero
      //  so the max from the previous iteration doesn't carry to here.
      relativeerror = 0.0;
      double denominator = 0.0;
        for (unsigned int t = 0; t < (m_MaxLabel+1); t++) {
          // compute each of the numerator terms.
          denominator = 0.0;
          for (unsigned int d = 0; d < (m_MaxLabel+1); d++) {
std::cout << "rater " << i << "t " << t << " d " << d << " oldval[t,d] == " << oldval[t*(m_MaxLabel+1) + d] << std::endl;

if ( (oldval[t*(m_MaxLabel+1) + d] <= 0.0) ||
     (oldval[t*(m_MaxLabel+1) + d] >= 1.0) ) {
  std::cout << "Prior probabilities on parameters allow performance estimate "
  << "of " << oldval[t*(m_MaxLabel+1) + d] << std::endl;
  if (oldval[t*(m_MaxLabel+1) + d] == 1.0) {
    oldval[t*(m_MaxLabel+1) + d] -= 1e-06;
  }
}
	    newval[t*(m_MaxLabel+1) + d] = wnnp[t*(m_MaxLabel+1)+d] + 
                        alphann[t*(m_MaxLabel+1) + d] - 1.0 +
                        betann[t*(m_MaxLabel+1) + d] - 1.0 +
    (betann[t*(m_MaxLabel+1) + d] - 1.0)/(oldval[t*(m_MaxLabel+1) + d] - 1.0);
            denominator += newval[t*(m_MaxLabel+1) + d];
          }
          for (unsigned int d = 0; d < (m_MaxLabel+1); d++) {
            if (denominator != 0.0) {
	      newval[t*(m_MaxLabel+1) + d] /= denominator;
            }
          }
        }
        for (unsigned int t = 0; t < (m_MaxLabel+1); t++) {
          if (denominator == 0.0) continue;
          double change = 0.0;
          // Now store the initial performance estimate
	  for (unsigned int d= 0; d< (m_MaxLabel+1); d++) {
            change = (newval[t*(m_MaxLabel+1) + d] - 
                   oldval[t*(m_MaxLabel+1) + d])/oldval[t*(m_MaxLabel+1) + d];
            if (change > relativeerror) relativeerror = change;
std::cout << "max change (rel conv) for seg " << i << " is " 
    << (relativeerror) << std::endl;
std::cout << "new val is " << newval[t*(m_MaxLabel+1) + d] << std::endl;
std::cout << "old val is " << oldval[t*(m_MaxLabel+1) + d] << std::endl;
              // Store the previous iteration result.
              oldval[t*(m_MaxLabel+1) + d] = newval[t*(m_MaxLabel+1) + d];

            }
	  }
std::cout << "MAP maxiterations remaining is " << (maxiterations - 1) << std::endl;
std::cout << "MAP largest relative change in performance value is " <<
  relativeerror << std::endl;
      } while ((--maxiterations > 0) && (relativeerror > relativeconvergence));

      // Now save the new performance parameter estimates:
      for (unsigned int d = 0; d < (m_MaxLabel+1); d++) {
        for (unsigned int t = 0; t < (m_MaxLabel+1); t++) {
          (*m_Performance[i])[t*(m_MaxLabel+1)+d] = newval[t*(m_MaxLabel+1)+d];
        }
      }

    }
    
    // Free memory for temporary arrays
    free(wnnp);
    free(alphann);
    free(betann);
    free(newval);
    free(oldval);
  }

  template< typename TInputImage, typename TOutputImage >
  double
  MSTAPLEImageFilter< TInputImage, TOutputImage >
  ::ExpertPerformanceTraceMean()
  {
    double trace = 0.0;
    unsigned int labelcount = 0;  // count number of diagonal elements
    for (unsigned int i = 0; i < m_Performance.size(); i++) {
      for (unsigned int j = 0; j < (m_MaxLabel+1); j++) {
	if (m_StationaryPrior[j] == 0.0) continue;
	trace += (*m_Performance[i])[j*(m_MaxLabel+1)+j];
	labelcount++;
      }
    }
    // Ideal value for 'trace' would be 1.0
    trace /= static_cast<double>(labelcount);
    return trace;
  }

  template< typename TInputImage, typename TOutputImage >
  void
  MSTAPLEImageFilter< TInputImage, TOutputImage >
  ::GenerateData()
  {
    /* This function is responsible for allocating the output.
     * ThreadedGenerateData is not.
     */

    typedef typename TInputImage::RegionType InRegionType;
    typedef typename TInputImage::SizeType   InSizeType;
    typedef typename TInputImage::IndexType   InIndexType;
    typedef typename TInputImage::ConstPointer InConstPointer;
    typedef itk::ImageRegionConstIteratorWithIndex< TInputImage> InIteratorType;

    typedef typename TOutputImage::RegionType OutRegionType;
    typedef typename TOutputImage::SizeType   OutSizeType;
    typedef typename TOutputImage::IndexType  OutIndexType;
    typedef typename TOutputImage::Pointer    OutImagePointer;
    typedef itk::ImageRegionIterator<TOutputImage> ProbabilityImageIteratorType;

    unsigned int i = 0;

    // My understanding is that this will allocate the output.  The pipeline
    // has been adjusted so that this has the desired size.
    try {
      this->AllocateOutputs();
    } catch (std::bad_alloc &exp) {
      std::cerr << "std::bad_alloc exception caught !" << std::endl;
      std::cerr << "too much memory needed...check the range of labels is OK" <<
	std::endl;
      exit(1);
    }

    OutImagePointer W = this->GetOutput();

    // Initialize the output to all 0's
    W->FillBuffer(0);

    // Construct an image to hold the local prior.
    m_LocalPriorImage = TOutputImage::New();
    m_LocalPriorImage->CopyInformation(W);
    m_LocalPriorImage->SetRegions(W->GetLargestPossibleRegion());
    m_LocalPriorImage->Allocate();
    m_LocalPriorImage->FillBuffer(0.0); // Assume all voxels are in the LocalPrior for now.

    InConstPointer inImage1 = this->GetInput(0);
    InRegionType   inRegion = inImage1->GetLargestPossibleRegion();
    InIteratorType                inputItr( this->GetInput(0), inRegion );

    // Construct an image to hold a potential automatically determined
    // region of interest image. Used when AssignConsensusVoxels is turned on.
    m_ROIImage = TInputImage::New();
    m_ROIImage->CopyInformation(inImage1);
    m_ROIImage->SetRegions(inImage1->GetLargestPossibleRegion());
    m_ROIImage->Allocate();
    m_ROIImage->FillBuffer(1); // Assume all voxels are in the ROI for now.

    // Will need to expose this as a parameter in the future.
    if (!m_InitialExpertPerformanceSet) {
      std::vector<double> ondiagonal(this->GetNumberOfInputs());
      for (unsigned int i = 0; i < ondiagonal.size(); i++) {
        ondiagonal[i] = 0.9999999999;
      }
      SetInitialExpertPerformance(ondiagonal);
    } else {
      if (m_InitialExpertPerformance.size() != this->GetNumberOfInputs()) {
        std::cout << "Number of initial performance parameters is " << 
           m_InitialExpertPerformance.size() << std::endl;
        std::cout << "Number of input segmentations is " << 
               this->GetNumberOfInputs() << std::endl;
        std::cout << "These must match, but they don't." << std::endl;
      }
      SetInitialExpertPerformance(m_InitialExpertPerformance);
    }

    double *last_parms = new double[m_MaxLabel*m_MaxLabel];
    for (unsigned int j = 0; j < m_MaxLabel*m_MaxLabel; j++) {
      last_parms[j] = -1.0;
    }

    // Use this to access the output indexes.
    OutIndexType outIndex;
    outIndex.Fill( static_cast<typename OutIndexType::IndexValueType>(0.0) );
    OutRegionType outRegion = W->GetLargestPossibleRegion();

    // Determine if the voxels for which all raters agree will be used in
    // the calculation (usual case) or if they will be skipped (sometimes 
    //  this can be a useful way of focusing on the area of wrong labelling
    //  and it can accelerate the calculation).
    if (this->GetAssignConsensusVoxels()) {
      // std::cout << "Assign consensus voxels is true" << std::endl;
      // for every voxel
      unsigned int label = 0;
      InIteratorType in(this->GetInput(0), inRegion);
      InIndexType id;
      in.GoToBegin();
      while (!in.IsAtEnd()) {
        id = in.GetIndex();
	label = this->GetInput(0)->GetPixel(id);
        m_ROIImage->SetPixel(id, 0); // assume it is outside the ROI.
	for (unsigned int j = 0; 
                         j < this->GetInput(0)->GetImageDimension(); j++) {
	  outIndex[j] = id[j];
	}
	outIndex[W->GetImageDimension() - 1] = label;
	W->SetPixel(outIndex, 1.0);
        // If this is a consensus pixel, we can set the prior to 1.0
        // We recompute the spatial prior for voxels in the ROI below.
        m_LocalPriorImage->SetPixel(outIndex, 1.0);
        for (unsigned int j = 1; j < this->GetNumberOfInputs(); j++) {
          if (label != this->GetInput(j)->GetPixel(id)) {
            m_ROIImage->SetPixel(id, 1); // It is in the ROI.
	    W->SetPixel(outIndex, 0.0);
	    m_LocalPriorImage->SetPixel(outIndex, 0.0);
            break;
          }
        }
        ++in; // Increment to the next voxel.
      }
    } else {
      ; // std::cout << "Assign consensus voxels is false" << std::endl;
    }

    // Check if the stationary prior has been provided by the user (unusual),
    // and estimate it if it has not been (this is typically the best usage.)

    if (m_StationaryPriorSet) {
      // Check if the prior is set to the correct size and is plausible.
      if (m_StationaryPrior.size() != (m_MaxLabel + 1)) {
	// It has been set to the wrong size. Issue a warning and continue.
	std::cerr << "Stationary prior has been set by user to size " <<
	  m_StationaryPrior.size() << " but size should be " << (m_MaxLabel + 1)
		  << std::endl;
	std::cerr << "WARNING: Recalculating priors from the input data."
		  << std::endl;
	m_StationaryPriorSet = false;
      } else {
	// Check the priors sum to 1.0
	double sum = 0.0;
	for (unsigned int i = 0; i < m_StationaryPrior.size(); i++) {
	  sum += m_StationaryPrior[i];
	}
	if ( std::abs(sum - 1.0) > 0.000001 ) {
	  std::cerr << "Sum of the stationary priors must equal 1.0" << std::endl;
	  std::cerr << "Instead it is " << sum << std::endl;
	  std::cerr << "WARNING: Recalculating priors from the input data."
		    << std::endl;
	  m_StationaryPriorSet = false;
	}
      }
    }
    if (!m_StationaryPriorSet) {
      // Estimation the stationary prior from the input segmentations
      m_StationaryPrior.resize(m_MaxLabel+1);
      for (unsigned int i = 0; i < m_StationaryPrior.size(); i++) {
	m_StationaryPrior[i] = 0.0;
      }
    }

    // Come up with an initial Wi which is simply the average of
    // all the segmentations.
    long int labelcounter = 0;
    double priorvote = (1.0/this->GetNumberOfInputs());
    for (i = 0; i < this->GetNumberOfInputs(); ++i) {
      //std::cout << "inRegion :\n" << inRegion << std::endl;
      if ( this->GetInput(i)->GetRequestedRegion() != inRegion ) {
      //std::cout << "this->GetInput(" << i << ")->GetRequestedRegion()" << this->GetInput(i)->GetRequestedRegion() << std::endl;
	itkExceptionMacro(<<"One or more input images do not contain matching RequestedRegions");
      }
      InIteratorType in(this->GetInput(i), inRegion);
      in.GoToBegin();
      while (!in.IsAtEnd()) {
        // If the pixel is not in the ROI skip it.
        if (m_ROIImage->GetPixel(in.GetIndex()) == 0) { ++in; continue; }
	unsigned int label = in.Get();
	if (!m_StationaryPriorSet) {
	  m_StationaryPrior[label]++;   // COUNT FOR THE PRIOR MODELS.
	}
	InIndexType id = in.GetIndex();
	for (unsigned int j = 0; j < this->GetInput(0)->GetImageDimension(); j++) {
	  outIndex[j] = id[j];
	}
	outIndex[W->GetImageDimension() - 1] = label;
	// If the label is larger than the size then we are dead...
	if (label >= W->GetLargestPossibleRegion().GetSize()[W->GetImageDimension() - 1]) {
	  ::std::cout << "label value exceeds size of largest dimension of output image." << ::std::endl;
	  ::std::cout << "label value is " << label << std::endl;
	}
	W->SetPixel(outIndex, W->GetPixel(outIndex) + priorvote);
	m_LocalPriorImage->SetPixel(outIndex, 
                      m_LocalPriorImage->GetPixel(outIndex) + priorvote );
	++in;
	labelcounter++;
      }
    }  // end for

    if (!m_StationaryPriorSet) {
      for (unsigned int i = 0; i < m_StationaryPrior.size(); i++) {
	m_StationaryPrior[i] /= static_cast<float>(labelcounter);
      }
    }

    for (unsigned int i = 0; i < m_StationaryPrior.size(); i++) {
      std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(15);
      std::cout << "m_StationaryPrior["<<i<<"] " <<
	m_StationaryPrior[i] << std::endl;
    }
    // Print out the priors in a form that can be used with spreadsheets.
    std::cout << "SPREADSHEET,Stationary Priors";
    for (unsigned int i = 0; i < m_StationaryPrior.size(); i++) {
      std::cout << "," << m_StationaryPrior[i];
    }
    std::cout << std::endl;

    double magrelerror = 10.0;
    double prevmagrelerror = 1.0;
    double tracesum = 0.0;
    // Ideally the trace sum would approach the number of labelled structures
    // in the input.
    double tracesumprev = 10.0;
    unsigned int itncount = 0;
    bool flag = false;
    if (m_StartAtEStep) {
      EstimateReferenceStandard();
    }
    while ( (!flag) && (itncount < m_MaximumIterations) &&
	    (magrelerror > m_RelativeConvergenceThreshold) ) {
      EstimatePerformanceParameters();
      EstimateReferenceStandard();
      /* Useful for tracking evolution of the parameters :
	 std::cout << "Expert performance parameters:" << std::endl;
	 PrintExpertPerformance();
      */

      ++itncount;

      tracesumprev = tracesum;
      tracesum = ExpertPerformanceTraceMean();
      prevmagrelerror = magrelerror;
      assert(tracesum > 0);
      magrelerror = fabs( (tracesum - tracesumprev) / tracesum );

      std::cout << "Trace of expert parameters at iteration " <<
	itncount << " is " << tracesum << std::endl;
      std::cout << "Relative error magnitude is " << magrelerror <<
	std::endl;


      this->InvokeEvent( itk::IterationEvent() );

      if( this->GetAbortGenerateData() )
	{
	  this->ResetPipeline();
	  flag = true;
	}

      if (flag == true)
	{
	  break;
	}
    }

    // Copy p's, q's, etc. to member variables

    m_ElapsedIterations = itncount;

    std::cout << "Expert performance parameters:" << std::endl;
    PrintExpertPerformance();

    std::cout << "Summary of expert performance" << std::endl;
    PrintExpertPerformanceSummary();

    std::cout << "Posterior probabilities indicate expert performance" <<
      std::endl;
    PrintExpertPosteriors();
  }

  template< typename TInputImage, typename TOutputImage >
  void
  MSTAPLEImageFilter< TInputImage, TOutputImage >
  ::PrintExpertPerformance()
  {
    // Print for each rater the performance parameter : Pr(Dij = d | Ti = k)
    std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(15);
    for (unsigned int i = 0; i < m_Performance.size(); i++) {
      std::cout << "Performance of segmentation " << i << std::endl;
      for (unsigned int k = 0; k < (m_MaxLabel+1); k++) {
	if (m_StationaryPrior[k] == 0.0) continue;
	for (unsigned int d = 0; d < (m_MaxLabel+1); d++) {
	  if (m_StationaryPrior[d] == 0.0) continue;
	  std::cout << "Pr(D="<<d<<"|"<<"T="<<k<<") = " <<
	    (*m_Performance[i])[k*(m_MaxLabel+1)+d]<<" " ;
	}
	std::cout << std::endl;
      }
    }

    // It is useful to have this data in a form that can be imported into a
    // spreadsheet.
    // Print for each rater the performance parameter : Pr(Dij = d | Ti = k)
    std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(15);
    for (unsigned int i = 0; i < m_Performance.size(); i++) {
      for (unsigned int k = 0; k < (m_MaxLabel+1); k++) {
	if (m_StationaryPrior[k] == 0.0) continue;
	std::cout << "SPREADSHEET,Expert Performance Parameters(Pr(Dij = d | Ti = k))," << i ;
	std::cout << "," << k ;
	for (unsigned int d = 0; d < (m_MaxLabel+1); d++) {
	  if (m_StationaryPrior[d] == 0.0) continue;
	  std::cout << "," << d << "," << (*m_Performance[i])[k*(m_MaxLabel+1)+d];
	}
	std::cout << std::endl;
      }
      std::cout << std::endl;
    }


  }

  template< typename TInputImage, typename TOutputImage >
  void
  MSTAPLEImageFilter< TInputImage, TOutputImage >
  ::PrintExpertPerformanceSummary()
  {
    // Print the summary of the performance of each rater : Pr(Dij = d | Ti = d)
    // \sum_{e != d} Pr(Dij = e | Ti = d)
    std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(15);
    for (unsigned int i = 0; i < m_Performance.size(); i++) {
      std::cout << "Summary of performance of segmentation " << i << std::endl;
      for (unsigned int k = 0; k < (m_MaxLabel+1); k++) {
	double sum = 0.0;
	if (m_StationaryPrior[k] == 0.0) continue;
	for (unsigned int d = 0; d < (m_MaxLabel+1); d++) {
	  if (m_StationaryPrior[d] == 0.0) continue;
	  if (d == k) {
	    std::cout << "Pr(D="<<d<<"|"<<"T="<<k<<") = " <<
	      (*m_Performance[i])[k*(m_MaxLabel+1)+d]<<" " ;
	  } else {
	    // Add up the off-diagonal terms along index d
	    sum += (*m_Performance[i])[k*(m_MaxLabel+1)+d];
	  }
	}
	std::cout << "Pr(D != "<<k<<"|"<<"T="<<k<<") = " << sum ;
	std::cout << std::endl;
      }
    }
  }


  /* Compute and print the expert decision posterior probability given
   * the global priors for the label and the estimated conditional
   * probabilities Pr(D|L).
   *
   * Pr(L | D) Pr(D) = Pr(D|L) Pr(L)
   * Pr(L | D) = Pr(D|L) Pr(L) / \sum_L Pr(D|L) Pr(L)
   * since
   *   Pr(D) = \sum_L Pr(D|L) Pr(L)
   *
   */
  template< typename TInputImage, typename TOutputImage >
  void
  MSTAPLEImageFilter< TInputImage, TOutputImage >
  ::PrintExpertPosteriors()
  {
    // Print for each rater the performance parameter : Pr(Dij = d | Ti = k)
    std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(15);

    unsigned int numlabels = (m_MaxLabel+1);

    for (unsigned int patientindex = 0; patientindex < m_Performance.size();
	 patientindex++) {
      std::cout << "Performance of segmentation " << patientindex << std::endl;
      // We need a vector here, because standard C++ does not support
      // variable length arrays
      //double posteriors[numlabels*numlabels];
      std::vector<double> posteriors;
      posteriors.resize(numlabels*numlabels);

      for (unsigned int d = 0; d < numlabels; d++) { // loop over decisions
	if (m_StationaryPrior[d] == 0.0) continue;
	double lsum = 00.0;
	for (unsigned int t = 0; t < numlabels; t++) { // loop over truth
	  if (m_StationaryPrior[t] == 0.0) continue;
	  posteriors[t*numlabels + d] =
	    (*m_Performance[patientindex])[t*numlabels + d]*m_StationaryPrior[t];
	  lsum += posteriors[t*numlabels + d];
	}
	for (unsigned int t = 0; t < numlabels; t++) { // loop over truth
	  if (m_StationaryPrior[t] == 0.0) continue;
	  if (lsum == 0.0) {
	    posteriors[t*numlabels + d] = 0.0;
	  } else {
	    posteriors[t*numlabels + d] /= lsum;
	  }
	}
      }

      for (unsigned int d = 0; d < numlabels; d++) {
	if (m_StationaryPrior[d] == 0.0) continue;
	for (unsigned int t = 0; t < numlabels; t++) {
	  if (m_StationaryPrior[t] == 0.0) continue;
	  std::cout << "Pr(L= " << t << "|D= " << d << ") = "
		    << posteriors[t*numlabels + d] << ", " ;
	}
	std::cout << std::endl;
      }

      for (unsigned int t = 0; t < numlabels; t++) {
	if (m_StationaryPrior[t] == 0.0) continue;
	std::cout << "PV " << t << " is Pr(L = " << t << " | D = " << t
		  << ") = " << posteriors[t*numlabels + t] << std::endl;
      }
      std::cout << "SPREADSHEET,PV," << patientindex ;
      for (unsigned int t = 0; t < numlabels; t++) {
	if (m_StationaryPrior[t] == 0.0) continue;
	std::cout << "," << t << "," << posteriors[t*numlabels + t] ;
      }
      std::cout << std::endl;
    }
  }

  template< typename TInputImage, typename TOutputImage >
  bool
  MSTAPLEImageFilter< TInputImage, TOutputImage >
  ::SetInitialExpertPerformance(std::vector<double> ondiagonal)
  {
    // precondition: this->GetNumberOfInputs() == ondiagonal.size
    unsigned int numberOfLabels = m_MaxLabel + 1;
    m_Performance.resize(this->GetNumberOfInputs());
    for (unsigned int i = 0; i < this->GetNumberOfInputs(); i++) {
      m_Performance[i] = new PerfType(numberOfLabels*numberOfLabels);
      double offdiagonal = (1.0 - ondiagonal[i])/numberOfLabels;
      // Set initial estimate for Pr(Dij = d | Ti = k)
      // This must sum to one over d, which we want to have on the row.
      for (unsigned int d = 0; d < numberOfLabels; d++) {
	for (unsigned int k = 0; k < numberOfLabels; k++) {
	  if (d == k) {
	    (*m_Performance[i])[k*numberOfLabels + d] = ondiagonal[i];
	  } else {
	    (*m_Performance[i])[k*numberOfLabels + d] = offdiagonal;
	  }
	}
      }
    }
    return true;
  }

} // end namespace crl

#endif
