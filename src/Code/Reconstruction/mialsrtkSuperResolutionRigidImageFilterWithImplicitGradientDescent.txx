/*==========================================================================

  © Université de Strasbourg - Centre National de la Recherche Scientifique

  Date: 02/12/2010
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

#ifndef __mialsrtkSuperResolutionRigidImageFilterWithImplicitGradientDescent_txx
#define __mialsrtkSuperResolutionRigidImageFilterWithImplicitGradientDescent_txx

#include "mialsrtkSuperResolutionRigidImageFilterWithImplicitGradientDescent.h"


namespace mialsrtk
{

/**
 * Initialize new instance
 */
template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage, TOutputImage   ,TInterpolatorPrecisionType>
::SuperResolutionRigidImageFilterWithImplicitGradientDescent()
{
  m_OutputSpacing.Fill(1.0);
  m_OutputOrigin.Fill(0.0);
  m_OutputDirection.SetIdentity();

  m_UseReferenceImage = false;

  m_Size.Fill( 0 );
  m_OutputStartIndex.Fill( 0 );

  m_DefaultPixelValue = itk::NumericTraits< PixelType >::ZeroValue();

  m_Iterations = 50;
  m_ConvergenceThreshold = 1e-4;
  m_Lambda = 0.1;
  m_PSF = GAUSSIAN;
  //m_PSF = BOXCAR;

  m_UseDebluringPSF = false;

  m_SliceGap = 0.0;

  m_CurrentOuterIteration = 0;
  m_CurrentBregmanLoop = 0;

  m_InitTime = 0.0;
  m_InnerOptTime = 0.0;
}

/**
 * Print out a description of self
 *
 * \todo Add details about this class
 */
template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
void
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage, TOutputImage   ,TInterpolatorPrecisionType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "DefaultPixelValue: "
     << static_cast<typename NumericTraits<PixelType>::PrintType>(m_DefaultPixelValue)
     << std::endl;
  os << indent << "Size: " << m_Size << std::endl;
  os << indent << "OutputStartIndex: " << m_OutputStartIndex << std::endl;
  os << indent << "OutputOrigin: " << m_OutputOrigin << std::endl;
  os << indent << "OutputSpacing: " << m_OutputSpacing << std::endl;
  os << indent << "OutputDirection: " << m_OutputDirection << std::endl;
  os << indent << "UseReferenceImage: " << (m_UseReferenceImage ? "On" : "Off") << std::endl;
  return;
}

template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
void
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::SetOutputSpacing(
  const double* spacing)
{
  SpacingType s(spacing);
  this->SetOutputSpacing( s );
}

/**
 * Return the current  value of the convergence criterion
 */
template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
double
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::GetCriterionValue()
{
  /*
  std::cout << "Get criterion :" << std::endl;
  std::cout << "Xold =" << m_xold.sum() << " (" << &m_xold << ")" << std::endl;
  std::cout << "X =" << m_x.sum() << " (" << &m_x << ")" << std::endl;
  std::cout << "diff X = " << (m_x - m_xold).sum() << std::endl;
  */
  double criterion = (m_x - m_xold).two_norm() / m_xold.two_norm();
  return criterion;
}

template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
void
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::AddInput(InputImageType* _arg)
{
  m_ImageArray.push_back(_arg);

  //this -> SetInput(_arg);

  // Add transforms for this image
  m_Transform.resize( m_Transform.size() + 1 );
  SizeType _argSize = _arg -> GetLargestPossibleRegion().GetSize();
  m_Transform[m_Transform.size()-1].resize(_argSize[2]);
  //std::cout << "TEST: m_transform size: " << m_Transform.size() << std::endl;
  //std::cout << "m_transform[]: " << m_Transform[m_Transform.size()-1].size() << std::endl;
 
  // Initialize transforms
  for (unsigned int i=0; i<_argSize[2]; i++)
    m_Transform[m_Transform.size()-1][i] = TransformType::New();
}

template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
typename SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>::IndexType
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::LinearToAbsoluteIndex(
  unsigned int linearIndex, InputImageRegionType region)
{
  IndexType absIndex;

  IndexType start = region.GetIndex();
  SizeType  size  = region.GetSize();
  IndexType diffIndex;

  diffIndex[2] = linearIndex / (size[0]*size[1]);

  diffIndex[1] = linearIndex - diffIndex[2]*size[0]*size[1];
  diffIndex[1] = diffIndex[1] / size[0];

  diffIndex[0] = linearIndex - diffIndex[2]*size[0]*size[1] - diffIndex[1]*size[0];

  absIndex[0] = diffIndex[0] + start[0];
  absIndex[1] = diffIndex[1] + start[1];
  absIndex[2] = diffIndex[2] + start[2];

  return absIndex;
}

/**
 * Run the TV-based optimization algorithm.
 *
 * The algorithm uses an accelerated primal-dual hybrid gradient method based on [], [], [].
 * The solution of the inner least-square problem is computed using a semi-implicit gradient descent scheme.
 */
template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
void
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::Optimize()
{
  //std::cout << "Begin OptimizeLeastSquare()" << std::endl << std::endl;
  // Fill x
  m_OutputImageRegion = this -> GetReferenceImage() -> GetLargestPossibleRegion();

  /*
  std::cout << "Reference image pointer : " << this -> GetReferenceImage() << std::endl;
  std::cout << "m_OutputImageRegion : " << m_OutputImageRegion << std::endl;
  */

  unsigned int ncols = m_OutputImageRegion.GetNumberOfPixels();

  m_x.set_size( ncols );
  OutputIteratorType hrIt( this -> GetReferenceImage(), m_OutputImageRegion );
  unsigned int linearIndex = 0;

  for (hrIt.GoToBegin(); !hrIt.IsAtEnd(); ++hrIt, linearIndex++)
    m_x[linearIndex] = hrIt.Get();

  SizeType size = m_OutputImageRegion.GetSize();
  vnl_vector<int>  x_size(3);
  x_size[0] = size[0]; x_size[1] = size[1]; x_size[2] = size[2];

  // Setup cost function
  //std::cout << "Setup the Costfunction" << std::endl << std::endl;
  TotalVariationCostFunctionWithImplicitGradientDescent<InputImageType> f(m_x.size());

  for(unsigned int im = 0; im < m_ImageArray.size(); im++)
  {
    f.AddImage(m_ImageArray[im]);
    f.AddRegion(m_InputImageRegion[im]);

    if ( m_MaskArray.size() > 0)
      f.AddMask( m_MaskArray[im] );

    for(unsigned int i=0; i<m_Transform[im].size(); i++)
      f.SetTransform(im,i,m_Transform[im][i]);
  }
  f.SetReferenceImage(this -> GetReferenceImage());
  f.SetLambda( m_Lambda );
  f.SetGamma( m_Gamma );
  f.SetSigma( m_Sigma );
  f.SetTau( m_Tau );
  f.SetTheta( m_Theta );
  f.SetDeltat( m_Deltat );
  f.SetPSF( m_PSF );
  f.SetSliceGap( m_SliceGap );
  f.SetUseDebluringPSF( m_UseDebluringPSF );
  
  std::cout << "Set xold and xest";
  //Variable Initialization
  if(m_CurrentOuterIteration == 0)
  {
    std::cout << " -- initialization : " << std::endl;
    
    f.SetXold(m_x);
    f.SetXest(m_x); 

    m_Px.set_size(ncols);
    m_Px.fill(0.0);

    m_Py.set_size(ncols);
    m_Py.fill(0.0);

    m_Pz.set_size(ncols);
    m_Pz.fill(0.0);

    f.SetComputeH(true);

    
  }
  else //Variable update
  {
    std::cout << " -- update : " << std::endl;

    // Converts and writes output image
    typename DuplicatorType::Pointer duplicatorb = DuplicatorType::New();
    duplicatorb->SetInputImage(this->GetReferenceImage());
    duplicatorb->Update();
    OutputImagePointer outputImb = duplicatorb->GetOutput();

    OutputImageRegionType outputImageRegionb = outputImb -> GetLargestPossibleRegion();

    //std::cout << outputImageRegionb << std::endl;

    itk::ImageRegionIterator<OutputImageType> outputItb( outputImb,outputImageRegionb );
    unsigned int linearIndexb = 0;
    for (outputItb.GoToBegin(); !outputItb.IsAtEnd(); ++outputItb, linearIndexb++)
      outputItb.Set(m_xest[linearIndexb]); 

    /*
    typename WriterType::Pointer writerb =  WriterType::New();
    writerb -> SetFileName( "/Users/sebastientourbier/Desktop/Patient01/SR/Manual/m_xest_in_set.nii.gz" );
    writerb -> SetInput( outputImb );
    writerb -> Update();
    */

    std::cout << "m_xold.size : " << m_xold.size()<<std::endl;
    std::cout << "m_xest.size : " << m_xest.size()<<std::endl;

    f.SetXold(m_xold);
    f.SetXest(m_xest);
    f.SetXsamp(m_xsamp);

    f.SetComputeH(false);
    // Sets generation matrix H (No need to recompute H for succesive outer loop of optimization)
    f.SetHMatrix(m_H);
    f.SetHtHMatrix(m_HtH);
   // f.SetAMatrix(m_A);
    //f.SetZVector(m_Z);
  }

  f.SetP(m_Px,m_Py,m_Pz);

  //std::cout << "Before init()" << std::endl; 

  //Precompute matrices and vectors used in the cost function of the optimization algorithm (H,DivP,A,b)
  double start_time = mialsrtk::getTime();
  m_scale = m_x.max_value();
  f.SetScale(m_scale);
  f.SetX(m_x);

  f.Initialize();

  double end_time = mialsrtk::getTime();
  m_InitTime = m_InitTime + (end_time -start_time);
  //m_A = f.GetAMatrix();

  if(m_CurrentBregmanLoop != 0)
  {
    std::cout << "m_Z set in SR resampler filter." << std::endl;
    f.SetZVector(m_Z);
  }

  //Gets matrix H and the vector of observations Y
  m_H = f.GetHMatrix();
  m_HtH = f.GetHtHMatrix();
  m_Y = f.GetObservationsY();

  m_xsamp = f.GetXsamp();

  unsigned int iteration = 0;
  double normXinner = 1;  

  VnlVectorType m_xoldinner;
  m_xoldinner.set_size(ncols);

  m_xold = m_x;

  f.SetXsamp(m_xsamp);

  double energy = 0.0;
  double energy_init = f.energy_value();

  start_time = mialsrtk::getTime();
  //while ( normXinner > m_ConvergenceThreshold &&  iteration < m_Iterations )
  for ( unsigned int iteration = 0 ; iteration < m_Iterations ; iteration++ )  
  {
    //Stores the old solution
    m_xoldinner = m_x;

    //Updates the new solution
    f.update();
    m_x = f.GetX();

    //Sets x as the new solution and evaluates the energy associated
    f.SetX(m_x);
    f.SetXsamp(m_xsamp);


    //std::cout << std::endl << "old x inner (pointer):" << m_xoldinner.sum() << " (" << &m_xoldinner << ")" << std::endl;
    //std::cout << "new x inner (pointer):" << m_x.sum() << " (" << &m_x << ")" << std::endl;
    
    //Computes the convergence criterion
    normXinner = (m_x - m_xoldinner).two_norm() / m_xoldinner.two_norm();
    std::cout<<"Inner loop criterion (iter = "<< iteration <<") : "<<normXinner<<std::endl;

    //double mse = (((m_x-m_xoldinner).squared_magnitude())) / m_x.size();
    //std::cout<<"Inner loop MSE : "<< mse <<std::endl;

    if( normXinner < m_ConvergenceThreshold )
    {
      std::cout<<"Inner loop has converged  ( Final value = " << normXinner << " )" << std::endl;
      break;
    }
  }

  std::cout << "sampling sum" << m_xsamp.sum() << std::endl;

  m_TVEnergy=f.energy_value();
  end_time = mialsrtk::getTime();

  m_InnerOptTime = m_InnerOptTime + (end_time - start_time);

  // Setup optimizer
  //std::cout << "Setup optimizer" << std::endl;
  /*
  vnl_conjugate_gradient cg(f);
  cg.set_f_tolerance( 1e-30 );
  cg.set_g_tolerance( 1e-30 );
  cg.set_x_tolerance( 1e-30 ); 
  //cg.set_epsilon_function( Epsilon_Function );
  cg.set_max_function_evals(m_Iterations);
  cg.set_verbose(true);

  // Start minimization

  m_xold = 1.0 * m_x;	
  std::cout << "Optimization : " << std::endl;
  cg.minimize(m_x);

  //std::cout << "Optimization done" << std::endl;

  cg.diagnose_outcome();
  */

  m_Px = f.GetPx();
  m_Py = f.GetPy();
  m_Pz = f.GetPz();

  m_Theta = f.GetTheta();

  m_Z = f.GetZVector();
  
  VnlVectorType m_xdiff = m_x - m_xold;

  m_xest = m_x + (float)m_Theta * m_xdiff;//+ f.GetTheta() * (m_x - m_xold);	
  std::cout << std::endl << "Estimate x at future outer iteration : sum(m_xest) = " << m_xest.sum() << std::endl << std::endl;
  
  // Converts and writes output image
  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage(this->GetReferenceImage());
  duplicator->Update();
  OutputImagePointer outputIm = duplicator->GetOutput();

  OutputImageRegionType outputImageRegion = outputIm -> GetLargestPossibleRegion();

  itk::ImageRegionIterator<OutputImageType> outputIt( outputIm,outputImageRegion );

  /*
  linearIndex = 0;
  for (outputIt.GoToBegin(); !outputIt.IsAtEnd(); ++outputIt, linearIndex++)
   outputIt.Set(m_xest[linearIndex]); 

  std::cout << "End optimize" << std::endl;
*/
  /*
  typename WriterType::Pointer writer2 =  WriterType::New();
  writer2 -> SetFileName( "/Users/sebastientourbier/Desktop/Patient01/SR/Manual/m_xest.nii.gz" );
  writer2 -> SetInput( outputIm );
  writer2 -> Update();
*/

  m_xsamp = f.GetXsamp();

  linearIndex = 0;
  for (outputIt.GoToBegin(); !outputIt.IsAtEnd(); ++outputIt, linearIndex++)
   outputIt.Set(m_xsamp[linearIndex]);

  /*
  typename WriterType::Pointer writer3 =  WriterType::New();
  writer3 -> SetFileName( "/home/tourbier/Desktop/x_sampling.nii.gz" );
  writer3 -> SetInput( outputIm );
  writer3 -> Update();
  */
  /*
  linearIndex = 0;
  for (outputIt.GoToBegin(); !outputIt.IsAtEnd(); ++outputIt, linearIndex++)
   outputIt.Set(m_xold[linearIndex]); 

  typename WriterType::Pointer writer4 =  WriterType::New();
  writer4 -> SetFileName( "/Users/sebastientourbier/Desktop/Patient01/SR/Manual/m_xold.nii.gz" );
  writer4 -> SetInput( outputIm );
  writer4 -> Update();
  */
}

/**
 * Update the Bergman variable
 */
template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
void
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::UpdateZ()
{
  std::cout << "Update Z : " << std::endl;

  vnl_vector<float> Hx;
  Hx.set_size(m_Z.size());
  m_H.pre_mult(m_x,Hx);

  std::cout << "Old value : " << m_Z.sum() << "(size:" << m_Z.size() << ")" << std::endl;
  std::cout << "Hx : " << Hx.sum() << "(size:" << Hx.size() << ")" << std::endl;
  std::cout << "Y : " << m_Y.sum() << "(size:" << m_Y.size() << ")" << std::endl;
  m_Z = m_Z + m_Y - Hx;

  std::cout << "New value : " << m_Z.sum() << std::endl;

}


template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
void
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::SetOutputOrigin(
  const double* origin)
{
  PointType p(origin);
  this->SetOutputOrigin( p );
}

/**
 * Execute TV-based optimization and link the output of the filter to the image reconstructed
 */
template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
void
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::GenerateData()
{
  Optimize();

  // Get the output pointers
  OutputImagePointer outputPtr = this->GetOutput();

  // Allocate data
  IndexType outputStart;
  outputStart[0] = 0; outputStart[1] = 0; outputStart[2] = 0;

  const OutputImageType * referenceImage = this->GetReferenceImage();

  SizeType outputSize = referenceImage -> GetLargestPossibleRegion().GetSize();

  OutputImageRegionType outputRegion;
  outputRegion.SetIndex(outputStart);
  outputRegion.SetSize(outputSize);

  outputPtr -> SetRegions(outputRegion);
  outputPtr -> Allocate();
  outputPtr -> FillBuffer(m_DefaultPixelValue);

  outputPtr -> SetOrigin( referenceImage -> GetOrigin() );
  outputPtr -> SetSpacing( referenceImage -> GetSpacing() );
  outputPtr -> SetDirection( referenceImage -> GetDirection() );

  IndexType hrIndex;
  IndexType hrStart = m_OutputImageRegion.GetIndex();
  SizeType  hrSize  = m_OutputImageRegion.GetSize();

  //std::cout <<"m_OutputImageRegion : "<< m_OutputImageRegion << std::endl;


  //ENH: If we iterate over output image and we check the value of the current index
  // in m_x(doing the inverse conversion), it may be faster.
  for (unsigned int i = 0; i<m_x.size(); i++)
  {
    IndexType hrDiffIndex;
    hrDiffIndex[2] = i / (hrSize[0]*hrSize[1]);

    hrDiffIndex[1] = i - hrDiffIndex[2]*hrSize[0]*hrSize[1];
    hrDiffIndex[1] = hrDiffIndex[1] / hrSize[0];

    hrDiffIndex[0] = i - hrDiffIndex[2]*hrSize[0]*hrSize[1] - hrDiffIndex[1]*hrSize[0];


    hrIndex[0] = hrDiffIndex[0] + hrStart[0];
    hrIndex[1] = hrDiffIndex[1] + hrStart[1];
    hrIndex[2] = hrDiffIndex[2] + hrStart[2];

    //*m_scale
    outputPtr -> SetPixel(hrIndex, m_x[i] );

  }

  //m_x.clear();
}

/**Check if the reference image used for quality evaluation was properly loaded.
 *
 * It happened that our reference image did not have the same dimension convention.
 * However, all processing tasks are performed on vectorized images.
 * Therefore, depdending on the dimension convention, the vectorialization should be adapted in order to have physical consistency in the comparison.
 */
template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
void
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::CheckGT(const vnl_vector<float>& x)
{  
  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage(this->GetReferenceImage());
  duplicator->Update();
  OutputImagePointer outputIm = duplicator->GetOutput();

  // Allocate data
  IndexType outputStart;
  outputStart[0] = 0; outputStart[1] = 0; outputStart[2] = 0;

  const OutputImageType * referenceImage = this->GetReferenceImage();

  SizeType outputSize = referenceImage -> GetLargestPossibleRegion().GetSize();

  OutputImageRegionType outputRegion;
  outputRegion.SetIndex(outputStart);
  outputRegion.SetSize(outputSize);

  outputIm -> SetRegions(outputRegion);
  outputIm -> Allocate();
  outputIm -> FillBuffer(m_DefaultPixelValue);

  vnl_matrix_fixed<double,3,3> vnlDirection  = referenceImage -> GetDirection().GetVnlMatrix();
  vnl_matrix_fixed<double,3,3> vnlDirectionMod = referenceImage -> GetDirection().GetVnlMatrix();

  //vnlDirectionMod.set_column(1,vnlDirection.get_column(2));
  //vnlDirectionMod.set_column(2,vnlDirection.get_column(1));

  typename TInputImage::DirectionType newDirection;
  newDirection = vnlDirectionMod;

  outputIm -> SetOrigin( referenceImage -> GetOrigin() );
  outputIm -> SetSpacing( referenceImage -> GetSpacing() );
  outputIm -> SetDirection( newDirection );

  IndexType hrIndex;
  IndexType hrStart = m_OutputImageRegion.GetIndex();
  SizeType  hrSize  = m_OutputImageRegion.GetSize();

  //std::cout <<"m_OutputImageRegion : "<< m_OutputImageRegion << std::endl;


  //ENH: If we iterate over output image and we check the value of the current index
  // in m_x(doing the inverse conversion), it may be faster.
  for (unsigned int i = 0; i<x.size(); i++)
  {
    IndexType hrDiffIndex;
    /*
    hrDiffIndex[2] = i / (hrSize[0]*hrSize[1]);

    hrDiffIndex[1] = i - hrDiffIndex[2]*hrSize[0]*hrSize[1];
    hrDiffIndex[1] = hrDiffIndex[1] / hrSize[0];

    hrDiffIndex[0] = i - hrDiffIndex[2]*hrSize[0]*hrSize[1] - hrDiffIndex[1]*hrSize[0];
    */
    hrDiffIndex[2] = i / (hrSize[0]*hrSize[1]);

    hrDiffIndex[1] = i - hrDiffIndex[2]*hrSize[0]*hrSize[1];
    hrDiffIndex[1] = hrDiffIndex[1] / hrSize[0];

    hrDiffIndex[0] = i - hrDiffIndex[2]*hrSize[0]*hrSize[1] - hrDiffIndex[1]*hrSize[0];

    hrIndex[0] = hrDiffIndex[0] + hrStart[0];
    hrIndex[1] = hrDiffIndex[1] + hrStart[1];
    hrIndex[2] = hrDiffIndex[2] + hrStart[2];

    outputIm -> SetPixel(hrIndex, x[i] );

  }

  typename WriterType::Pointer writer2 =  WriterType::New();
  //writer2 -> SetFileName( "/home/tourbier/Documents/Dropbox/Data/NewBornDataValidation/ParameterSettings/SR/gt_in_algo.nii.gz" );
  writer2 -> SetFileName( "/Users/sebastientourbier/Desktop/gt_in_algo.nii.gz" );
  writer2 -> SetInput( outputIm );
  writer2 -> Update();

  std::cout << "GT saved" << std::endl;

  //m_x.clear();
}

// TODO: We are not requiring any image region since we are using several inputs.
// We should check if this creates some problems at level of pipeline execution.
// We should also consider the implementation the class derivating from ProcessObject,
// perhaps it's a more logical choice (it does not assume a single input image)
// Same changes should be applied to ResampleImageByInjection and ResampleLabelByInjection
// classes.

/**
 * Inform pipeline of necessary input image region
 *
 * Determining the actual input region is non-trivial, especially
 * when we cannot assume anything about the transform being used.
 * So we do the easy thing and request the entire input image.
 */
template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
void
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::GenerateInputRequestedRegion()
{
  // call the superclass's implementation of this method

  //std::cout << "GenerateInputRequestedRegion : " << std::endl;
  Superclass::GenerateInputRequestedRegion();

  if ( !this->GetInput() )
    {
    return;
    }

  // get pointers to the input and output
  InputImagePointer  inputPtr  =
    const_cast< TInputImage *>( this->GetInput() );

  // Request the entire input image
  InputImageRegionType inputRegion;
  inputRegion = inputPtr->GetLargestPossibleRegion();
  inputPtr->SetRequestedRegion(inputRegion);
  /*
  std::cout << "Filter input pointer : " << inputPtr << std::endl; 
  std::cout << "Filter input region : " << std::endl << inputRegion << std::endl << std::endl;
  */
}

template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
void
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::SetZVector(vnl_vector<float>& v)
{
  m_Z = v;
}

template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
vnl_vector<float>
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::GetObservationsY()
{
  return m_Y;
}

template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
vnl_sparse_matrix<float>
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::GetAcquisitionMatrixH()
{
  return m_H;
}

template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
vnl_sparse_matrix<float>
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::GetMatrixHtH()
{
  return m_HtH;
}

template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
vnl_vector<float>
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::GetSolutionX()
{
  return m_x;
}

template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
vnl_vector<float>
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::GetZVector()
{
  return m_Z;
}

/**
 * Get the smart pointer to the reference image that will provide
 * the grid parameters for the output image.
 */
template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
const typename SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>::OutputImageType *
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::GetReferenceImage() const
{
  Self * surrogate = const_cast< Self * >( this );
  const OutputImageType * referenceImage =
    static_cast<const OutputImageType *>(this->ProcessObject::GetInput(0));
  /*
  std::cout << "Debug in getReferenceImage() :  " << std::endl;
  std::cout << "Image pointer : " << referenceImage << std::endl;
  std::cout << "Region : " << referenceImage -> GetLargestPossibleRegion() << std::endl;
  */
  return referenceImage;
}

/**
 * Set the smart pointer to the reference image that will provide
 * the grid parameters for the output image.
 */
template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
void
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::SetReferenceImage( const TOutputImage *image )
{
  itkDebugMacro("setting input ReferenceImage to " << image);
  /*
  std::cout << "Debug in setReferenceImage() : " << std::endl;
  std::cout << "Input image : " << std::endl;
  std::cout << "Pointer :  " << image << std::endl;
  std::cout << "Region" << image -> GetLargestPossibleRegion() << std::endl;
  */
  if( image != static_cast<const TOutputImage *>(this->ProcessObject::GetInput( 0 )) )
    {
    this->ProcessObject::SetNthInput(0, const_cast< TOutputImage *>( image ) );
    this->Modified();
    }

  /*  
  const OutputImageType * referenceImage =
    static_cast<const OutputImageType *>(this->ProcessObject::GetInput(0));
  std::cout << "Test access to ref image : " << std::endl;
  std::cout << "Pointer :  " << referenceImage << std::endl;
  std::cout << "Region" << referenceImage -> GetLargestPossibleRegion() << std::endl;  
  */
}

/** Helper method to set the output parameters based on this image */
template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
void
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::SetOutputParametersFromImage ( const ImageBaseType * image )
{
  //std::cout << "SetOutputParametersFromImage" << std::endl;
  this->SetOutputOrigin ( image->GetOrigin() );
  this->SetOutputSpacing ( image->GetSpacing() );
  this->SetOutputDirection ( image->GetDirection() );
  this->SetOutputStartIndex ( image->GetLargestPossibleRegion().GetIndex() );
  this->SetSize ( image->GetLargestPossibleRegion().GetSize() );
}

/**
 * Inform pipeline of required output region
 */
template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
void
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::GenerateOutputInformation()
{  
  //std::cout << "In GenerateOutputInformation : " << std::endl;
  // call the superclass' implementation of this method
  Superclass::GenerateOutputInformation();

  // get pointers to the input and output
  OutputImagePointer outputPtr = this->GetOutput();
  if ( !outputPtr )
    {
    return;
    }

  const OutputImageType * referenceImage = this->GetReferenceImage();

  /*  
  std::cout << "Filter output pointer : " << std::endl;
  std::cout << outputPtr << std::endl;  
  std::cout << "m_UseReferenceImage : " << m_UseReferenceImage << std::endl;
  std::cout << "referenceImage : " << referenceImage << std::endl << std::endl;
  */

  // Set the size of the output region
  if( m_UseReferenceImage && referenceImage )
    {
    //std::cout << "referenceImage->GetLargestPossibleRegion()" << referenceImage->GetLargestPossibleRegion() << std::endl;
    outputPtr->SetLargestPossibleRegion( referenceImage->GetLargestPossibleRegion() );
    }
  else
    {
    typename TOutputImage::RegionType outputLargestPossibleRegion;
    outputLargestPossibleRegion.SetSize( m_Size );
    outputLargestPossibleRegion.SetIndex( m_OutputStartIndex );
    outputPtr->SetLargestPossibleRegion( outputLargestPossibleRegion );
    }

  // Set spacing and origin
  if (m_UseReferenceImage && referenceImage)
    {
    outputPtr->SetOrigin( referenceImage->GetOrigin() );
    outputPtr->SetSpacing( referenceImage->GetSpacing() );
    outputPtr->SetDirection( referenceImage->GetDirection() );
    }
  else
    {
    outputPtr->SetOrigin( m_OutputOrigin );
    outputPtr->SetSpacing( m_OutputSpacing );
    outputPtr->SetDirection( m_OutputDirection );
    }

}

/**
 * Verify if any of the components has been modified.
 */
template <class TInputImage, class TOutputImage  , class TInterpolatorPrecisionType>
unsigned long
SuperResolutionRigidImageFilterWithImplicitGradientDescent<TInputImage,TOutputImage   ,TInterpolatorPrecisionType>
::GetMTime( void ) const
{
  unsigned long latestTime = Object::GetMTime();

  if( m_Transform.size()!=0 )
    {
      for(unsigned int i=0; i<m_Transform.size(); i++)
      {
        for(unsigned int j=0; j<m_Transform[i].size(); j++)
        {
          if( latestTime < m_Transform[i][j]->GetMTime() )
          {
            latestTime = m_Transform[i][j]->GetMTime();
          }
        }
      }
    }

  return latestTime;
}

} // end namespace mialsrtk

#endif
