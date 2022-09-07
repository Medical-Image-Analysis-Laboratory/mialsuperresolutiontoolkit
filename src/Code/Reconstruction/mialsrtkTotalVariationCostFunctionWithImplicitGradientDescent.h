/*==========================================================================

  © Université de Lausanne (UNIL) & Centre Hospitalier Universitaire de Lausanne (CHUV) - Centre d'Imagerie BioMédicale

  Date: 28/04/14
  Author(s): Sebastien Tourbier (sebastien.tourbier@unil.ch)

  ==========================================================================*/

#ifndef mialsrtkTOTALVARIATIONCOSTFUNCTIONWITHIMPLICITGRADIENTDESCENT_H_
#define mialsrtkTOTALVARIATIONCOSTFUNCTIONWITHIMPLICITGRADIENTDESCENT_H_

#include "vnl/vnl_cost_function.h"
#include "vnl/vnl_matops.h"
#include "vnl/vnl_trace.h"
#include "btkLinearInterpolateImageFunctionWithWeights.h"
#include "mialsrtkOrientedSpatialFunction.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkTransform.h"
#include "itkBSplineInterpolationWeightFunction.h"

using namespace btk;
using namespace itk;

namespace mialsrtk
{
/** \class TotalVariationCostFunctionWithImplicitGradientDescent
 * \brief TV-based error function class
 *
 * TotalVariationCostFunctionWithImplicitGradientDescent implements the TV-based cost function and its gradient
 * H, Y, and HtY are stored in this class to save memory.
 *
 * \ingroup Reconstruction
 */
template <class TImage>
class TotalVariationCostFunctionWithImplicitGradientDescent
{
  public:

  typedef TImage   ImageType;
  typedef typename ImageType::Pointer ImagePointer;
  typedef typename ImageType::ConstPointer ImageConstPointer;

  typedef typename ImageType::RegionType  RegionType;
  typedef typename ImageType::SizeType    SizeType;
  typedef typename ImageType::IndexType   IndexType;
  typedef typename ImageType::SpacingType SpacingType;
  typedef typename ImageType::PointType   PointType;

  typedef ImageMaskSpatialObject< TImage::ImageDimension > MaskType;
  typedef typename MaskType::Pointer   MaskPointer;


  //typedef itk::Transform<double>    TransformType;
  //typedef Euler3DTransform<double> TransformType;
  typedef itk::VersorRigid3DTransform<double> TransformType;
  typedef typename TransformType::Pointer TransformPointerType;

  typedef LinearInterpolateImageFunctionWithWeights<ImageType, double> InterpolatorType;
  typedef typename InterpolatorType::Pointer InterpolatorPointer;

  typedef ContinuousIndex<double, TImage::ImageDimension> ContinuousIndexType;

  /**Oriented spatial function typedef. */
  typedef OrientedSpatialFunction<double, 3, PointType> FunctionType;

  /**Const iterator typedef. */
  typedef ImageRegionConstIteratorWithIndex< ImageType >  ConstIteratorType;
    typedef ImageRegionConstIteratorWithIndex< MaskType >  ConstMaskIteratorType;

  typedef vnl_vector<float> VnlVectorType;
  typedef vnl_sparse_matrix<float> VnlSparseMatrixType;

  struct img_size {
     unsigned int width;
     unsigned int height;
     unsigned int depth;
  } x_size;

  TotalVariationCostFunctionWithImplicitGradientDescent(unsigned int dim);

  /**Update the solution**/
  void update();

  /**Return the value of the TV-based Cost function. */
  double energy_value();

  /**Gradient of the cost function, required for optimization with
    vnl_conjugate_gradient . */
  void gradf(const vnl_vector<double>& x, vnl_vector<double>& g);

  /**Computes DivP**/
  void computeDivP();

  /**Construction of vector Y and matrix H of the observation model Y=H*X. */
  void Initialize();

  /**Construction of vector Y and matrix H of the observation model Y=H*X. */
  void Initialize2();

  /**Construction of vector Y and matrix H of the observation model Y=H*X. */
  void ComputeSRHMatrix();

  /**Computes terms related to Total variation regularization. */
  void ComputeTotalVariationTerms();

  void SetX(const vnl_vector<float>& x);

  vnl_vector<float> GetX();

  /**Initialization of image x estimated at iteration n+1. */
  void SetXest(const vnl_vector<float>& x);

  /**Initialization of image x corresponding to the image solution at iteration n-1. */
  void SetXold(const vnl_vector<float>& x);

  /**Sets the weight of the regularization term.*/
  void SetLambda(float value);

  /**Sets the weight gamma.*/
  void SetGamma(float value);

  /**Sets the weight sigma.*/
  void SetSigma(float value);

  /**Sets the parameter deltat.*/
  void SetDeltat(float value);

  /**Sets the weight tau.*/
  void SetTau(float value);
 
  /**Sets the weight theta (optimizer time step).*/
  void SetTheta(float value);

  /**Sets the gap between slices.*/
  void SetSliceGap(double gap);

  /**Gets the gap between slices.*/
  double GetSliceGap();

  /** Sets the boolean value of m_ComputeH.*/
  void SetComputeH(bool value);

  /** Sets the boolean value of m_UseDebluringPSF.*/
  void SetUseDebluringPSF(bool value);

  /** Gets the boolean value of m_ComputeH.*/
  //bool GetComputeH(ComputeH, bool);

  /**Adds a low-resolution image.*/
  void AddImage( ImageType* image );

  /**Adds the image region of the corresponding low-resolution image.*/
  void AddRegion( RegionType region);

  /**Adds the image mask of the corresponding low-resolution image.*/
  void AddMask( MaskType *mask);

  /**Sets the reference image, i.e. an image providing the spatial positions
  where to compute the intensity values (each X value). This image is expected
  to have isotropic voxels, with a size equal to the in-plane voxel size of the
  low resolution images. However, any image can be provided to this end, the
  code has been written generically. */
  void SetReferenceImage( const ImageType * image );

  // TODO This function must be modified after modifying this class to use slice
  // by slice transforms.
  /**Sets the transforms obtained with the reconstruction method. These
  transformations correct the movements of the fetus during image acquisition.*/
  void SetTransform( int i, int j, TransformType* transform );

  /** Sets the type of PSF (Boxcar, Gaussian). */
  void SetPSF(unsigned int psf)
  {
    m_PSF = psf;
  }

  /** Gets the type of PSF (Boxcar, Gaussian). */
  itkGetMacro(PSF, unsigned int);

  /** Gets the value of theta (optimizor time step). */
  float GetTheta();

  vnl_sparse_matrix<float> GetHMatrix();

  /** Sets Px Py and Pz**/
  void SetP(const vnl_vector<float>& px,const vnl_vector<float>& py,const vnl_vector<float>& pz);
  vnl_vector<float> GetPx();
  vnl_vector<float> GetPy();
  vnl_vector<float> GetPz();

  /** Sets H (Used in H is not computed during initialization)**/
  void SetHMatrix(const vnl_sparse_matrix<float>& M);

  /** Sets HtH **/
  void SetHtHMatrix(const vnl_sparse_matrix<float>& M);

  /** Gets HtH **/
  vnl_sparse_matrix<float> GetHtHMatrix();

  /** Gets Matrix A **/
  vnl_sparse_matrix<float> GetAMatrix();

  /** Sets Matrix A **/
  void SetAMatrix(const vnl_sparse_matrix<float>& M);

  //Sets/Gets the sampling vector Xsamp
  void SetXsamp(vnl_vector<float>&v)
  {
      Xsamp=v;
  };
 vnl_vector<float> GetXsamp()
 {
     return Xsamp;
 };

 float GetScale()
 {
     return scale;
 };
void SetScale(float&v)
 {
     scale=v;
 };

    //Sets the vector Z
  void SetZVector(vnl_vector<float>& v);

  //Gets the vector Z
  vnl_vector<float> GetZVector();

  //Gets the vector of observations Y
  vnl_vector<float> GetObservationsY();

void SetVerbose( bool verbose )
{
    m_verbose = verbose;
}

bool GetVerbose()
{
    return this -> m_verbose;
}

  private:

  void set(float * array, int size, float value);

  /** Mirror of the position pos. abs(pos) must not be > 2*(size-1) */
  int mirror(int pos, int size);

  /** Gets an image row as a vector. */
  void get_row(const vnl_vector<float>& image, img_size & size, int row,
      int frame, float * output);

  /** Sets an image row from a vector. */
  void set_row(vnl_vector<float>& image, img_size & size, int row, int frame,
      float * input);

  /** Gets an image column as a vector. */
  void get_col(const vnl_vector<float>& image, img_size & size, int col,
      int frame, float * output);

  /** Sets an image column from a vector. */
  void set_col(vnl_vector<float>& image, img_size & size, int col, int frame,
      float * input);

  /** Gets an image z-axis as a vector. */
  void get_spec(const vnl_vector<float>& image, img_size & size, int row,
      int col, float * output);

  /** Sets an image z-axis from a vector. */
  void set_spec(vnl_vector<float>& image, img_size & size, int row, int col,
      float * input);

  void convol1d(float * kernel, int ksize, float * src, int src_size, float * dest);

  // 3D convolution : over the rows
  void convol3dx(const vnl_vector<float>& image, vnl_vector<float>& image_conv,
      img_size& size, float * kernel, int ksize);

  // 3D convolution : over the columns
  void convol3dy(const vnl_vector<float>& image, vnl_vector<float>& image_conv,
      img_size & size, float * kernel, int ksize);

  // 3D convolution : over the spectra
  void convol3dz(const vnl_vector<float>& image, vnl_vector<float>& image_conv,
      img_size & size, float * kernel, int ksize);

  vnl_sparse_matrix<float> H;
  vnl_sparse_matrix<float> HtH;
  //vnl_vector<float> HtY;
  vnl_vector<float> Y;
  vnl_vector<float> Xold;
  vnl_vector<float> Xest;  
  vnl_vector<float> X;
  vnl_vector<float> Xsamp;
  //vnl_vector<float> m_xold;

  vnl_vector<float> Z;

  vnl_vector<float>  Px;
  vnl_vector<float>  Py;
  vnl_vector<float>  Pz;
  vnl_vector<float>  DivP;
  
  vnl_sparse_matrix<float>  A;
  vnl_vector<float>  b;

  float theta;
  float tau;
  float gamma;
  float sigma;
  float normD;

  float scale;

  float deltat;

  float lambda;

  std::vector<ImagePointer>  m_Images;
  std::vector<RegionType>    m_Regions;
  std::vector<MaskPointer>   m_Masks;
  ImageConstPointer					 m_ReferenceImage;
 // std::vector< std::vector<TransformPointerType> > m_Transforms;
  std::vector< std::vector<TransformPointerType> > m_Transforms;
  RegionType m_OutputImageRegion;
  unsigned int m_PSF;

  bool m_ComputeH;

  double m_SliceGap;

  bool m_UseDebluringPSF;
  bool m_verbose;
};

} // namespace mialsrtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "mialsrtkTotalVariationCostFunctionWithImplicitGradientDescent.txx"
#endif

#endif /* mialsrtkTOTALVARIATIONCOSTFUNCTIONWITHIMPLICITGRADIENTDESCENT_H_ */
