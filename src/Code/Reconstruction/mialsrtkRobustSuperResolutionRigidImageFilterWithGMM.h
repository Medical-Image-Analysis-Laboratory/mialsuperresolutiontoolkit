/*==========================================================================

  © Université de Lausanne (UNIL) & Centre Hospitalier Universitaire de Lausanne (CHUV) - Centre d'Imagerie BioMédicale
    Harvard Medical School & Boston Children's Hospital - Computational Radiology Laboratory

  Date: 08/04/2015
  Author(s): Sebastien Tourbier (sebastien.tourbier@unil.ch)

  As a counterpart to the access to the source code, users are provided only
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

==========================================================================*/

#ifndef __mialsrtkRobustSuperResolutionRigidImageFilterWithGMM_h
#define __mialsrtkRobustSuperResolutionRigidImageFilterWithGMM_h

#include "itkFixedArray.h"
#include "itkTransform.h"
#include "itkEuler3DTransform.h"
#include "itkVersorRigid3DTransform.h"
#include "itkImageToImageFilter.h"
#include "itkSize.h"
#include "btkUserMacro.h"
#include "vnl/vnl_sparse_matrix.h"
#include "vnl/algo/vnl_conjugate_gradient.h"
#include "mialsrtkJointRobustTVGMMCostFunctionWithImplicitGradientDescent.h"

#include "itkVectorImageToImageAdaptor.h"

#include "itkImageDuplicator.h"
#include "itkImageFileWriter.h"  

#include "mialsrtkTimeHelper.h"

namespace mialsrtk
{

/** @class RobustSuperResolutionRigidImageFilterWithGMM
 * @brief Super-resolution reconstruction based on Total Variation using a set of low resolution images and the
 * reconstructed image. Use a semi-implicit gradient descent scheme to solve the inner least square problem.
 *
 * RobustSuperResolutionRigidImageFilterWithGMM allows to obtain a super-resolution image from a
 * set of low-resolution images and the reconstructed image. The class is templated
 * over the types of the input and output images.
 *
 * The implemented method is based on a fast and efficient TV-based optimization algorithm.
 * It uses an accelerated primal-dual hybrid gradient method based on [], [], [], where the solution of the
 * inner least-square problem is computed using a semi-implicit gradient descent scheme.
 *
 * @author Sebastien Tourbier
 * @ingroup Reconstruction
 */

template <class TInputImage, class TOutputImage, class TInterpolatorPrecisionType=double>
class RobustSuperResolutionRigidImageFilterWithGMM:
        public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:

    typedef enum {
        BOXCAR=0,
        GAUSSIAN=1
    } PSF_type;

    /** Standard class typedefs. */
    typedef RobustSuperResolutionRigidImageFilterWithGMM                Self;
    typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
    typedef itk::SmartPointer<Self>                            Pointer;
    typedef itk::SmartPointer<const Self>                      ConstPointer;

    typedef TInputImage                             InputImageType;
    typedef TOutputImage                            OutputImageType;
    typedef typename InputImageType::Pointer        InputImagePointer;
    typedef typename InputImageType::ConstPointer   InputImageConstPointer;
    typedef typename OutputImageType::Pointer       OutputImagePointer;

    typedef typename InputImageType::RegionType     InputImageRegionType;
    typedef std::vector<InputImageRegionType>       InputImageRegionVectorType;

    //Vector image type
    typedef itk::VectorImage<float,TInputImage::ImageDimension> VectorImageType;
    typedef typename VectorImageType::Pointer VectorImagePointer;

    //Vector image to scalar image adaptor
    typedef itk::VectorImageToImageAdaptor<float,TInputImage::ImageDimension> ImageAdaptorType;

    typedef typename  itk::ImageDuplicator< OutputImageType > DuplicatorType;
    typedef itk::ImageFileWriter< OutputImageType >   WriterType;

    typedef itk::ImageMaskSpatialObject< TInputImage::ImageDimension > MaskType;
    typedef typename MaskType::Pointer   MaskPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(SuperResolutionRigidImageFilter, itk::ImageToImageFilter);

    /** Number of dimensions. */
    itkStaticConstMacro(ImageDimension, unsigned int,
                        TInputImage::ImageDimension);

    /** Transform typedef. */
    //typedef TTransform    TransformType;
    typedef itk::VersorRigid3DTransform<TInterpolatorPrecisionType> TransformType;
    typedef typename TransformType::Pointer TransformPointerType;

    //TODO This should be replaced by a std::vector of btkSliceBySliceTransform.
    /** Type of the transform list. */
    typedef std::vector< std::vector<TransformPointerType> > TransformPointerArrayType;

    /** Image size typedef. */
    typedef itk::Size<itkGetStaticConstMacro(ImageDimension)> SizeType;

    /** Image index typedef support. */
    typedef typename TOutputImage::IndexType IndexType;

    /** Image point typedef support. */
    typedef typename TOutputImage::PointType    PointType;

    /** Image pixel typedef support. */
    typedef typename TOutputImage::PixelType   PixelType;

    /** Input image pixel typedef support. */
    typedef typename TInputImage::PixelType    InputPixelType;

    /** Typedef to describe the output image region type. */
    typedef typename TOutputImage::RegionType OutputImageRegionType;

    /** Image spacing typedef support. */
    typedef typename TOutputImage::SpacingType   SpacingType;

    /** Image direction typedef support. */
    typedef typename TOutputImage::DirectionType DirectionType;

    /** base type for images of the current ImageDimension */
    typedef itk::ImageBase<itkGetStaticConstMacro(ImageDimension)> ImageBaseType;

    /**Const iterator typedef. */
    typedef itk::ImageRegionConstIteratorWithIndex< OutputImageType >  OutputIteratorType;

    /**Const vector image iterator typedef. */
    typedef itk::ImageRegionConstIteratorWithIndex< VectorImageType >  VectorImageIteratorType;

    /**VnlVectorType typedef. */
    typedef vnl_vector<float> VnlVectorType;

    /** Overrides SetInput to resize the transform. */
    void AddInput(InputImageType* _arg);

    /** Set the transform array. */
    void SetTransform( int i, int j, TransformType* transform )
    {
        //std::cout << "Set transform of slice " << j << " of stack " << i << std::endl;

        //std::cout << "TEST: m_transform size: " << m_Transform.size() << std::endl;
        //std::cout << "m_transform[]: " << m_Transform[m_Transform.size()-1].size() << std::endl;

        //std::cout << "m_Transform size: " << m_Transform.size() << std::end;
        //std::cout << "m_Transform_size[i,j] : " <<  m_Transform[i][j]  << std::endl;
        //std::cout << "transform size: " << transform << std::endl;
        m_Transform[i][j] = transform;
        //std::cout << "end" << std::endl;
    }

    /** Returns the convergence criterion value**/
    double GetCriterionValue();

    void CheckGT(const vnl_vector<float>& x);

    /** Converts from a linear index (li = i+i*x_size+k*x_size*y_size) to an absolute
   * index (ai = [i j k]). */
    IndexType LinearToAbsoluteIndex( unsigned int linearIndex, InputImageRegionType region );

    /** Set the size of the output image. */
    itkSetMacro( Size, SizeType );

    /** Get the size of the output image. */
    itkGetConstReferenceMacro( Size, SizeType );

    /** Set the pixel value when a transformed pixel is outside of the
   * image.  The default default pixel value is 0. */
    itkSetMacro( DefaultPixelValue, PixelType );

    /** Get the pixel value when a transformed pixel is outside of the image */
    itkGetConstReferenceMacro( DefaultPixelValue, PixelType );

    /** Set the output image spacing. */
    itkSetMacro( OutputSpacing, SpacingType );

    /** Set the output image spacing as a const array of values. */
    virtual void SetOutputSpacing( const double* values );

    /** Get the output image spacing. */
    itkGetConstReferenceMacro( OutputSpacing, SpacingType );

    /** Set the output image origin. */
    itkSetMacro( OutputOrigin, PointType );

    /** Set the output image origin as a const array of values. */
    virtual void SetOutputOrigin( const double* values);

    /** Get the output image origin. */
    itkGetConstReferenceMacro( OutputOrigin, PointType );

    /** Set the output direciton cosine matrix. */
    itkSetMacro( OutputDirection, DirectionType );
    itkGetConstReferenceMacro( OutputDirection, DirectionType );

    /** Helper method to set the output parameters based on this image */
    void SetOutputParametersFromImage ( const ImageBaseType * image );

    /** Set the start index of the output largest possible region.
   * The default is an index of all zeros. */
    itkSetMacro( OutputStartIndex, IndexType );

    /** Get the start index of the output largest possible region. */
    itkGetConstReferenceMacro( OutputStartIndex, IndexType );

    /** Copy the output information from another Image.  By default,
   *  the information is specified with the SetOutputSpacing, Origin,
   *  and Direction methods. UseReferenceImage must be On and a
   *  Reference image must be present to override the default behavior.
   *  NOTE: This function seems redundant with the
   *  SetOutputParametersFromImage( image ) function */
    void SetReferenceImage ( const TOutputImage *image );

    /** Gets the reference image. */
    const TOutputImage * GetReferenceImage( void ) const;

    /** Sets the use of a reference image to true/false. */
    itkSetMacro( UseReferenceImage, bool );

    /** Adds UseReferenceImageOff/On. */
    itkBooleanMacro( UseReferenceImage );

    /** Gets the status of the UseReferenceImage variable. */
    itkGetConstMacro( UseReferenceImage, bool );

    /** SuperResolutionRigidImageFilter produces an image which is a different size
   * than its input.  As such, it needs to provide an implementation
   * for GenerateOutputInformation() in order to inform the pipeline
   * execution model.  The original documentation of this method is
   * below. \sa ProcessObject::GenerateOutputInformaton() */
    virtual void GenerateOutputInformation( void );

    /** SuperResolutionRigidImageFilter needs a different input requested region than
   * the output requested region.  As such, SuperResolutionRigidImageFilter needs
   * to provide an implementation for GenerateInputRequestedRegion()
   * in order to inform the pipeline execution model.
   * \sa ProcessObject::GenerateInputRequestedRegion() */
    virtual void GenerateInputRequestedRegion( void );

    /** Method Compute the Modified Time based on changed to the components. */
    unsigned long GetMTime( void ) const;

    /** Adds an image region. The regions must be added in the same order than the
   * input images.*/
    void AddRegion(InputImageRegionType _arg)
    {
        m_InputImageRegion.push_back(_arg);
    }

    /** Adds an image mask. Masks must be added in the same order than the
   * input images.*/
    void AddMask(MaskType *mask)
    {
        m_MaskArray.push_back( mask );
    }

    /** Adds a vector image of GMM weights.*/
    void AddGMMWeights(VectorImageType *weights)
    {
        m_GMMWeights.push_back(weights);
    }

    /** Adds a vector image of GMM means.*/
    void AddGMMMeans(VectorImageType *means)
    {
        m_GMMMeans.push_back(means);
    }
   
    /** Adds a vector image of GMM variances.*/
    void AddGMMVars(VectorImageType *vars)
    {
        m_GMMVars.push_back(vars);
    }

    /** Sets the output image region.*/
    itkSetMacro(OutputImageRegion, OutputImageRegionType);

    /** Gets the output image region.*/
    itkGetMacro(OutputImageRegion, OutputImageRegionType);

    /** Sets the number of iterations.*/
    itkSetMacro(Iterations, unsigned int);

    /** Gets the number of iterations.*/
    itkGetMacro(Iterations, unsigned int);

    /** Sets the lambda value for TV regularization.*/
    itkSetMacro(Lambda, float);

    /** Gets the lambda value for TV regularization.*/
    itkGetMacro(Lambda, float);

    /** Sets the beta value for GMM regularization.*/
    itkSetMacro(Beta, float);

    /** Gets the beta value for GMM regularization.*/
    itkGetMacro(Beta, float);

    /** Sets the optimizer sigma value.*/
    itkSetMacro(Sigma, float);

    /** Gets the optimizer sigma value.*/
    itkGetMacro(Sigma, float);

    /** Sets the optimizer gamma value.*/
    itkSetMacro(Gamma, float);

    /** Gets the optimizer gamma value.*/
    itkGetMacro(Gamma, float);

    /** Sets the optimizer tau value.*/
    itkSetMacro(Tau, float);

    /** Gets the optimizer tau value.*/
    itkGetMacro(Tau, float);

    /** Sets the optimizer theta value.*/
    itkSetMacro(Theta, float);

    /** Gets the optimizer theta value.*/
    itkGetMacro(Theta, float);

    /** Sets the type of PSF (Boxcar/Gaussian).*/
    itkSetMacro(PSF, unsigned int);

    /** Gets the type of PSF (Boxcar/Gaussian).*/
    itkGetMacro(PSF, unsigned int);

    /** Sets the current outer iteration index.*/
    itkSetMacro(CurrentOuterIteration, unsigned int);

    /** Gets the current outer iteration index.*/
    itkGetMacro(CurrentOuterIteration, unsigned int);

    /** Sets the inner loop convergence threshold.*/
    itkSetMacro(ConvergenceThreshold, double);

    /** Gets the inner loop convergence threshold.*/
    itkGetMacro(ConvergenceThreshold, double);

    /** Sets the parameter deltat.*/
    itkSetMacro(Deltat, float);

    /** Gets the parameter deltat.*/
    itkGetMacro(Deltat, float);

    /** Sets the parameter bregman loop.*/
    itkSetMacro(CurrentBregmanLoop, unsigned int);

    /** Gets the parameter bregman loop.*/
    itkGetMacro(CurrentBregmanLoop, unsigned int);

    /** Gets the run time for initializing the cost function and computing H.*/
    itkGetMacro(InitTime, double);

    /** Gets the run time for inner loop optimization.*/
    itkGetMacro(InnerOptTime, double);

    /** Sets the gap between slices (in mm).*/
    itkSetMacro(SliceGap, double);

    /** Gets the gap between slices (in mm).*/
    itkGetMacro(SliceGap, double);

    /** Gets the value of TV energy.*/
    itkGetMacro(TVEnergy, double);

    /** Sets the vector Z.*/
    void SetZVector(vnl_vector<float>& M);

    /** Set to use robust energy in SR reconstruction.*/
    itkSetMacro(UseRobustSR, bool);

    /** Set tuning parameter rho1 (voxel-wise weight) used in the robust energy. */
    itkSetMacro(Rho1, float);

    /** Set tuning parameter rho2 (slice-wise weight) used in the robust energy. */
    itkSetMacro(Rho2, float);

    /** Gets the vector Z.*/
    vnl_vector<float> GetZVector();

    /** Updates vector Z (Z = Z + Y - Hx)**/
    void UpdateZ();

    /** Gets the parameter deltat.*/
    //vnl_sparse_matrix<float> GetHMatrix();

    //Gets the vector of observations Y
    vnl_vector<float> GetObservationsY();

    /** Sets the Huber norm threshold.*/
    itkSetMacro(HuberCriterion, float);

    /** Gets the Huber norm threshold.*/
    itkGetMacro(HuberCriterion, float);

    void SetVerbose( bool verbose )
    {
        m_verbose = verbose;
    }

    bool GetVerbose ()
    {
        return this -> m_verbose;
    }


#ifdef ITK_USE_CONCEPT_CHECKING
    /** Begin concept checking */
    itkConceptMacro(OutputHasNumericTraitsCheck,
                    (Concept::HasNumericTraits<PixelType>));
    /** End concept checking */
#endif

protected:
    RobustSuperResolutionRigidImageFilterWithGMM( void );
    ~RobustSuperResolutionRigidImageFilterWithGMM( void ) {};

    void PrintSelf( std::ostream& os, itk::Indent indent ) const;

    /** mialsrtkSuperResolutionRigidImageFilterV2 cannot be implemented as a multithreaded filter.  Therefore,
   * this implementation only provides a GenerateData() routine that executes the optimization and allocates output
   * image data.
   * */
    void GenerateData();

    virtual void VerifyInputInformation() {};

private:

    RobustSuperResolutionRigidImageFilterWithGMM( const Self& ); //purposely not implemented
    void operator=( const Self& ); //purposely not implemented

    void Optimize();

    SizeType                    m_Size;         /**< Size of the output image. */
    TransformPointerArrayType   m_Transform; /** Array of slice by slice transforms */
    InputImageRegionVectorType  m_InputImageRegion;

    OutputImageRegionType       m_OutputImageRegion;

    std::vector<InputImagePointer>  m_ImageArray; /** Array of LR images */
    std::vector<MaskPointer>  			m_MaskArray;/** Array of LR masks */

    std::vector<VectorImagePointer> m_GMMWeights; /** Array of Vector images containing the GMM weights*/
    std::vector<VectorImagePointer> m_GMMMeans; /** Array of Vector images containing the GMM weights*/
    std::vector<VectorImagePointer> m_GMMVars; /** Array of Vector images containing the GMM weights*/

    VnlVectorType     m_x;/**<Solution X at the current outer iteration*/
    VnlVectorType     m_xold;/**< Solution X at the previous outer iteration */
    VnlVectorType     m_xest;/**< Solution of X at the current outer iteration, estimated at the previous outer iteration*/

    VnlVectorType     m_Z;/**< Bregman variable*/

    vnl_vector<float> m_Y;/**< Vector containing all observations (LR images)*/

    vnl_sparse_matrix<float> m_H;
    vnl_sparse_matrix<float> m_HtH;
    vnl_sparse_matrix<float> m_A;

    vnl_vector<float> m_Px;/**< x-axis component of P*/
    vnl_vector<float> m_Py;/**< y-axis component of P*/
    vnl_vector<float> m_Pz;/**< z-axis component of P*/

    float m_Beta; /**< GMM Regularization weight of the optimizer. */
    float m_Lambda; /**< TV Regularization weight of the optimizer. */
    float m_Sigma; /**< Weight sigma of the optimizer. */
    float m_Tau; /**< Weight tau of the optimizer. */
    float m_Theta;/**< Weight theta of the optimizer. */
    float m_Gamma;/**< Weight gamma of the optimizer. */
    float m_Deltat;/**< Weight deltat of the optimizer. */

    unsigned int      m_CurrentOuterIteration;

    unsigned int      m_CurrentBregmanLoop;

    unsigned int 	    m_Iterations;/** Number of iterations in the inner least-square optimization problem */

    double             m_ConvergenceThreshold;/** Convergence threshold of the inner least-square optimization problem */

    PixelType         m_DefaultPixelValue; /**< Default pixel value if the point
                                              falls outside the image. */
    SpacingType       m_OutputSpacing;     /**< Spacing of the output image. */
    PointType   			m_OutputOrigin;      /**< Origin of the output image. */
    DirectionType     m_OutputDirection;   /**< Direction of the output image. */
    IndexType         m_OutputStartIndex;  /**< Start index of the output image.*/
    bool              m_UseReferenceImage;

    unsigned int m_PSF;

    bool m_UseRobustSR;
    float m_Rho1, m_Rho2;
    float m_HuberCriterion;

    double m_InitTime;/**< Run time for initialization **/
    double m_InnerOptTime;/**< Run time for optimization **/

    double m_TVEnergy;/**< Run time for optimization **/

    double m_SliceGap;/** Gap between slices**/
    bool m_verbose;

};


} // end namespace mialsrtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "mialsrtkRobustSuperResolutionRigidImageFilterWithGMM.txx"
#endif

#endif
