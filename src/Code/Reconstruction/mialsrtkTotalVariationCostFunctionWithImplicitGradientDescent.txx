/*==========================================================================

  © Université de Lausanne (UNIL) & Centre Hospitalier Universitaire de Lausanne (CHUV) - Centre d'Imagerie BioMédicale

  Date: 28/04/14
  Author(s): Sebastien Tourbier (sebastien.tourbier@unil.ch)

  ==========================================================================*/

#ifndef __TotalVariationCostFunctionWithImplicitGradientDescent_txx
#define __TotalVariationCostFunctionWithImplicitGradientDescent_txx

#include "itkImageDuplicator.h"
#include "itkImageFileWriter.h"

#include "itkChangeInformationImageFilter.h"

#include <vcl_iostream.h>  

namespace mialsrtk
{

template <class TImage>
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::TotalVariationCostFunctionWithImplicitGradientDescent(unsigned int dim)
{
    lambda = 0.1;
    gamma=1.0;
    normD=12.0;
    tau=1/sqrt(normD);
    sigma=1/sqrt(normD);
    m_PSF = FunctionType::GAUSSIAN;
    m_SliceGap = 0.0;
    m_ComputeH = true;
    m_UseDebluringPSF = false;
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::set(float * array, int size, float value)
{
    for (int i = 0; i < size; i++)
        array[i] = value;
}

// Mirror of the position pos. abs(pos) must not be > 2*(size-1)
template <class TImage>
int
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::mirror(int pos, int size)
{
    int output = abs(pos);

    while(output >= size)
    {
        output = std::abs(output - (output - size + 1) * 2);
    }

    while(output < 0)
    {
        output = std::abs(output);
    }

    return output;
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::get_row(const vnl_vector<float>& image,
                                                                       img_size & size, int row, int frame, float * output)
{
    for (unsigned int i = 0; i < size.width; i++)
        output[i] = image[i + row * size.width + frame * size.width * size.height];
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::set_row(vnl_vector<float>& image,
                                                                       img_size & size, int row, int frame, float * input)
{
    for (unsigned int i = 0; i < size.width; i++)
        image[i + row * size.width + frame * size.width * size.height] = input[i];
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::get_col(const vnl_vector<float>& image,
                                                                       img_size & size, int col, int frame, float * output)
{
    for (unsigned int i = 0; i < size.height; i++)
        output[i] = image[col + i * size.width + frame * size.width * size.height];
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::set_col(vnl_vector<float>& image,
                                                                       img_size & size, int col, int frame, float * input)
{
    for (unsigned int i = 0; i < size.height; i++)
        image[col + i * size.width + frame * size.width * size.height] = input[i];
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::get_spec(const vnl_vector<float>& image,
                                                                        img_size & size, int row, int col, float * output)
{
    for (unsigned int i = 0; i < size.depth; i++)
        output[i] = image[col + row * size.width + i * size.width * size.height];
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::set_spec(vnl_vector<float>& image,
                                                                        img_size & size, int row, int col, float * input)
{
    for (unsigned int i = 0; i < size.depth; i++)
        image[col + row * size.width + i * size.width * size.height] = input[i];
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::convol1d(float * kernel, int ksize,
                                                                        float * src, int src_size, float * dest)
{
    int n2 = (int)std::floor(ksize / 2);
    int k;

    set(dest, src_size, 0);
    for (int i = 0; i < src_size; i++)
    {
        for (int j = 0; j < ksize; j++)
        {
            k = i + j - n2;
            k = mirror(k, src_size);
            dest[i] += kernel[j] * src[k];
        }
    }
}

// 3D convolution : over the rows
template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::convol3dx(const vnl_vector<float>& image,
                                                                         vnl_vector<float>& image_conv, img_size& size, float * kernel, int ksize)
{
    float * drow = new float[size.width];
    float * srow = new float[size.width];

    for (unsigned int l = 0; l < size.depth; l++) {
        for (unsigned int py = 0; py < size.height; py++) {
            get_row(image, size, py, l, srow);
            convol1d(kernel, ksize, srow, size.width, drow);
            set_row(image_conv, size, py, l, drow);
        }
    }

    delete[] drow;
    delete[] srow;
}

// 3D convolution : over the columns
template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::convol3dy(const vnl_vector<float>& image,
                                                                         vnl_vector<float>& image_conv, img_size & size, float * kernel, int ksize)
{
    float * dcol = new float[size.height];
    float * scol = new float[size.height];

    for (unsigned int l = 0; l < size.depth; l++) {
        for (unsigned int px = 0; px < size.width; px++) {
            get_col(image, size, px, l, scol);
            convol1d(kernel, ksize, scol, size.height, dcol);
            set_col(image_conv, size, px, l, dcol);
        }
    }

    delete[] dcol;
    delete[] scol;
}

// 3D convolution : over the spectra
template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::convol3dz(const vnl_vector<float>& image,
                                                                         vnl_vector<float>& image_conv, img_size & size, float * kernel, int ksize)
{
    float * dspec = new float[size.depth];
    float * sspec = new float[size.depth];

    for (unsigned int py = 0; py < size.height; py++) {
        for (unsigned int px = 0; px < size.width; px++) {
            get_spec(image, size, py, px, sspec);
            convol1d(kernel, ksize, sspec, size.depth, dspec);
            set_spec(image_conv, size, py, px, dspec);
        }
    }

    delete[] dspec;
    delete[] sspec;
}


template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::update()
{

    //std::cout<<std::endl<<"Inner loop update :"<<std::endl;
    //std::cout<<"old x:"<<X.sum()<<" ("<<&X<<")"<<std::endl;

    //Old implementation
    /*
    vnl_vector<float> Ax;
    Ax.set_size(X.size());
    A.pre_mult(X,Ax);

    X = ( X + deltat * Xold - Ax + b ) / ( 1 + deltat );

    for(int i = 0; i < X.size(); i++)
    {
        if(X[i] < 0)
        {
            X[i] = 0.0;
        }
    }

    Ax.clear();
    */

    //New implementation

    vnl_vector<float> Hx;
    H.mult(X,Hx);

    // Calculate Ht*Hx. Note that this is calculated as Hx*H since
    // Ht*Hx = (Hxt*H)t and for Vnl (Hxt*H)t = Hxt*H = Hx*H because
    // the vnl_vector doesn't have a 2nd dimension. This allows us
    // to save a lot of memory because we don't need to store Ht.

    vnl_vector<float> HtHx;
    H.pre_mult(Hx,HtHx);
    Hx.clear();

    float oneMinusLambda=(1.0f -lambda);

    //X = ( X + deltat * lambda* Xold + oneMinusLambda * deltat * tau * HtHx + b ) / ( 1 + deltat );
    X = ( X + deltat * Xold - deltat * lambda * tau * HtHx + b ) / ( 1 + deltat );

    HtHx.clear();

    //float max = Y.max_value();
    //float max = 638.0;

    //std::cout << "X min : " << X.min_value() << std::endl;
    //std::cout << "X max : " << X.max_value() << std::endl;

    //Projection onto the positive convex set
    for(int i = 0; i < X.size(); i++)
    {
        if(X[i] < 0.0)
        {
            X[i] = 0.0;
        }
        else if(X[i]>255.0)
        {
            X[i] = 255.0;
        }
    }

    //X = X - X.min_value();
    //std::cout << "New X min : " << X.min_value() << std::endl;
    //std::cout << "New X max : " << X.max_value() << std::endl;
    //std::cout<<"new x:"<<X.sum()<<" ("<<&X<<")"<<std::endl;
}

template <class TImage>
vnl_vector<float>
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::GetX()
{
    return X;
}

template <class TImage>
double
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::energy_value()
{
    // Calculate the error with respect to the low resolution images

    //std::cout << "X in energy_value() : "<< X.sum() << " (" << &X << ")" <<std::endl;

    vnl_vector<float> Hx;
    H.mult(X,Hx);

    vnl_vector<float> HxMinusY;
    HxMinusY = Hx - Y;
    //HxMinusY = Y - Hx;

    Hx.clear();

    double mse = HxMinusY.squared_magnitude()/ HxMinusY.size();

    HxMinusY.clear();

    //Computes <X,DivP>
    double XdotDivP = dot_product(X,DivP) / X.size();

    //Computes the MSE between X at previous and current iterations, estimates X at future iteration
    vnl_vector<float> XcurrentMinusXold;
    XcurrentMinusXold.set_size( X.size() );
    XcurrentMinusXold= X - Xold ;
    double l2dist = XcurrentMinusXold.squared_magnitude() / XcurrentMinusXold.size();
    XcurrentMinusXold.clear();

    float oneMinusLambda=(1.0f -lambda);
    //double value = 0.5 * ( oneMinusLambda *  tau * mse + lambda * l2dist ) - tau * lambda * XdotDivP;

    double value = 0.5 * ( lambda * tau * mse + l2dist ) - tau * XdotDivP;

    //std::cout << "Global energy = " << value << " ( 0.5*lambda*tau*MSE = "<< 0.5*lambda*tau*mse <<" , (1/2)*l2dist = "<<(0.5)*l2dist<<", tau * <X,DivP> = " << tau * XdotDivP << " )" << std::endl;
    //std::cout << "Fixed parameters : sigma = " << sigma << " , tau = "<<tau<<" , theta = "<<theta<<" , lambda = "<<lambda<<" , gamma = "<<gamma<<std::endl;
    //std::cout << "sum(x) = "<<X.sum()<< " , sum(xold) = "<<Xold.sum()<<" , sum(xest) = "<< Xest.sum() <<" , sum(Divp) = "<<DivP.sum()<<std::endl;
    //std::cout << "---------------------------------------------------------------------" << std::endl;

    //delete[] bKernel;

    return value;

}
/**Sets the solut
ion image at the k-th iteration. */
template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetX(const vnl_vector<float>& x)
{
    X = x;
}

/**Initialization of image x estimated at iteration n+1. */
template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetXest(const vnl_vector<float>& x)
{
    std::cout << "Xest (resampler)= " << x.sum() << std::endl;
    Xest = x;
    std::cout << "Xest (costfunction) = " << Xest.sum() << std::endl << std::endl;

    /*
  // Converts and writes output image
  typename itk::ImageDuplicator<ImageType>::Pointer duplicator = itk::ImageDuplicator<ImageType>::New();
  duplicator->SetInputImage(m_ReferenceImage);
  duplicator->Update();
  ImagePointer outputIm = duplicator->GetOutput();

  typename ImageType::RegionType outputImageRegion = outputIm -> GetLargestPossibleRegion();

  */

    /*
  std::cout << outputImageRegion << std::endl;
  */

    /*
  itk::ImageRegionIterator<ImageType> outputIt( outputIm,outputImageRegion );
  unsigned int linearIndex = 0;
  for (outputIt.GoToBegin(); !outputIt.IsAtEnd(); ++outputIt, linearIndex++)
    outputIt.Set(Xest[linearIndex]);

  typename itk::ImageFileWriter< ImageType >::Pointer writer =  itk::ImageFileWriter< ImageType >::New();
  writer -> SetFileName( "/Users/sebastientourbier/Desktop/Patient01/SR/Manual/xest_costfunction.nii.gz" );
  writer -> SetInput( outputIm );
  writer -> Update();

  linearIndex = 0;
  for (outputIt.GoToBegin(); !outputIt.IsAtEnd(); ++outputIt, linearIndex++)
    outputIt.Set(x[linearIndex]);

  typename itk::ImageFileWriter< ImageType >::Pointer writer2 =  itk::ImageFileWriter< ImageType >::New();
  writer2 -> SetFileName( "/Users/sebastientourbier/Desktop/Patient01/SR/Manual/xest_passed_from_resampler.nii.gz" );
  writer2 -> SetInput( outputIm );
  writer2 -> Update();
  */

}

/**Initialization of image x corresponding to the image solution at iteration n-1. */
template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetXold(const vnl_vector<float>& x)
{
    std::cout << "Xold (resampler)= " << x.sum() << std::endl;
    Xold = x;
    std::cout << "Xold (costfunction) = " << Xold.sum() << std::endl;
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::Initialize()
{
    //itk::ChangeInformationImageFilter < ImageType > ChangeInfoFilterType;
    //Add gap between slices to the spacing in the slice-select direction
    /*
    for(unsigned int im=0; im < m_Images.size(); im++)
    {
        ChangeInfoFilterType::Pointer infoFilter = ChangeInfoFilterType::New();
        infoFilter->SetOutputSpacing( spacing );
        infoFilter->ChangeSpacingOn();
    }
    */


    //We use linear interpolation for the estimation of point influence in matrix H
    typedef itk::BSplineInterpolationWeightFunction<double, 3, 1> itkBSplineFunction;

    m_OutputImageRegion = m_ReferenceImage -> GetLargestPossibleRegion();
    IndexType start_hr  = m_OutputImageRegion.GetIndex();
    SizeType  size_hr   = m_OutputImageRegion.GetSize();

    //x_size : size of the SR image (used in other functions)
    x_size.width  = size_hr[0];
    x_size.height = size_hr[1];
    x_size.depth  = size_hr[2];

    IndexType end_hr;
    end_hr[0] = start_hr[0] + size_hr[0] - 1 ;
    end_hr[1] = start_hr[1] + size_hr[1] - 1 ;
    end_hr[2] = start_hr[2] + size_hr[2] - 1 ;

    // Differential continuous indexes to perform the neighborhood iteration
    SpacingType spacing_lr = m_Images[0] -> GetSpacing();
    SpacingType spacing_hr = m_ReferenceImage -> GetSpacing();

    //spacing_lr[2] is assumed to be the lowest resolution
    //compute the index of the PSF in the LR image resolution
    std::vector<ContinuousIndexType> deltaIndexes;

    double upsamp = 5.0;

    double ratioLRHRX = upsamp*spacing_lr[0] / spacing_hr[0];
    double ratioLRHRY = upsamp*spacing_lr[1] / spacing_hr[1];

    double ratioHRLRX =  (1.0/upsamp)*spacing_hr[0] / spacing_lr[0];
    double ratioHRLRY =  (1.0/upsamp)*spacing_hr[1] / spacing_lr[1];

    double ratioLRHRZ = upsamp*spacing_lr[2] / spacing_hr[2];
    double ratioHRLRZ = (1.0/upsamp)*spacing_hr[2] / spacing_lr[2];

    bool ratioXisEven = true;
    bool ratioYisEven = true;
    bool ratioZisEven = true;
    const bool verbose = false;
    
    if((((int)round(ratioLRHRX)) % 2)) ratioXisEven = false;
    if((int)round(ratioLRHRY) % 2) ratioYisEven = false;
    if((int)round(ratioLRHRZ) % 2) ratioZisEven = false;

    std::cout << "ratioXisEven : " << ratioXisEven << std::endl;
    std::cout << "ratioYisEven : " << ratioYisEven << std::endl;
    std::cout << "ratioZisEven : " << ratioZisEven << std::endl;

    float factorPSF=1.0;
    int npointsX = 0;
    float initpointX = 0.0;
    if(ratioXisEven)
    {
        int k = floor(0.5* ((factorPSF-ratioHRLRX)/ratioHRLRX));
        npointsX = 2 * (k+1);
        std::cout << "npointx 1: " << npointsX << std::endl;
        initpointX = - (float)(0.5+k) * ratioHRLRX;
    }
    else
    {
        int k = floor(factorPSF*0.5 /ratioHRLRX);
        npointsX = 2*k + 1;
        std::cout << "npointx 2: " << npointsX << std::endl;
        initpointX = - (float)(k) * ratioHRLRX;
    }

    int npointsY = 0;
    float initpointY = 0.0;
    if(ratioYisEven)
    {
        int k = floor(0.5 * ((factorPSF-ratioHRLRY)/ratioHRLRY));
        npointsY = 2 * (k+1);
        std::cout << "npointy 1: " << npointsY << std::endl;
        initpointY = - (float)(0.5+k) * ratioHRLRY;
    }
    else
    {
        int k = floor(factorPSF*0.5 /ratioHRLRY);
        npointsY = 2*k + 1;
        std::cout << "npointy 2: " << npointsY << std::endl;
        initpointY = - (float)(k) * ratioHRLRY;
    }

    int npointsZ = 0;
    float initpointZ = 0.0;
    if(ratioZisEven)
    {
        int k = floor(0.5 * ((factorPSF-ratioHRLRZ)/ratioHRLRZ));
        npointsZ = 2 * (k+1);
        std::cout << "npointz 1: " << npointsZ << std::endl;
        initpointZ = - (float)(0.5+k) * ratioHRLRZ;
    }
    else
    {
        int k = floor(factorPSF*0.5/ratioHRLRZ);
        npointsZ = 2*k + 1;
        std::cout << "npointz 2: " << npointsZ << std::endl;
        initpointZ = - (float)(k) * ratioHRLRZ;
    }

    std::cout << "Spacing LR X: " << spacing_lr[0] << " / Spacing HR X: " << spacing_hr[0]<< std::endl;
    std::cout << "Spacing LR Y: " << spacing_lr[1] << " / Spacing HR Y: " << spacing_hr[1]<< std::endl;
    std::cout << "Spacing LR Z: " << spacing_lr[2] << " / Spacing HR Z: " << spacing_hr[2]<< std::endl;

    std::cout << " , 1/2 * LR/HR X:" << 0.5 * ratioLRHRX << " , NPointsX : " << npointsX << std::endl;
    std::cout << " , 1/2 * LR/HR Y:" << 0.5 * ratioLRHRY << " , NPointsY : " << npointsY << std::endl;
    std::cout << " , 1/2 * LR/HR Z:" << 0.5 * ratioLRHRZ << " , NPointsZ : " << npointsZ << std::endl;


    ContinuousIndexType delta;
    delta[0] = 0;
    delta[1] = 0;
    delta[2] = 0;

    std::vector<float> gaussPSF;
    float cst = 8*sqrt(log(2));//FWHM: 2.35...
    float weight = 0.0;
    std::vector<float> dist(3);

    std::vector<float> var(3);
    var[0] = 1.2*(spacing_lr[0]/cst);
    var[0] = var[0] * var[0];

    var[1] = 1.2*(spacing_lr[1]/cst);
    var[1] = var[1] * var[1];

    var[2] = (spacing_lr[2]/cst);
    var[2] = var[2] * var[2];

    for(int i = 0; i < npointsX; i++)
    {
        for(int j = 0; j < npointsY; j++)
        {
            for(int k = 0; k < npointsZ; k++)
            {
                delta[0] = initpointX + (float)i * ratioHRLRX;
                delta[1] = initpointY + (float)j * ratioHRLRY;
                delta[2] = initpointZ + (float)k * ratioHRLRZ;

                dist[0] = initpointX + (float)i * ratioHRLRX;
                dist[1] = initpointY + (float)j * ratioHRLRY;
                dist[2] = initpointZ + (float)k * ratioHRLRZ;


                weight = exp( - 0.5* ( ((dist[0]*dist[0])/var[0] ) + ((dist[1]*dist[1])/var[1] ) +((dist[2]*dist[2])/var[2] )  ) );

                 gaussPSF.push_back(weight);
                deltaIndexes.push_back(delta);
                if (verbose){
                std::cout << " delta : " << delta[0] << " , " << delta[1] << " , " << delta[2] << " ;  w = "<< weight << std::endl;
                }

            }
        }
    }

    //    // Differential continuous indexes to perform the neighborhood iteration
    //    SpacingType spacing_lr = m_Images[0] -> GetSpacing();
    //    SpacingType spacing_hr = m_ReferenceImage -> GetSpacing();

    //    //spacing_lr[2] is assumed to be the lowest resolution
    //    //compute the index of the PSF in the LR image resolution
    //    std::vector<ContinuousIndexType> deltaIndexes;
    //    int npoints =  ceil(spacing_lr[2] / (2.0 * spacing_hr[2])) ;

    //    std::cout << "Spacing LR : " << spacing_lr[2] << " / Spacing HR : " << spacing_hr[2];

    //    std::cout << "NPoints : " << 2*(npoints)+1 << std::endl;

    //    ContinuousIndexType delta;
    //    delta[0] = 0.0; delta[1] = 0.0;

    //    // // doubled PSF
    //    //for (int i = -npoints ; i <= npoints; i++ )
    //    for (int i =( -1*npoints) ; i <= (1*npoints); i++ )
    //    {
    //        //FIXED
    //        // delta[2] = i * 0.5 / (double) npoints; //BTK version!
    //        //delta[2] = i / (double) (2 * npoints + 1);
    //        //delta[2] = (double) i  * spacing_hr[2];
    //        //delta[2] = (double) i ;
    //        delta[2] = ((double) i  * spacing_hr[2]) / spacing_lr[2]; // - 0.3 /spacing_lr[2]; // Space of 1 between Index in the LR image corresponds to the slice thickness (3mm for instance).
    //        deltaIndexes.push_back(delta);
    //    }

    //    std::cout << "DeltaIndexes : " << deltaIndexes[0] << deltaIndexes[1] << deltaIndexes[2] << std::endl;

    // Set size of matrices
    unsigned int ncols = m_OutputImageRegion.GetNumberOfPixels();

    unsigned int nrows = 0;
    for(unsigned int im = 0; im < m_Images.size(); im++)
        nrows += m_Regions[im].GetNumberOfPixels();
    /*
     {
        ConstIteratorType itIM(m_Images[im],m_Regions[im]);

        for(itIM.GoToBegin(); !itIM.IsAtEnd(); ++itIM)
            {
                PointType pnt;
                m_Images[im]->TransformIndexToPhysicalPoint(itIM.GetIndex(),pnt);
                if(m_Masks[im]->IsInside(pnt))
                    nrows += 1;
            }
      }*/

    ////nrows += m_Regions[im].GetNumberOfPixels();

    vnl_sparse_matrix<float> fullH;
    vnl_vector<float> fullY;


    if(m_ComputeH)
    {
        std::cout << "Initialize H and Z ..." << std::endl;
        fullH.set_size(nrows, ncols);
        Xsamp.set_size(ncols);
        Xsamp.fill(0.0);
        Z.set_size(nrows);
        Z.fill(0.0);
    }
    else{
        Z.set_size(nrows);
        Z.fill(0.0);
    }

    fullY.set_size(nrows);
    fullY.fill(0.0);

    //e.set_size(nrows);
    //e.fill(0.0);


    std::cout << "Size of fullH :  #rows = " << fullH.rows() << ", #cols = "<<fullH.cols() << std::endl;

    /*
  Xold.set_size(ncols);
  Xold = vnl_matops::d2f(x_init);

  Xest.set_size(ncols);
  Xest = vnl_matops::d2f(x_init);
  */



    //m_xold.set_size(ncols);
    //m_xold.fill(0.0);

    //Rescale image intensity to range [0,1]
    /*
  typename RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
  rescaleFilter->SetInput(m_ReferenceImage.GetPointer());
  rescaleFilter->SetOutputMinimum(0);
  rescaleFilter->SetOutputMaximum(1.0);
  rescaleFilter->Update();
  m_ReferenceImage = rescaleFilter->GetOutput();
  //m_ReferenceImage->DisconnectPipeline();
  */

    int counterDebug = 0;
    unsigned int im;
//#pragma omp parallel for private(im) schedule(dynamic)
    for(im = 0; im < m_Images.size(); im++)
    {
        //Rescale image intensity to range [0,1]
        /*
    typename RescaleFilterType::Pointer rescaleFilterLR = RescaleFilterType::New();
    rescaleFilterLR->SetInput(m_Images[im].GetPointer());
    rescaleFilterLR->SetOutputMinimum(0);
    rescaleFilterLR->SetOutputMaximum(1.0);
    rescaleFilterLR->Update();
    m_Images[im] = rescaleFilterLR->GetOutput();
    //m_Images[im]->DisconnectPipeline();
    */

        // Interpolator for HR image
        InterpolatorPointer interpolator = InterpolatorType::New();
        interpolator -> SetInputImage( m_ReferenceImage );

        SpacingType inputSpacing = m_Images[im] -> GetSpacing();

        SpacingType inputSpacing2 = m_Images[im] -> GetSpacing();
        inputSpacing2[0] = inputSpacing2[0] ;
        inputSpacing2[1] = inputSpacing2[1] ;
        inputSpacing2[2] = inputSpacing2[2];

        std::cout << "input spacing 2 : " << inputSpacing2 << std::endl;


        // PSF definition
        typename FunctionType::Pointer function = FunctionType::New();
        function -> SetPSF(  FunctionType::GAUSSIAN );
        function -> SetRES(FunctionType::ANISOTROPIC );
        function -> SetDirection( m_Images[im] -> GetDirection() );

        //std::cout << "Image # "<< im << " sizes : " << m_Images[im]->GetLargestPossibleRegion().GetSize() << std::endl;
        //std::cout << "Image # "<< im << " direction : " << m_Images[im]->GetDirection() << std::endl;
        //std::cout << "Image # "<< im << " spacing : " << m_Images[im]->GetSpacing() << std::endl;

        function -> SetSpacing( inputSpacing2 );

        // PSF HR definition
        typename FunctionType::Pointer functionHR = FunctionType::New();
        functionHR -> SetPSF(  FunctionType::GAUSSIAN );
        functionHR -> SetRES( FunctionType::ISOTROPIC );
        functionHR -> SetDirection( m_ReferenceImage -> GetDirection() );
        functionHR -> SetSpacing( m_ReferenceImage->GetSpacing() );



        //function -> SetSpacing( m_Images[im] -> GetSpacing() );
        //function -> Print(std::cout);

        //Define the ROI of the current LR image
        IndexType inputIndex = m_Regions[im].GetIndex();
        SizeType  inputSize  = m_Regions[im].GetSize();

        //Define all indexes needed for iteration over the slices
        IndexType lrIndex;              //index of a point in the LR image im
        IndexType lrDiffIndex;          //index of this point in the current ROI of the LR image im
        unsigned int lrLinearIndex;     //index lineaire de ce point dans le vecteur

        IndexType hrIndex;
        IndexType hrDiffIndex;
        ContinuousIndexType hrContIndex;
        ContinuousIndexType lrContIndex;
        unsigned int hrLinearIndex;

        ContinuousIndexType nbIndex;

        PointType lrPoint;            //PSF center in world coordinates (PointType = worldcoordinate for ITK)
        PointType nbPoint;            //PSF point in world coordinate
        PointType hrPoint;            //PSF point in world coordinate
        PointType transformedPoint;   //after applying current transform (registration)
         PointType invTransformedPoint;   //after applying current transform (registration)

        unsigned int offset = 0;
        for(unsigned int im2 = 0; im2 < im; im2++)
            offset += m_Regions[im2].GetNumberOfPixels();

        std::vector<float> listHrIndex;
        std::vector<float> listHrWeight;

        // Iteration over slices
        for ( unsigned int i=inputIndex[2]; i < inputIndex[2] + inputSize[2]; i++ )
        {

            //TODO: outlier rejection scheme, if slice was excluded, we process directly the next one
            // It would probably require to save a list of outlier slices during motion correction and
            // to load it here as input and create a global vector.

            RegionType wholeSliceRegion;
            wholeSliceRegion = m_Regions[im];

            IndexType  wholeSliceRegionIndex = wholeSliceRegion.GetIndex();
            SizeType   wholeSliceRegionSize  = wholeSliceRegion.GetSize();

            wholeSliceRegionIndex[2]= i;
            wholeSliceRegionSize[2] = 1;

            wholeSliceRegion.SetIndex(wholeSliceRegionIndex);
            wholeSliceRegion.SetSize(wholeSliceRegionSize);

            ConstIteratorType fixedIt( m_Images[im], wholeSliceRegion);

            double lrValue;
            double hrValue;

            for(fixedIt.GoToBegin(); !fixedIt.IsAtEnd(); ++fixedIt)
            {
                //Current index in the LR image
                lrIndex = fixedIt.GetIndex();

                //lrIndex[0] = lrIndex[0] + deltaIndexes[1][0];
                //lrIndex[1] = lrIndex[1] + deltaIndexes[1][1];
                //lrIndex[2] = lrIndex[2] + deltaIndexes[1][2];

                //World coordinates of lrIndex using the image header
                m_Images[im] -> TransformIndexToPhysicalPoint( lrIndex, lrPoint );

                //Compute the coordinates in the SR using the estimated registration
                transformedPoint = m_Transforms[im][i] -> TransformPoint( lrPoint );
                //transformedPoint = lrPoint;

                //check if this point is in the SR image (m_ReferenceImage)

                if ( ! interpolator -> IsInsideBuffer( transformedPoint ) )
                    continue;


                //From the LR image coordinates to the LR ROI coordinates
                lrDiffIndex[0] = lrIndex[0] - inputIndex[0];
                lrDiffIndex[1] = lrIndex[1] - inputIndex[1];
                lrDiffIndex[2] = lrIndex[2] - inputIndex[2];

                //Compute the corresponding linear index of lrDiffIndex
                if(1)
                {
                    lrLinearIndex = lrDiffIndex[0] + lrDiffIndex[1]*inputSize[0] +
                            lrDiffIndex[2]*inputSize[0]*inputSize[1];
                }
                else
                {
                    lrLinearIndex = lrDiffIndex[0] + lrDiffIndex[1]*inputSize[0] +
                            lrDiffIndex[2]*inputSize[0]*inputSize[1];
                }
                //Get the intensity value in the LR image
                //Y[lrLinearIndex + offset] = fixedIt.Get();
                if ( m_Masks.size() > 0)
                {
                    if ( m_Masks[im] -> IsInside(lrPoint) )
                    {
                        //std::cout << "add point yk..." << std::endl;
                        fullY[lrLinearIndex + offset] = fixedIt.Get();
                        //sliceIds[lrLinearIndex + offset] = sliceId;
                    }
                    else
                    {
                        continue;
                    }
                }

                //std::cout << "ids :::: " << sliceIds[lrLinearIndex + offset] << std::endl;

                //Set the center point of the PSF
                function -> SetCenter( lrPoint );

                if(!listHrIndex.empty()) listHrIndex.clear();
                if(!listHrWeight.empty()) listHrWeight.clear();

                //function -> Print(std::cout);

                //lrIndex[0] = lrIndex[0] - deltaIndexes[1][0];
                //lrIndex[1] = lrIndex[1] - deltaIndexes[1][1];
                //lrIndex[2] = lrIndex[2] - deltaIndexes[1][2];

                if(m_ComputeH)
                {
                    //std::cout << "Populates H ..." << std::endl;
                    //Loop over points of the PSF

                    //std::cout << "Loop over PSF points : " << deltaIndexes.size() << "points" << std::endl;
                    for(unsigned int k=0; k<deltaIndexes.size(); k++)
                    {
                        //Coordinates in the LR image
                        nbIndex[0] = deltaIndexes[k][0] + lrIndex[0];
                        nbIndex[1] = deltaIndexes[k][1] + lrIndex[1];
                        nbIndex[2] = deltaIndexes[k][2] + lrIndex[2];

                        //World coordinates using LR image header
                        m_Images[im] -> TransformContinuousIndexToPhysicalPoint( nbIndex, nbPoint );

                        //Compute the PSF value at this point
                        //lrValue = function -> Evaluate(nbPoint);

                        lrValue = gaussPSF[k];
                       // std::cout << lrValue << "   ";

                        if ( lrValue > 0)
                        {
                            //Compute the world coordinate of this point in the SR image
                            // transformedPoint = nbPoint;
                            transformedPoint = m_Transforms[im][i] -> TransformPoint( nbPoint );


                            //Set this coordinate in continuous index in SR image space
                            m_ReferenceImage -> TransformPhysicalPointToContinuousIndex(
                                        transformedPoint, hrContIndex );

                           m_ReferenceImage->TransformPhysicalPointToIndex(transformedPoint,hrIndex);
                           m_ReferenceImage->TransformIndexToPhysicalPoint(hrIndex,hrPoint);

                            functionHR -> SetCenter(hrPoint );
                            hrValue = function -> Evaluate(nbPoint);

                            // OLD VERSION (BTK V1)

                            bool isInsideHR = true;

                            // FIXME This checking should be done for all points first, and
                            // discard the point if al least one point is out of the reference
                            // image

                            if ( (hrContIndex[0] < start_hr[0]) || (hrContIndex[0] > end_hr[0]) ||
                                 (hrContIndex[1] < start_hr[1]) || (hrContIndex[1] > end_hr[1]) ||
                                 (hrContIndex[2] < start_hr[2]) || (hrContIndex[2] > end_hr[2]) )
                                isInsideHR = false;


                            if ( (hrIndex[0] < start_hr[0]) || (hrIndex[0] > end_hr[0]) ||
                                 (hrIndex[1] < start_hr[1]) || (hrIndex[1] > end_hr[1]) ||
                                 (hrIndex[2] < start_hr[2]) || (hrIndex[2] > end_hr[2]) )
                                isInsideHR = false;


                            if(isInsideHR)
                            {
                                //Index in the ROI of the SR index
                                hrDiffIndex[0] = hrIndex[0] - start_hr[0];
                                hrDiffIndex[1] = hrIndex[1] - start_hr[1];
                                hrDiffIndex[2] = hrIndex[2] - start_hr[2];

                                //Compute the corresponding linear index
                                hrLinearIndex = hrDiffIndex[0] + hrDiffIndex[1]*size_hr[0] +
                                        hrDiffIndex[2]*size_hr[0]*size_hr[1];

                                Xsamp[hrLinearIndex]=Xsamp[hrLinearIndex]+1;

                                /*
                                m_ReferenceImage->TransformIndexToPhysicalPoint(hrIndex,hrPoint);

                                TransformPointerType inv_transform = TransformType::New();
                                inv_transform -> SetCenter(m_Transforms[im][i]->GetCenter());
                                bool response2 = m_Transforms[im][i] ->GetInverse(inv_transform);

                                invTransformedPoint = inv_transform->TransformPoint(hrPoint);

                                 m_Images[im]->TransformPhysicalPointToContinuousIndex(invTransformedPoint,lrContIndex);

                                dist[0] = initpointX + (float)i * ratioHRLRX;
                                dist[1] = initpointY + (float)j * ratioHRLRY;
                                dist[2] = initpointZ + (float)k * ratioHRLRZ;

                                lrValue = exp( - 0.5* ( ((dist[0]*dist[0])/var[0] ) + ((dist[1]*dist[1])/var[1] ) +((dist[2]*dist[2])/var[2] )  ) );
                                */
                                //Add the correct value in H !
                                if(fullH(lrLinearIndex + offset, hrLinearIndex) <lrValue) fullH(lrLinearIndex + offset, hrLinearIndex) =   lrValue;
                            }


                            /*
                            if ( isInsideHR )
                            {
                                //Compute the corresponding value in the SR image -> useless
                                //Allows to compute the set of contributing neighbors
                                hrValue = interpolator -> Evaluate( transformedPoint );

                                //std::cout << "Number of contributing neighbors for point " << transformedPoint << " : " << interpolator -> GetContributingNeighbors() << std::endl;
                                //std::cout << hrValue << "  ";

                                //Loop over points affected using the interpolation
                                for(unsigned int n=0; n<interpolator -> GetContributingNeighbors();
                                    n++)
                                {
                                    //Index in the SR image
                                    hrIndex = interpolator -> GetIndex(n);

                                    if ( (hrIndex[0] < start_hr[0]) || (hrIndex[0] > end_hr[0]) ||
                                         (hrIndex[1] < start_hr[1]) || (hrIndex[1] > end_hr[1]) ||
                                         (hrIndex[2] < start_hr[2]) || (hrIndex[2] > end_hr[2]) )
                                        isInsideHR = false;

                                    if ( isInsideHR )
                                    {
                                        m_ReferenceImage->TransformIndexToPhysicalPoint(hrIndex,hrPoint);

                                        TransformPointerType inv_transform = TransformType::New();
                                        inv_transform -> SetCenter(m_Transforms[im][i]->GetCenter());
                                        bool response = m_Transforms[im][i] ->GetInverse(inv_transform);

                                        invTransformedPoint = inv_transform->TransformPoint(hrPoint);

                                        //functionHR -> SetCenter(hrPoint );
                                       // hrValue = function -> Evaluate(nbPoint);
                                        lrValue = function -> Evaluate(invTransformedPoint);

                                        //std::cout << lrValue  << "  ";

                                        //if(lrValue >= 0.5)//Inside PSF
                                        //{

                                            //Index in the ROI of the SR index
                                            hrDiffIndex[0] = hrIndex[0] - start_hr[0];
                                            hrDiffIndex[1] = hrIndex[1] - start_hr[1];
                                            hrDiffIndex[2] = hrIndex[2] - start_hr[2];

                                            //Compute the corresponding linear index
                                            hrLinearIndex = hrDiffIndex[0] + hrDiffIndex[1]*size_hr[0] +
                                                    hrDiffIndex[2]*size_hr[0]*size_hr[1];

                                            Xsamp[hrLinearIndex]=Xsamp[hrLinearIndex]+1;

                                            bool addPoint = true;
                                            for(int k=0;k<listHrIndex.size();k++)
                                            {
                                                if(listHrIndex[k]==hrLinearIndex && listHrWeight[k] > lrValue ) addPoint = false;
                                            }


                                            if(addPoint)
                                            {
                                                listHrIndex.push_back(hrLinearIndex);
                                                listHrWeight.push_back(lrValue);

                                                //std::cout << "Adding point with weigth : " << lrValue << std::endl;

                                                //Add the correct value in H !
                                                fullH(lrLinearIndex + offset, hrLinearIndex) +=lrValue;
                                                //H(lrLinearIndex + offset, hrLinearIndex) += interpolator -> GetOverlap(n)* 1.0;
                                                //H(lrLinearIndex + offset, hrLinearIndex) +=  lrValue;
                                                //H(lrLinearIndex + offset, hrLinearIndex) +=  lrValue;
                                            }
                                        //}
                                    }

                                }

                            }
                            */



                            /*
                            //std::cout << "Bspline flag#1" << std::endl;
                            itkBSplineFunction::Pointer bsplineFunction = itkBSplineFunction::New();
                            itkBSplineFunction::WeightsType bsplineWeights;
                            bsplineWeights.SetSize(8); // (bsplineOrder + 1)^3
                            itkBSplineFunction::IndexType   bsplineStartIndex;
                            itkBSplineFunction::IndexType   bsplineEndIndex;
                            itkBSplineFunction::SizeType    bsplineSize;
                            RegionType                      bsplineRegion;

                            //Get the interpolation weight using itkBSplineInterpolationWeightFunction
                            bsplineFunction->Evaluate(hrContIndex,bsplineWeights,bsplineStartIndex);

                            //Get the support size for interpolation
                            bsplineSize = bsplineFunction->GetSupportSize();

                            //Check if the bspline support region is inside the HR image
                            bsplineEndIndex[0] = bsplineStartIndex[0] + bsplineSize[0];
                            bsplineEndIndex[1] = bsplineStartIndex[1] + bsplineSize[1];
                            bsplineEndIndex[2] = bsplineStartIndex[2] + bsplineSize[2];

                            //std::cout << "bsplineStart" << bsplineStartIndex[0] << "," << bsplineStartIndex[1] << "," << bsplineStartIndex[2] << std::endl;
                            //std::cout <<"bsplineEnd" << bsplineEndIndex[0] << "," << bsplineEndIndex[1] << "," << bsplineEndIndex[2] << std::endl;
                            //std::cout <<"bsplineSize" << bsplineSize[0] << "," << bsplineSize[1] << "," << bsplineSize[2] << std::endl;

                            //std::cout << "Bspline flag#2" << std::endl;

                            if(m_ReferenceImage->GetLargestPossibleRegion().IsInside(bsplineStartIndex)
                                    && m_ReferenceImage->GetLargestPossibleRegion().IsInside(bsplineEndIndex))
                            {
                                //Set the support region
                                bsplineRegion.SetSize(bsplineSize);
                                bsplineRegion.SetIndex(bsplineStartIndex);

                                //std::cout << "Bspline flag#3" << std::endl;

                                //Instantiate an iterator on HR image over the bspline region
                                ImageRegionConstIteratorWithIndex< ImageType > itHRImage(m_ReferenceImage,bsplineRegion);

                                //linear index of bspline weights
                                unsigned int weightLinearIndex = 0;

                                //Loop over the support region
                                for(itHRImage.GoToBegin(); !itHRImage.IsAtEnd(); ++itHRImage)
                                {

                                    //Get coordinate in HR image
                                    IndexType hrIndex = itHRImage.GetIndex();
                                    //Compute the corresponding linear index
                                    if(im==0)
                                    {
                                        hrLinearIndex = hrIndex[0] + hrIndex[1]*size_hr[0] + hrIndex[2]*size_hr[0]*size_hr[1];
                                    }
                                    else
                                    {
                                        hrLinearIndex = hrIndex[0] + hrIndex[1]*size_hr[0] + hrIndex[2]*size_hr[0]*size_hr[1];
                                        //hrLinearIndex = hrIndex[0] + hrIndex[2]*size_hr[0] + hrIndex[1]*size_hr[0]*size_hr[2];//working with coro
                                        //hrLinearIndex = hrIndex[2] + hrIndex[0]*size_hr[2] + hrIndex[1]*size_hr[2]*size_hr[0];
                                        //hrLinearIndex = hrIndex[1] + hrIndex[2]*size_hr[1] + hrIndex[0]*size_hr[2]*size_hr[1];
                                        //hrLinearIndex = hrIndex[1] + hrIndex[0]*size_hr[1] + hrIndex[2]*size_hr[1]*size_hr[0];
                                        //hrLinearIndex = hrIndex[2] + hrIndex[1]*size_hr[2] + hrIndex[0]*size_hr[2]*size_hr[1];
                                        //hrLinearIndex = hrIndex[2] + hrIndex[0]*size_hr[2] + hrIndex[1]*size_hr[2]*size_hr[0];
                                    }

                                    //Add weight*PSFValue to the corresponding element in H
                                    H(lrLinearIndex, hrLinearIndex)  +=1.0 * bsplineWeights[weightLinearIndex];//BOXCAR profile
                                    // H(lrLinearIndex, hrLinearIndex)  +=lrValue * bsplineWeights[weightLinearIndex];//Gaussian profile
                                    weightLinearIndex++;

                                } //end of loop over the support region
                                //std::cout << "Bspline flag#4" << std::endl;

                            }// end if bspline index inside sr image
                            //
                            */

                        } // if psf point is inside sr image

                    }//End of loop over PSF points

                }// if H has to be computed

            }// Loop over all pixels of the slice

        }//Loop over all slices
    }

   // std::cout << "H norm: " << fullH.
/*
    nrows = 0;
    for(unsigned int id = 0; id <fullY.size() ; id++)
     {
        if(fullY.get(id)>0) nrows += 1;
      }

    Y.set_size(nrows);
    H.set_size(nrows,ncols);
    //H.fill(0.0);
*/

    if(m_ComputeH)
    {
        // Normalize H
        H = fullH;

        std::cout << "scale : "<< scale << std::endl;

        for (unsigned int i = 0; i < H.rows(); i++)
        {
            double sum = H.sum_row(i);

            VnlSparseMatrixType::row & r = H.get_row(i);
            VnlSparseMatrixType::row::iterator col_iter = r.begin();

            for ( ;col_iter != r.end(); ++col_iter)
                (*col_iter).second = (*col_iter).second / sum;
        }

        std::cout << "H was computed and normalized ..." << std::endl << std::endl;
        // Normalize H
        /*
        unsigned int cnt_id = 0;
        for (unsigned int i = 0; i < fullH.rows(); i++)
        {
            if(fullY.get(i)>0)
            {
                Y.put(cnt_id,fullY.get(i));

                double sum = fullH.sum_row(i);

                VnlSparseMatrixType::row & r = fullH.get_row(i);
                VnlSparseMatrixType::row::iterator col_iter = r.begin();

                //Compact H (only observation contained inside the mask) / Otherwise line in H for every observations even if not contained in the LR mask
                VnlSparseMatrixType::row & rc = H.get_row(cnt_id);
                VnlSparseMatrixType::row::iterator ccol_iter = rc.begin();

                for ( ;col_iter != r.end(); ++col_iter, ++ccol_iter)
                    (*ccol_iter).second = (*col_iter).second / sum;

                cnt_id++;
            }

        }
        fullH.clear();
        */

    }
    else
    {
        std::cout << "Initialization of H with old value" << std::endl << std::endl;
        //H=fullH;
    }

    scale=1;
    Y = fullY/scale;
    Xest /= scale;
    Xold /= scale;
    X /= scale;

    std::cout << "Size of Y:" << Y.size() << std::endl;
    std::cout << "Sizes of H:" << H.rows() << " x " << H.columns() << std::endl;

    //vnl_matrix<float> tH = vnl_trace(H);

    //vnl_vector<float> DivP;
    DivP.set_size(ncols);
    DivP.fill(0.0);

    //Old implementation:  Precomputes A = deltat * lambda * tau * Ht * H
    // BUT Ht * H is computationally expensive (multiplication of two sparse matrix)
    // New implementation: Use only multiplication between sparse matrix and a vector which are much more efficient
    /*
    if(m_ComputeH)
    {
        HtH = H.transpose() * H;
    }

    A = deltat * lambda * tau * HtH;
    */

    std::cout << "Update P :" << std::endl;
    std::cout << "Old values : ";
    std::cout<<"Px="<<Px.sum()<<" , Py="<<Py.sum()<<" , Pz="<<Pz.sum()<<" , Xest="<<Xest.sum()<<" , Xold="<<Xold.sum()<<", DivP="<<DivP.sum()<<" , b="<<b.sum()<<std::endl<<std::endl;
    //vsl_print_summary(vcl_cout,A);
    // Converts and writes output image
    typename itk::ImageDuplicator<ImageType>::Pointer duplicator2 = itk::ImageDuplicator<ImageType>::New();
    duplicator2->SetInputImage(m_ReferenceImage);
    duplicator2->Update();
    ImagePointer outputIm2 = duplicator2->GetOutput();

    typename ImageType::RegionType outputImageRegion2 = outputIm2 -> GetLargestPossibleRegion();

    itk::ImageRegionIterator<ImageType> outputIt2( outputIm2,outputImageRegion2 );
    unsigned int linearIndex2 = 0;
    for (outputIt2.GoToBegin(); !outputIt2.IsAtEnd(); ++outputIt2, linearIndex2++)
        outputIt2.Set(Xest[linearIndex2]);

    /*
  typename itk::ImageFileWriter< ImageType >::Pointer writer3 =  itk::ImageFileWriter< ImageType >::New();
  writer3 -> SetFileName( "/Users/sebastientourbier/Desktop/Patient01/SR/Manual/pre_xest.nii.gz" );
  writer3 -> SetInput( outputIm2 );
  writer3 -> Update();
  */

    //Precomputes splitting variable P as it is constant over the optimization
    // Creates backward and forward derivative kernels
    float* fKernel = new float[3];
    fKernel[0] = 0; fKernel[1] = -1; fKernel[2] = 1;

    float* bKernel = new float[3];
    bKernel[0] = -1; bKernel[1] = 1; bKernel[2] = 0;

    //Computes P buggy!!! Get & set Px to old value
    vnl_vector<float> DfxXest;
    DfxXest.set_size( ncols );
    convol3dx(Xest, DfxXest, x_size, fKernel, 3);
    Px = Px + sigma * DfxXest;
    //DfxXest.clear();

    vnl_vector<float> DfyXest;
    DfyXest.set_size( ncols );
    convol3dy(Xest, DfyXest, x_size, fKernel, 3);
    Py = Py + sigma * DfyXest;
    //DfyXest.clear();

    vnl_vector<float> DfzXest;
    DfzXest.set_size( ncols );
    convol3dz(Xest, DfzXest, x_size, fKernel, 3);
    Pz = Pz + sigma * DfzXest;
    //DfzXest.clear();

    // Converts and writes output image
    typename itk::ImageDuplicator<ImageType>::Pointer duplicator = itk::ImageDuplicator<ImageType>::New();
    duplicator->SetInputImage(m_ReferenceImage);
    duplicator->Update();
    ImagePointer outputIm = duplicator->GetOutput();

    typename ImageType::RegionType outputImageRegion = outputIm -> GetLargestPossibleRegion();

    itk::ImageRegionIterator<ImageType> outputIt( outputIm,outputImageRegion );
    unsigned int linearIndex = 0;
    for (outputIt.GoToBegin(); !outputIt.IsAtEnd(); ++outputIt, linearIndex++)
        outputIt.Set(DfxXest[linearIndex]);

    /*
  typename itk::ImageFileWriter< ImageType >::Pointer writer2 =  itk::ImageFileWriter< ImageType >::New();
  writer2 -> SetFileName( "/Users/sebastientourbier/Desktop/Patient01/SR/Manual/dfxxest.nii.gz" );
  writer2 -> SetInput( outputIm );
  writer2 -> Update();
  */

    DfxXest.clear();
    DfyXest.clear();
    DfzXest.clear();

    vnl_vector<double> dNormP = vnl_matops::f2d( element_product(Px,Px) + element_product(Py,Py) + element_product(Pz,Pz)) ;
    dNormP = dNormP.apply(sqrt);

    vnl_vector<float> NormP = vnl_matops::d2f(dNormP);
    dNormP.clear();

    //Normalizes P
    for(int i = 0; i < ncols ; i++)
    {
        if(NormP[i]>1)
        {
            //std::cout << "NormP[" << i << "] = " << NormP[i] << std::endl;
            Px[i] = Px[i] / NormP[i];
            Py[i] = Py[i] / NormP[i];
            Pz[i] = Pz[i] / NormP[i];
        }
    }
    NormP.clear();

    //std::cout << "Debug flag 6" << std::endl;

    //Computes DivP
    vnl_vector<float> DbxPx;
    DbxPx.set_size( ncols );
    convol3dx(Px, DbxPx, x_size, bKernel, 3);

    vnl_vector<float> DbyPy;
    DbyPy.set_size( ncols );
    convol3dy(Py, DbyPy, x_size, bKernel, 3);

    vnl_vector<float> DbzPz;
    DbzPz.set_size( ncols );
    convol3dz(Pz, DbzPz, x_size, bKernel, 3);

    //DivP = - (DbxPx + DbyPy + DbzPz);
    //vnl_vector<float> DivP;
    //DivP.set_size( ncols );
    DivP = DbxPx + DbyPy + DbzPz;

    // Precalcule Ht*Y. Note that this is calculated as Y*H since
    // Ht*Y = (Yt*H)t and for Vnl (Yt*H)t = (Yt*H) = Y*H because
    // the vnl_vector doesn't have a 2nd dimension. This allows us
    // to save a lot of memory because we don't need to store Ht.
    vnl_vector<float> HtY;
    HtY.set_size(ncols);
    H.pre_mult(Y,HtY);

    vnl_vector<float> HtZ;
    HtZ.set_size(ncols);
    H.pre_mult(Z,HtZ);

    //Precompute b as it is constant over optimization iteration. Used in update()
    float oneMinusLambda=(1.0f -lambda);
    //b = - oneMinusLambda * deltat * tau * ( HtY + HtZ ) - deltat * tau * lambda* DivP;
    b = deltat * lambda * tau * ( HtY + HtZ ) - deltat * tau * DivP;

    DbxPx.clear();
    DbyPy.clear();
    DbzPz.clear();

    std::cout << "New values : ";
    std::cout<<"Px="<<Px.sum()<<" , Py="<<Py.sum()<<" , Pz="<<Pz.sum()<<" , Xest="<<Xest.sum()<<" , Xold="<<Xold.sum()<<", DivP="<<DivP.sum()<<" , b="<<b.sum()<<std::endl<<std::endl;
    //vsl_print_summary(vcl_cout,A);
    HtY.clear();
    //DivP.clear();

    delete[] bKernel;
    delete[] fKernel;
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::Initialize2()
{
    //Computes Matrix H, vectorize LR observation in vector Y
    this -> ComputeSRHMatrix();

    //Computes optimization terms related the total variation regularization
    this -> ComputeTotalVariationTerms();
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::ComputeSRHMatrix()
{
    //We use linear interpolation for the estimation of point influence in matrix H
    typedef itk::BSplineInterpolationWeightFunction<double, 3, 1> itkBSplineFunction;

    m_OutputImageRegion = m_ReferenceImage -> GetLargestPossibleRegion();
    IndexType start_hr  = m_OutputImageRegion.GetIndex();
    SizeType  size_hr   = m_OutputImageRegion.GetSize();

    //x_size : size of the SR image (used in other functions)
    x_size.width  = size_hr[0];
    x_size.height = size_hr[1];
    x_size.depth  = size_hr[2];

    IndexType end_hr;
    end_hr[0] = start_hr[0] + size_hr[0] - 1 ;
    end_hr[1] = start_hr[1] + size_hr[1] - 1 ;
    end_hr[2] = start_hr[2] + size_hr[2] - 1 ;

    // Differential continuous indexes to perform the neighborhood iteration
    SpacingType spacing_lr = m_Images[0] -> GetSpacing();
    SpacingType spacing_hr = m_ReferenceImage -> GetSpacing();

    //spacing_lr[2] is assumed to be the lowest resolution
    //compute the index of the PSF in the LR image resolution
    std::vector<ContinuousIndexType> deltaIndexes;
    int npoints =  ceil(spacing_lr[2] / (2.0 * spacing_hr[2])) ;

    std::cout << "NPoints : " << 2*(npoints)+1 << std::endl;

    ContinuousIndexType delta;
    delta[0] = 0.0; delta[1] = 0.0;

    // // doubled PSF
    for (int i =( -npoints) ; i <= (npoints); i++ )
    {
        //FIXED
        delta[2] = ((double) i  * spacing_hr[2]) / spacing_lr[2];// - 0.3 /spacing_lr[2]; // Space of 1 between Index in the LR image corresponds to the slice thickness (3mm for instance).
        deltaIndexes.push_back(delta);
    }

    std::cout << "DeltaIndexes : " << deltaIndexes[0] << deltaIndexes[1] << deltaIndexes[2] << std::endl;

    // Set size of matrices
    unsigned int ncols = m_OutputImageRegion.GetNumberOfPixels();

    unsigned int nrows = 0;
    for(unsigned int im = 0; im < m_Images.size(); im++)
        nrows += m_Regions[im].GetNumberOfPixels();

    if(m_ComputeH)
    {
        std::cout << "Initialize H and Z ..." << std::endl;
        H.set_size(nrows, ncols);
        Z.set_size(nrows);
        Z.fill(0.0);
    }
    else{
        Z.set_size(nrows);
        Z.fill(0.0);
    }

    Y.set_size(nrows);
    Y.fill(0.0);

    std::cout << "Size of H :  #rows = " << H.rows() << ", #cols = "<<H.cols() << std::endl;

    int counterDebug = 0;
    unsigned int im;
    //#pragma omp parallel for private(im) schedule(dynamic)
    for(im = 0; im < m_Images.size(); im++)
    {
        // Interpolator for HR image
        InterpolatorPointer interpolator = InterpolatorType::New();
        interpolator -> SetInputImage( m_ReferenceImage );

        SpacingType inputSpacing = m_Images[im] -> GetSpacing();

        SpacingType inputSpacing2 = m_Images[im] -> GetSpacing();
        inputSpacing2[0] = inputSpacing2[0] ;
        inputSpacing2[1] = inputSpacing2[1] ;
        inputSpacing2[2] = inputSpacing2[2] ;

        std::cout << "input spacing 2 : " << inputSpacing2 << std::endl;

        // PSF definition
        typename FunctionType::Pointer function = FunctionType::New();
        function -> SetPSF(  FunctionType::GAUSSIAN );
        function -> SetDirection( m_Images[im] -> GetDirection() );
        function -> SetSpacing( inputSpacing2 );
        //function -> Print(std::cout);

        std::cout << "Image # "<< im << " sizes : " << m_Images[im]->GetLargestPossibleRegion().GetSize() << std::endl;
        std::cout << "Image # "<< im << " direction : " << m_Images[im]->GetDirection() << std::endl;
        std::cout << "Image # "<< im << " spacing : " << inputSpacing2 << std::endl;

        //Define the ROI of the current LR image
        IndexType inputIndex = m_Regions[im].GetIndex();
        SizeType  inputSize  = m_Regions[im].GetSize();

        //Define all indexes needed for iteration over the slices
        IndexType lrIndex;              //index of a point in the LR image im
        IndexType lrDiffIndex;          //index of this point in the current ROI of the LR image im
        unsigned int lrLinearIndex;     //index lineaire de ce point dans le vecteur

        IndexType hrIndex;
        IndexType hrDiffIndex;
        ContinuousIndexType hrContIndex;
        unsigned int hrLinearIndex;

        ContinuousIndexType nbIndex;

        PointType lrPoint;            //PSF center in world coordinates (PointType = worldcoordinate for ITK)
        PointType nbPoint;            //PSF point in world coordinate
        PointType transformedPoint;   //after applying current transform (registration)

        unsigned int offset = 0;
        for(unsigned int im2 = 0; im2 < im; im2++)
            offset += m_Regions[im2].GetNumberOfPixels();


        ConstIteratorType fixedIt(m_Images[im], m_Regions[im]);

        double lrValue;
        double hrValue;

        //Iteration over voxels of the 3D LR image
        for(fixedIt.GoToBegin(); !fixedIt.IsAtEnd(); ++fixedIt)
        {
            //Current index in the LR image
            lrIndex = fixedIt.GetIndex();

            //World coordinates of lrIndex using the image header
            m_Images[im] -> TransformIndexToPhysicalPoint( lrIndex, lrPoint );

            if ( m_Masks.size() > 0)
                if ( ! m_Masks[im] -> IsInside(lrPoint) )
                    continue;

            //Compute the coordinates in the SR using the estimated registration
            TransformType * trans = static_cast< TransformType * >(m_Transforms[im]);
            // trans = static_cast< TransformType * > m_Transforms[im];
            transformedPoint = trans -> TransformPoint( lrPoint );

            //check if this point is in the SR image (m_ReferenceImage)
            if ( ! interpolator -> IsInsideBuffer( transformedPoint ) )
                continue;

            //From the LR image coordinates to the LR ROI coordinates
            lrDiffIndex[0] = lrIndex[0] - inputIndex[0];
            lrDiffIndex[1] = lrIndex[1] - inputIndex[1];
            lrDiffIndex[2] = lrIndex[2] - inputIndex[2];

            //Compute the corresponding linear index of lrDiffIndex
            lrLinearIndex = lrDiffIndex[0] + lrDiffIndex[2]*inputSize[0] +
                    lrDiffIndex[1]*inputSize[0]*inputSize[2];

            //Get the intensity value in the LR image
            Y[lrLinearIndex + offset] = fixedIt.Get();

            //Set the center point of the PSF
            function -> SetCenter( lrPoint );
            //function -> Print(std::cout);

            //std::cout << "Loop over PSF points : " << deltaIndexes.size() << "points" << std::endl;
            for(unsigned int k=0; k<deltaIndexes.size(); k++)
            {
                //Coordinates in the LR image
                nbIndex[0] = deltaIndexes[k][0] + lrIndex[0];
                nbIndex[1] = deltaIndexes[k][1] + lrIndex[1];
                nbIndex[2] = deltaIndexes[k][2] + lrIndex[2];

                //World coordinates using LR image header
                m_Images[im] -> TransformContinuousIndexToPhysicalPoint( nbIndex, nbPoint );

                //Compute the PSF value at this point
                lrValue = function -> Evaluate(nbPoint);

                if ( lrValue > 0)
                {
                    //Compute the world coordinate of this point in the SR image
                    transformedPoint = nbPoint;
                    //transformedPoint = m_Transforms[im][i] -> TransformPoint( nbPoint );


                    //Set this coordinate in continuous index in SR image space
                    m_ReferenceImage -> TransformPhysicalPointToContinuousIndex(
                                transformedPoint, hrContIndex );

                    // OLD VERSION (BTK V1)

                    bool isInsideHR = true;

                    // FIXME This checking should be done for all points first, and
                    // discard the point if al least one point is out of the reference
                    // image

                    if ( (hrContIndex[0] < start_hr[0]) || (hrContIndex[0] > end_hr[0]) ||
                         (hrContIndex[1] < start_hr[1]) || (hrContIndex[1] > end_hr[1]) ||
                         (hrContIndex[2] < start_hr[2]) || (hrContIndex[2] > end_hr[2]) )
                        isInsideHR = false;

                    if ( isInsideHR )
                    {
                        //Compute the corresponding value in the SR image -> useless
                        //Allows to compute the set of contributing neighbors
                        hrValue = interpolator -> Evaluate( transformedPoint );

                        //std::cout << "Number of contributing neighbors for point " << transformedPoint << " : " << interpolator -> GetContributingNeighbors() << std::endl;

                        //Loop over points affected using the interpolation
                        for(unsigned int n=0; n<interpolator -> GetContributingNeighbors();
                            n++)
                        {
                            //Index in the SR image
                            hrIndex = interpolator -> GetIndex(n);

                            //Index in the ROI of the SR index
                            hrDiffIndex[0] = hrIndex[0] - start_hr[0];
                            hrDiffIndex[1] = hrIndex[1] - start_hr[1];
                            hrDiffIndex[2] = hrIndex[2] - start_hr[2];

                            //Compute the corresponding linear index
                            hrLinearIndex = hrDiffIndex[0] + hrDiffIndex[1]*size_hr[0] +
                                    hrDiffIndex[2]*size_hr[0]*size_hr[1];

                            //Add the correct value in H !
                            H(lrLinearIndex + offset, hrLinearIndex) += interpolator -> GetOverlap(n)* lrValue;
                            //H(lrLinearIndex + offset, hrLinearIndex) += interpolator -> GetOverlap(n)* 1.0;
                            //H(lrLinearIndex + offset, hrLinearIndex) +=  lrValue;
                            //H(lrLinearIndex + offset, hrLinearIndex) +=  1.0;


                        }

                    }// if psf point is inside sr image

                } //if psf value in not zero

            }//End of loop over voxels in PSF

        }//End of loop over voxels in LR images

    }//End of loop over LR images

    if(m_ComputeH)
    {

        std::cout << "H was computed and normalized ..." << std::endl << std::endl;
        // Normalize H
        for (unsigned int i = 0; i < H.rows(); i++)
        {
            double sum = H.sum_row(i);

            VnlSparseMatrixType::row & r = H.get_row(i);
            VnlSparseMatrixType::row::iterator col_iter = r.begin();

            for ( ;col_iter != r.end(); ++col_iter)
                (*col_iter).second = (*col_iter).second / sum;
        }

    }
    else
    {
        std::cout << "Initialization of H with old value" << std::endl << std::endl;
    }
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::ComputeTotalVariationTerms()
{
    int ncols = H.cols();

    //vnl_vector<float> DivP;
    DivP.set_size(ncols);
    DivP.fill(0.0);

    //Old implementation:  Precomputes A = deltat * lambda * tau * Ht * H
    // BUT Ht * H is computationally expensive (multiplication of two sparse matrix)
    // New implementation: Use only multiplication between sparse matrix and a vector which are much more efficient
    /*
    if(m_ComputeH)
    {
        HtH = H.transpose() * H;
    }

    A = deltat * lambda * tau * HtH;
    */

    std::cout << "Update P :" << std::endl;
    std::cout << "Old values : ";
    std::cout<<"Px="<<Px.sum()<<" , Py="<<Py.sum()<<" , Pz="<<Pz.sum()<<" , Xest="<<Xest.sum()<<" , Xold="<<Xold.sum()<<", DivP="<<DivP.sum()<<" , b="<<b.sum()<<std::endl<<std::endl;
    //vsl_print_summary(vcl_cout,A);
    // Converts and writes output image
    typename itk::ImageDuplicator<ImageType>::Pointer duplicator2 = itk::ImageDuplicator<ImageType>::New();
    duplicator2->SetInputImage(m_ReferenceImage);
    duplicator2->Update();
    ImagePointer outputIm2 = duplicator2->GetOutput();

    typename ImageType::RegionType outputImageRegion2 = outputIm2 -> GetLargestPossibleRegion();

    itk::ImageRegionIterator<ImageType> outputIt2( outputIm2,outputImageRegion2 );
    unsigned int linearIndex2 = 0;
    for (outputIt2.GoToBegin(); !outputIt2.IsAtEnd(); ++outputIt2, linearIndex2++)
        outputIt2.Set(Xest[linearIndex2]);

    /*
  typename itk::ImageFileWriter< ImageType >::Pointer writer3 =  itk::ImageFileWriter< ImageType >::New();
  writer3 -> SetFileName( "/Users/sebastientourbier/Desktop/Patient01/SR/Manual/pre_xest.nii.gz" );
  writer3 -> SetInput( outputIm2 );
  writer3 -> Update();
  */

    //Precomputes splitting variable P as it is constant over the optimization
    // Creates backward and forward derivative kernels
    float* fKernel = new float[3];
    fKernel[0] = 0; fKernel[1] = -1; fKernel[2] = 1;

    float* bKernel = new float[3];
    bKernel[0] = -1; bKernel[1] = 1; bKernel[2] = 0;

    //Computes P buggy!!! Get & set Px to old value
    vnl_vector<float> DfxXest;
    DfxXest.set_size( ncols );
    convol3dx(Xest, DfxXest, x_size, fKernel, 3);
    Px = Px + sigma * DfxXest;
    //DfxXest.clear();

    vnl_vector<float> DfyXest;
    DfyXest.set_size( ncols );
    convol3dy(Xest, DfyXest, x_size, fKernel, 3);
    Py = Py + sigma * DfyXest;
    //DfyXest.clear();

    vnl_vector<float> DfzXest;
    DfzXest.set_size( ncols );
    convol3dz(Xest, DfzXest, x_size, fKernel, 3);
    Pz = Pz + sigma * DfzXest;
    //DfzXest.clear();

    // Converts and writes output image
    typename itk::ImageDuplicator<ImageType>::Pointer duplicator = itk::ImageDuplicator<ImageType>::New();
    duplicator->SetInputImage(m_ReferenceImage);
    duplicator->Update();
    ImagePointer outputIm = duplicator->GetOutput();

    typename ImageType::RegionType outputImageRegion = outputIm -> GetLargestPossibleRegion();

    itk::ImageRegionIterator<ImageType> outputIt( outputIm,outputImageRegion );
    unsigned int linearIndex = 0;
    for (outputIt.GoToBegin(); !outputIt.IsAtEnd(); ++outputIt, linearIndex++)
        outputIt.Set(DfxXest[linearIndex]);

    /*
  typename itk::ImageFileWriter< ImageType >::Pointer writer2 =  itk::ImageFileWriter< ImageType >::New();
  writer2 -> SetFileName( "/Users/sebastientourbier/Desktop/Patient01/SR/Manual/dfxxest.nii.gz" );
  writer2 -> SetInput( outputIm );
  writer2 -> Update();
  */

    DfxXest.clear();
    DfyXest.clear();
    DfzXest.clear();

    vnl_vector<double> dNormP = vnl_matops::f2d( element_product(Px,Px) + element_product(Py,Py) + element_product(Pz,Pz)) ;
    dNormP = dNormP.apply(sqrt);

    vnl_vector<float> NormP = vnl_matops::d2f(dNormP);
    dNormP.clear();

    //Normalizes P
    for(int i = 0; i < ncols ; i++)
    {
        if(NormP[i]>1)
        {
            //std::cout << "NormP[" << i << "] = " << NormP[i] << std::endl;
            Px[i] = Px[i] / NormP[i];
            Py[i] = Py[i] / NormP[i];
            Pz[i] = Pz[i] / NormP[i];
        }
    }
    NormP.clear();

    //std::cout << "Debug flag 6" << std::endl;

    //Computes DivP
    vnl_vector<float> DbxPx;
    DbxPx.set_size( ncols );
    convol3dx(Px, DbxPx, x_size, bKernel, 3);

    vnl_vector<float> DbyPy;
    DbyPy.set_size( ncols );
    convol3dy(Py, DbyPy, x_size, bKernel, 3);

    vnl_vector<float> DbzPz;
    DbzPz.set_size( ncols );
    convol3dz(Pz, DbzPz, x_size, bKernel, 3);

    //DivP = - (DbxPx + DbyPy + DbzPz);
    //vnl_vector<float> DivP;
    //DivP.set_size( ncols );
    DivP = DbxPx + DbyPy + DbzPz;

    // Precalcule Ht*Y. Note that this is calculated as Y*H since
    // Ht*Y = (Yt*H)t and for Vnl (Yt*H)t = (Yt*H) = Y*H because
    // the vnl_vector doesn't have a 2nd dimension. This allows us
    // to save a lot of memory because we don't need to store Ht.
    vnl_vector<float> HtY;
    HtY.set_size(ncols);
    H.pre_mult(Y,HtY);

    vnl_vector<float> HtZ;
    HtZ.set_size(ncols);
    H.pre_mult(Z,HtZ);

    //Precompute b as it is constant over optimization iteration. Used in update()
    b = deltat * lambda * tau * ( HtY + HtZ ) - deltat * tau * DivP;

    DbxPx.clear();
    DbyPy.clear();
    DbzPz.clear();

    std::cout << "New values : ";
    std::cout<<"Px="<<Px.sum()<<" , Py="<<Py.sum()<<" , Pz="<<Pz.sum()<<" , Xest="<<Xest.sum()<<" , Xold="<<Xold.sum()<<", DivP="<<DivP.sum()<<" , b="<<b.sum()<<std::endl<<std::endl;
    //vsl_print_summary(vcl_cout,A);
    HtY.clear();
    //DivP.clear();

    delete[] bKernel;
    delete[] fKernel;
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetLambda(float value)
{
    lambda = value;
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetGamma(float value)
{
    gamma = value;
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetSigma(float value)
{
    sigma = value;
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetTau(float value)
{
    tau = value;
}


template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetTheta(float value)
{
    theta = value;
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetDeltat(float value)
{
    deltat = value;
}


template <class TImage>
float
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::GetTheta()
{
    float value = 0.0;
    value = 1 / sqrt( 1 + 2 * gamma * tau);
    return value;
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetP(const vnl_vector<float>& px,const vnl_vector<float>& py,const vnl_vector<float>& pz)
{
    Px = px;
    Py = py;
    Pz = pz;
}



template <class TImage>
vnl_vector<float> 
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::GetPx()
{
    return Px;
}

template <class TImage>
vnl_vector<float> 
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::GetPy()
{
    return Py;
}

template <class TImage>
vnl_vector<float> 
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::GetPz()
{
    return Pz;
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::AddImage( ImageType* image )
{
    m_Images.push_back( image );

    // Add transforms for this image
    m_Transforms.resize( m_Transforms.size() + 1 );
    SizeType imageSize = image -> GetLargestPossibleRegion().GetSize();
    m_Transforms[m_Transforms.size()-1].resize( imageSize[2]);

    // Initialize transforms
    //for (unsigned int i=0; i<imageSize[2]; i++)
    //   m_Transforms[m_Transforms.size()-1][i] = TransformType::New();

}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::AddRegion( RegionType region)
{
    m_Regions.push_back( region );
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::AddMask( MaskType *mask)
{
    m_Masks.push_back( mask );
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetReferenceImage( const ImageType * image )
{
    m_ReferenceImage = image;
}

template <class TImage>
void
TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetTransform( int i, int j, TransformType* transform )
{
    m_Transforms[i][j] = transform;
}
template <class TImage>
void TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetHMatrix(const vnl_sparse_matrix<float>& m)
{
    H = m;
}
template <class TImage>
vnl_sparse_matrix<float> TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::GetHMatrix()
{
    return H;
}
template <class TImage>
void TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetHtHMatrix(const vnl_sparse_matrix<float>& m)
{
    HtH = m;
}
template <class TImage>
vnl_sparse_matrix<float> TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::GetHtHMatrix()
{
    return HtH;
}
template <class TImage>
void TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetAMatrix(const vnl_sparse_matrix<float>& m)
{
    A = m;
}
template <class TImage>
vnl_sparse_matrix<float> TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::GetAMatrix()
{
    return A;
}
template <class TImage>
void TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetComputeH(bool value)
{
    m_ComputeH = value;
}
template <class TImage>
void TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetZVector(vnl_vector<float>& m)
{
    Z = m;
}
template <class TImage>
vnl_vector<float> TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::GetZVector()
{
    return Z;
}
template <class TImage>
vnl_vector<float> TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::GetObservationsY()
{
    return Y;
}
template <class TImage>
void TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetSliceGap(double gap)
{
    m_SliceGap = gap;
}
template <class TImage>
double TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::GetSliceGap()
{
    return m_SliceGap;
}
template <class TImage>
void TotalVariationCostFunctionWithImplicitGradientDescent<TImage>::SetUseDebluringPSF(bool value)
{
    m_UseDebluringPSF = value;
}

} // namespace mialsrtk

#endif /* TotalVariationCostFunctionWithImplicitGradientDescent_txx */
