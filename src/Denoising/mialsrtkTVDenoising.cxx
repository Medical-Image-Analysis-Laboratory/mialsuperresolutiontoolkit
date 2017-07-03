/*==========================================================================
  © 
  Date: 01/05/2015
  Author(s): Sebastien Tourbier (sebastien.tourbier@unil.ch)
==========================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

/* Standard includes */
#include <tclap/CmdLine.h>

/* Itk includes */
#include "itkImage.h"
#include "itkImageRegionIterator.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageMaskSpatialObject.h"
#include "itkCastImageFilter.h"
#include "itkImageDuplicator.h"

#include "vnl/vnl_matops.h"
#include <vnl/vnl_random.h>

struct img_size {
   unsigned int width;
   unsigned int height;
   unsigned int depth;
};

void set(float * array, int size, float value)
{
  for (int i = 0; i < size; i++)
  array[i] = value;
}

// Mirror of the position pos. abs(pos) must not be > 2*(size-1)
int mirror(int pos, int size)
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

void get_row(const vnl_vector<float>& image,
    img_size & size, int row, int frame, float * output)
{
  for (unsigned int i = 0; i < size.width; i++)
  output[i] = image[i + row * size.width + frame * size.width * size.height];
}

void set_row(vnl_vector<float>& image,
    img_size & size, int row, int frame, float * input)
{
  for (unsigned int i = 0; i < size.width; i++)
    image[i + row * size.width + frame * size.width * size.height] = input[i];
}

void get_col(const vnl_vector<float>& image,
    img_size & size, int col, int frame, float * output)
{
  for (unsigned int i = 0; i < size.height; i++)
    output[i] = image[col + i * size.width + frame * size.width * size.height];
}

void set_col(vnl_vector<float>& image,
    img_size & size, int col, int frame, float * input)
{
  for (unsigned int i = 0; i < size.height; i++)
    image[col + i * size.width + frame * size.width * size.height] = input[i];
}

void get_spec(const vnl_vector<float>& image,
    img_size & size, int row, int col, float * output)
{
  for (unsigned int i = 0; i < size.depth; i++)
  output[i] = image[col + row * size.width + i * size.width * size.height];
}

void set_spec(vnl_vector<float>& image,
    img_size & size, int row, int col, float * input)
{
  for (unsigned int i = 0; i < size.depth; i++)
    image[col + row * size.width + i * size.width * size.height] = input[i];
}

void convol1d(float * kernel, int ksize,
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
void convol3dx(const vnl_vector<float>& image,
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
void convol3dy(const vnl_vector<float>& image,
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
void convol3dz(const vnl_vector<float>& image,
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


int main( int argc, char *argv[] )
{

  try {

  const char *input = NULL;
  const char *outImage = NULL;
  const char *refImage = NULL;

  std::vector< int > x1, y1, z1, x2, y2, z2;

  float lambda;
  float normD = 12.0;
  float theta = 1.0;
  float gamma = 1.0;
  float tau = 1 / sqrt (12.0);
  float sigma = 1 / sqrt(12.0);
  float threshold = 1e-4;

  float stepscale = 0.0;

  // Parse arguments

  TCLAP::CmdLine cmd("Apply super-resolution algorithm using one or multiple input images.", ' ', "Unversioned");

  TCLAP::ValueArg<std::string> inputArg("i","input","Low-resolution image file",true,"","string",cmd);
  TCLAP::ValueArg<std::string> outArg  ("o","output","Super resolution output image",true,"","string",cmd);
  TCLAP::ValueArg<float> lambdaArg  ("","lambda","Regularization factor (default = 0.1)",false, 0.1,"float",cmd);
  TCLAP::ValueArg<float> gammaArg  ("","gamma","Gamma parameter (default = 1)",false, 1,"float",cmd);
  TCLAP::ValueArg<float> scaleArg  ("","step-scale","Step scale parameter (default = 1)",false, 1,"float",cmd);
  TCLAP::ValueArg<float> convArg  ("","cthresh","Convergence threshold (default = 1e-4)",false, 0.0001,"float",cmd);
  TCLAP::ValueArg<int> loopArg  ("","loop","Number of loops (denoising) (default = 5)",false, 5,"int",cmd);
    

  // Parse the argv array.
  cmd.parse( argc, argv );

  input = inputArg.getValue().c_str();
  outImage = outArg.getValue().c_str();
  lambda = lambdaArg.getValue();
  gamma = gammaArg.getValue();
  threshold = convArg.getValue();

  stepscale = scaleArg.getValue();

  tau = (stepscale) * tau;
  sigma = (1.0 / stepscale) * sigma;
  
  int numberOfLoops = loopArg.getValue();

  // typedefs
  const   unsigned int    Dimension = 3;

  typedef float  PixelType;

  typedef itk::Image< PixelType, Dimension >  ImageType;
  typedef ImageType::Pointer                  ImagePointer;
  typedef std::vector<ImagePointer>           ImagePointerArray;

  typedef itk::Image< unsigned char, Dimension >  ImageMaskType;
  typedef itk::ImageFileReader< ImageMaskType > MaskReaderType;
  typedef itk::ImageMaskSpatialObject< Dimension > MaskType;

  typedef ImageType::SizeType    SizeType;

  typedef ImageType::RegionType               RegionType;
  typedef std::vector< RegionType >           RegionArrayType;

  typedef itk::ImageRegionConstIteratorWithIndex< ImageType > IteratorType;

  typedef itk::ImageFileReader< ImageType >   ImageReaderType;
  typedef itk::ImageFileWriter< ImageType >   WriterType;

  // Filter setup
  ImageType::IndexType  roiIndex;
  ImageType::SizeType   roiSize;

  // Set reference image
  std::cout<<"Reading the input image : "<<input<<std::endl;
  ImageReaderType::Pointer inputReader = ImageReaderType::New();
  inputReader -> SetFileName( input );
  inputReader -> Update();

  ImagePointer inputIm = inputReader -> GetOutput();

  RegionType inputImageRegion = inputIm -> GetLargestPossibleRegion();
  unsigned int nelements = inputImageRegion.GetNumberOfPixels();

  SizeType  size_input   = inputImageRegion.GetSize();

  //x_size : size of the input image 
  img_size x_size;
  x_size.width  = size_input[0];
  x_size.height = size_input[1];
  x_size.depth  = size_input[2];

  //Vectorize the input image
  vnl_vector<float> x;
  x.set_size( nelements );
  IteratorType inputIt( inputIm,inputImageRegion );
  unsigned int linearIndex = 0;

  for (inputIt.GoToBegin(); !inputIt.IsAtEnd(); ++inputIt, linearIndex++)
    x[linearIndex] = inputIt.Get();	

  
  //float dynRange = x.max_value() - x.min_value();

  // Add gaussian noise to image
  /*
  float noiselevel = 0.5;
  vnl_vector<float> noise;
  noise.set_size( nelements );
  for (int index = 0 ; index<nelements ; index++)
	noise[index] = (std::rand()/RAND_MAX) * noiselevel * dynRange;

  x = x + noise;
  */
  
  //Initializes variables
  vnl_vector<float> y;
  y.set_size( nelements );
  y = x;

  vnl_vector<float> x_est;
  x_est.set_size( nelements );
  x_est = x;

  vnl_vector<float> x_old;
  x_old.set_size( nelements );
  x_old.fill(0.0);

  vnl_vector<float> Px;
  Px.set_size( nelements );
  Px.fill(0.0);

  vnl_vector<float> Py;
  Py.set_size( nelements );
  Py.fill(0.0);

  vnl_vector<float> Pz;
  Pz.set_size( nelements );
  Pz.fill(0.0);

  vnl_vector<float> DivP;
  DivP.set_size( nelements );
  DivP.fill(0.0);

  float criterion = RAND_MAX;

  for (int i=0; i<numberOfLoops; i++){

    // Creates backward and forward derivative kernels
  	float* fKernel = new float[3];
  	fKernel[0] = 0; fKernel[1] = -1; fKernel[2] = 1;

  	float* bKernel = new float[3];
  	bKernel[0] = -1; bKernel[1] = 1; bKernel[2] = 0;

  	//Computes P
  	vnl_vector<float> DfxXest;
  	DfxXest.set_size( nelements );
  	convol3dx(x_est, DfxXest, x_size, fKernel, 3);
  	Px = Px + sigma * DfxXest;
  	DfxXest.clear();

  	vnl_vector<float> DfyXest;
  	DfyXest.set_size( nelements );
  	convol3dy(x_est, DfyXest, x_size, fKernel, 3);
  	Py = Py + sigma * DfyXest;
  	DfyXest.clear();

  	vnl_vector<float> DfzXest;
  	DfzXest.set_size( nelements );
  	convol3dz(x_est, DfzXest, x_size, fKernel, 3);
  	Pz = Pz + sigma * DfzXest;
  	DfzXest.clear();

  	vnl_vector<double> dNormP = vnl_matops::f2d( element_product(Px,Px) + element_product(Py,Py) + element_product(Pz,Pz)) ;
  	dNormP = dNormP.apply(sqrt);
  	vnl_vector<float> NormP = vnl_matops::d2f(dNormP);
  	dNormP.clear();

  	//Normalizes P
  	for(int j = 0; j < nelements ; j++)
  	{
    	if(NormP[j]>1)
    	{
      	Px[j] = Px[j] / NormP[j];
      	Py[j] = Py[j] / NormP[j];
      	Pz[j] = Pz[j] / NormP[j];
    	}
  	}
  	NormP.clear();

  	//Computes DivP 
  	vnl_vector<float> DbxPx;
  	DbxPx.set_size( nelements );
  	convol3dx(Px, DbxPx, x_size, bKernel, 3);
  
  	vnl_vector<float> DbyPy;
  	DbyPy.set_size( nelements );
  	convol3dy(Py, DbyPy, x_size, bKernel, 3);
  
  	vnl_vector<float> DbzPz;
  	DbzPz.set_size( nelements );
  	convol3dz(Pz, DbzPz, x_size, bKernel, 3);  

  	DivP = DbxPx + DbyPy + DbzPz;

  	DbxPx.clear();
  	DbyPy.clear();
  	DbzPz.clear();

  	delete[] bKernel;
  	delete[] fKernel;

    std::cout<<"Loop "<<i<<" : "<< std::endl; 
    std::cout<<"theta = "<<theta<<" , tau = "<<tau<<" , sigma = "<<sigma<<" , gamma = "<<gamma<<" , sumDivP = "<<DivP.sum()<<std::endl;
	std::cout<<"sumY = "<<y.sum()<<" , sumXold = "<<x_old.sum()<<" , sumXest = "<<x_est.sum()<<" , sumX = "<<x.sum()<<std::endl;

    //std::cout<<"Loop : "<< i <<" with theta = "<<theta<<" , tau = "<<tau<<" , sigma = "<<sigma<<" , sumDivP = "<<DivP.sum()<<" , sumY = "<<y.sum()<<" , sumXold = "<<x_old.sum()<<" , sumXest = "<<x_est.sum()<<" , sumX = "<<x.sum()<<std::endl; 

	//Store x(n)
    x_old = x;

    //Computes x(n+1)
    float tauXlambda = tau*lambda;
    x = (1 / ( 1 + tauXlambda )) * (x_old + tauXlambda*y + tau*DivP );
    
	//Update theta(n), tau(n+1), sigma(n+1)
    theta = (1 / sqrt( 1 + 2*gamma*tau ));
    sigma = sigma / theta;
    tau = theta * tau;

	//Estimates x(n+2)
    vnl_vector<float> diffX = (x -x_old); 
    x_est = x + theta * diffX;

    //Computes and verifies convergence criterion and the energy associated
    float criterion = diffX.squared_magnitude() / x_old.squared_magnitude();
    double energy = 0.5 * diffX.squared_magnitude() - dot_product(x,DivP);
    DivP.clear();
    
	std::cout<<"Criterion  : "<<criterion<<" ( Energy = "<<energy<<" )"<<std::endl;
    std::cout<<"-------------------------------------------------------------------------------"<<std::endl;
	if(criterion < threshold)
	  {
        std::cout<<"Energy has converged after "<<i<<" iterations , Criterion value = "<<criterion<<" ( Energy = "<<energy<<" )"<<std::endl;
		break;
      }
  }

  // Converts and writes output image
  typedef itk::ImageDuplicator< ImageType > DuplicatorType;
  DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage(inputIm);
  duplicator->Update();
  ImagePointer outputIm = duplicator->GetOutput();

  RegionType outputImageRegion = outputIm -> GetLargestPossibleRegion();

  itk::ImageRegionIterator<ImageType> outputIt( outputIm,outputImageRegion );
  linearIndex = 0;
  for (outputIt.GoToBegin(); !outputIt.IsAtEnd(); ++outputIt, linearIndex++)
	outputIt.Set(x[linearIndex]);	

  WriterType::Pointer writer =  WriterType::New();
  writer -> SetFileName( outImage );
  writer -> SetInput( outputIm );

  if ( strcmp(outImage,"") != 0)
  {
    std::cout << "Writing " << outImage << " ... ";
    writer->Update();
    std::cout << "done." << std::endl;
  }

  } catch (TCLAP::ArgException &e)  // catch any exceptions
  { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

  return EXIT_SUCCESS;
}

