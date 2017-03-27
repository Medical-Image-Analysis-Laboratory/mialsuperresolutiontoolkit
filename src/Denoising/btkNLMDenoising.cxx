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
This program implements a denoising method proposed by Coupé et al. described in :
 Coupé, P., Yger, P., Prima, S., Hellier, P., Kervrann, C., Barillot, C., 2008. 
 An optimized blockwise nonlocal means denoising filter for 3-D magnetic resonance images.
 IEEE Transactions on Medical Imaging 27 (4), 425–441.
*/



#include <tclap/CmdLine.h>

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkImage.h"
#include "itkConstrainedValueDifferenceImageFilter.h"

#include "btkNLMTool.h"

#include <vector>

int main(int argc, char** argv)
{

  try {

    TCLAP::CmdLine cmd("Non-Local mean denoising: implementation of the method proposed by Coupé et al., IEEE TMI 2008 ", ' ', "1.0", true);

    TCLAP::ValueArg<std::string> inputImageArg("i","image_file","input image file (short)",true,"","string");
    cmd.add( inputImageArg );
    TCLAP::ValueArg<std::string> outputImageArg("o","output_file","output image file (short)",true,"","string");
    cmd.add( outputImageArg );
    TCLAP::ValueArg<std::string> inputMaskArg("m","mask_file","filename of the mask image",false,"","string");
    cmd.add( inputMaskArg );
    TCLAP::ValueArg<std::string> inputReferenceArg("r","ref_file","filename of the reference image",false,"","string");
    cmd.add( inputReferenceArg );
    TCLAP::ValueArg< float > paddingArg("p","pad","padding value (used if no mask image is provided, default is 0)",false,0,"float");
    cmd.add( paddingArg );
    TCLAP::ValueArg< int > hwnArg("","hwn","patch half size (default is 1)",false,1,"int");
    cmd.add( hwnArg );
    TCLAP::ValueArg< int > hwvsArg("","hwvs","half size of the volume search area, i.e. the spatial bandwidth (default is 5)",false,5,"int");
    cmd.add( hwvsArg );
    TCLAP::ValueArg< float > betaArg("b","beta","beta: smoothing parameter (high beta produces smoother result, default is 1)",false,1,"float");
    cmd.add( betaArg );
    TCLAP::ValueArg< int > blockArg("","block","0: pointwise, 1: blockwise, 2: fast blockwise (default is 1)",false,1,"int");
    cmd.add( blockArg );
    TCLAP::ValueArg< int > centerArg("c","center","weight of the central patch (possible value: 0, 1, -1 (max)) (default is -1)",false,-1,"int");
    cmd.add( centerArg );
    TCLAP::ValueArg< int > optimizedArg("","opt","optimized mode (use mean and standard deviation of patches) (0: no, 1: yes) (default is 1)",false,1,"int");
    cmd.add( optimizedArg );
    TCLAP::ValueArg< float > lowerMeanThresholdArg("","lmt","lower mean threshold (0.95 by default) -- for optimized mode only",false,0.95,"float");
    cmd.add( lowerMeanThresholdArg );
    TCLAP::ValueArg< float > lowerVarianceThresholdArg("","lvt","lower variance threshold (0.5 by default) -- for optimized mode only",false,0.5,"float");
    cmd.add( lowerVarianceThresholdArg );
    TCLAP::ValueArg<std::string> outputDifferenceImageArg("d","difference_file","filename of the difference image",false,"","string");
    cmd.add( outputDifferenceImageArg );
    TCLAP::ValueArg< int > localArg("","local","Estimation of the smoothing parameter. 0: global, 1: local (default is 0)",false,0,"int");
    cmd.add( localArg );
    
 
    // Parse the args.
    cmd.parse( argc, argv );


    // Get the value parsed by each arg.
    std::string input_file       = inputImageArg.getValue();
    std::string output_file      = outputImageArg.getValue();

    std::string mask_file        = inputMaskArg.getValue();
    std::string ref_file         = inputReferenceArg.getValue();
    float padding                = paddingArg.getValue();
    int hwn                      = hwnArg.getValue();
    int hwvs                     = hwvsArg.getValue();
    float beta                   = betaArg.getValue();
    int block                    = blockArg.getValue();
    int center                   = centerArg.getValue();
    int optimized                = optimizedArg.getValue();
    float lowerMeanThreshold     = lowerMeanThresholdArg.getValue();
    float lowerVarianceThreshold = lowerVarianceThresholdArg.getValue();
    std::string difference_file  = outputDifferenceImageArg.getValue();
    int localSmoothing           = localArg.getValue();

    //ITK declaration
    typedef float PixelType;
    const   unsigned int        Dimension = 3;
    typedef itk::Image< PixelType, Dimension >    ImageType; //same type for input and output
    typedef ImageType::Pointer ImagePointer;

    typedef itk::ImageFileReader< ImageType >  ReaderType;
    typedef itk::ImageFileWriter< ImageType >  WriterType;

    //Read the image
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( input_file );
    reader->Update();
    ImagePointer inputImage = reader->GetOutput();

    ImagePointer outputImage = ImageType::New();

    ImagePointer maskImage;
    ImagePointer refImage;

    btk::NLMTool<PixelType> myTool;

    myTool.SetInput(inputImage);

    if (mask_file != ""){               //reading the mask image
      ReaderType::Pointer maskReader = ReaderType::New();
      maskReader->SetFileName( mask_file );
      maskReader->Update();
      maskImage = maskReader->GetOutput();
      myTool.SetMaskImage(maskImage);
    }
    else                                 //creating a mask image using the padding value
      myTool.SetPaddingValue(padding);

    myTool.SetPatchSize(hwn);
    myTool.SetSpatialBandwidth(hwvs);
    
    if (ref_file != ""){
      ReaderType::Pointer refReader = ReaderType::New();
      refReader->SetFileName( ref_file );
      refReader->Update();
      refImage = refReader->GetOutput();
      myTool.SetReferenceImage(refImage);    
    }    
    
    myTool.SetCentralPointStrategy(center);
    myTool.SetBlockwiseStrategy(block);
    myTool.SetOptimizationStrategy(optimized);
    myTool.SetLowerThresholds(lowerMeanThreshold, lowerVarianceThreshold);

    myTool.SetSmoothing(beta);
    if(localSmoothing == 1)
      myTool.SetLocalSmoothing(beta);

    
    myTool.ComputeOutput();

    outputImage = myTool.GetOutput();

    //Write the result
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName( output_file );
    writer->SetInput( outputImage );
    writer->Update();

    if (difference_file != ""){
      itk::ConstrainedValueDifferenceImageFilter<ImageType,ImageType,ImageType>::Pointer diffFilter = itk::ConstrainedValueDifferenceImageFilter<ImageType,ImageType,ImageType>::New();

      diffFilter->SetInput1( inputImage );
      diffFilter->SetInput2( outputImage );

      WriterType::Pointer diffWriter = WriterType::New();
      diffWriter->SetFileName( difference_file );
      diffWriter->SetInput( diffFilter->GetOutput() );
      diffWriter->Update();
    }



    return 1;

  } catch (TCLAP::ArgException &e)  // catch any exceptions
  { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

}
