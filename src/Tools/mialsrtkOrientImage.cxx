#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkOrientImageFilter.h>

#include <tclap/CmdLine.h>

int main( int argc, char ** argv )
{

  std::string inputFile;
  std::string outputFile;
  std::string orientation;
  try
    {
      TCLAP::CmdLine cmd("Reorient an image", ' ', "Unversioned");
      TCLAP::ValueArg<std::string> inputArg("i","input","Input Image File",true,"","string",cmd);
      TCLAP::ValueArg<std::string> outputArg("o","output","Output Image File",true,"","string",cmd);
      TCLAP::ValueArg<std::string> orientationArg("O","orientation","Output orientation: axial (default), sagittal, coronal, or RIP",false,"axial","string",cmd);

      cmd.parse(argc,argv);

      if (inputArg.isSet()) inputFile = inputArg.getValue();
      if (outputArg.isSet()) outputFile = outputArg.getValue();
      if (orientationArg.isSet()) orientation = orientationArg.getValue();
    }
  catch (TCLAP::ArgException& e)
    {
      std::cerr << "error: " << e.error() << "for argument " << e.argId() << std::endl;
      exit(-1);
    }


  typedef float PixelType;
  typedef itk::Image< PixelType, 3 >  ImageType;
  typedef itk::ImageFileReader< ImageType > ReaderType;

  std::cout << "Loading input image " << inputFile.c_str() << std::endl;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( inputFile.c_str() );
  try
    {
    reader->Update();
    }
  catch ( itk::ExceptionObject &err)
    {
    std::cout << "ExceptionObject caught !" << std::endl;
    std::cout << err << std::endl;
    return -1;
    }

  typedef itk::OrientImageFilter<ImageType,ImageType> OrientImageFilterType;
  OrientImageFilterType::Pointer orienter = OrientImageFilterType::New();

  orienter->UseImageDirectionOn();

  if (orientation == "axial") {
    std::cout << "Orientation set to axial" << std::endl;
    orienter->SetDesiredCoordinateOrientationToAxial();
  } else if (orientation == "sagittal") {
    std::cout << "Orientation set to sagittal" << std::endl;
    orienter->SetDesiredCoordinateOrientationToSagittal();
  } else if (orientation == "coronal") {
    std::cout << "Orientation set to coronal" << std::endl;
    orienter->SetDesiredCoordinateOrientationToCoronal();
  } else if (orientation == "RIP") {
    std::cout << "Orientation set to RIP" << std::endl;
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP);
  } else {
    std::cerr << "Invalid orientation : " << orientation.c_str() << std::endl;
    std::cerr << "Possible choices : axial sagittal coronal RIP" << std::endl;
    exit(1);
  }

  orienter->SetInput(reader->GetOutput());
  orienter->Update();

  std::cout << "Writing output image " << outputFile.c_str() << std::endl;

  typedef itk::ImageFileWriter< ImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( outputFile.c_str() );

  try
    {
    writer->SetInput(orienter->GetOutput());
    writer->UseCompressionOn( );
    writer->Update();
    }
  catch ( itk::ExceptionObject &err)
    {
    std::cout << "ExceptionObject caught !" << std::endl;
    std::cout << err << std::endl;
    return -1;
    }

  return 0;

}
