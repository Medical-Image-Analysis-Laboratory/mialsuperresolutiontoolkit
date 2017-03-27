/*=========================================================================

Program: Combine multiple masks in one unique mask (using: majoriting voting, global weighting or, local weighting)
Language: C++
Date: $Date: 2013-08-22 $
Version: $Revision: 1.4 $
Author: $Sebastien Tourbier$

==========================================================================*/
#include "mialsrtkSegmentationWeightedFusion.h"


void prompt_start(std::vector< std::string > & inputFileNames, std::vector< std::string > & inputRegFileNames, std::vector< std::string > & maskFileNames, const char* outputFileName, unsigned int & method)
{
  std::cout << std::endl << "----------------------------------------------------------------"<<std::endl;
  std::cout << " Multi-Atlas Segmentation Refinement Program " << std::endl;
  std::cout << "----------------------------------------------------------------"<<std::endl<<std::endl;
  std::cout << std::endl << "Fusion method : ";

  if(method==0)
  {
    std::cout << "Majority Voting" << std::endl << std::endl;
  }
  else if(method==1)
  {
    std::cout << "Global weighting (MSD or NC)" << std::endl << std::endl;
  }
  else if(method==2)
  {
    std::cout << "Local weighting (MSD or NC)" << std::endl << std::endl;
  }

  unsigned int numberOfImages = inputFileNames.size();

  for(unsigned int i=0; i < numberOfImages; i++)
  {
    std::cout << "Inputs for atlas " << int2str(i) << ":" << std::endl;
    std::cout << "Input image: "<< inputFileNames[i] << std::endl;
    std::cout << "Input SDI image (reg.): "<< inputRegFileNames[i]<<std::endl;
    std::cout << "Input mask: "<< maskFileNames[i]<<std::endl << std::endl;
  }
  std::cout << "Output mask: " << outputFileName << std::endl;
};

int main( int argc, char * argv [] )
{

  std::vector< std::string > inputFileNames;
  std::vector< std::string > inputRegFileNames;
  std::vector< std::string > maskFileNames;

  const char *outputFileName = NULL;

  const char *outputCSVFileName = NULL;

  unsigned int method;
  unsigned int localPatchRadius;

  // Parse arguments

  TCLAP::CmdLine cmd("Fusion multiple masks into one", ' ', "Unversioned");

  TCLAP::MultiArg<std::string> inputArg("i","input","Low-resolution image file",true,"string",cmd);
  TCLAP::MultiArg<std::string> inputRegArg("r","input-reg","Registered low-resolution image file",true,"string",cmd);
  TCLAP::MultiArg<std::string> maskArg("m","mask","Low-resolution mask file",true,"string",cmd);
  TCLAP::ValueArg<std::string> outArg  ("o","output","Output mask file",true,"","string",cmd);
  TCLAP::ValueArg<int> methodArg  ("f","fusion-method","Fusion method (0: Majority voting (by default), 1: Global weighted voting, 2: Local weighted voting)",false, 0,"int",cmd);
  TCLAP::ValueArg<int> radiusArg  ("p","patch-radius","Patch radius used by the local weighted voting fusion method (radius = 1 by default)",false, 1,"int",cmd);

  TCLAP::SwitchArg itkNCCArg("","use-itk-ncc", "Use itk NCC as weight (if flag not set, use the patch-based NCC)", cmd, false );

  TCLAP::ValueArg<std::string> timeProfilingArg ("t","profiling-csv","Output csv file for time profiling",false,"undefined","string",cmd);

  // Parse the argv array.
  cmd.parse( argc, argv );

  inputFileNames = inputArg.getValue();
  inputRegFileNames = inputRegArg.getValue();
  maskFileNames = maskArg.getValue();
  outputFileName = outArg.getValue().c_str();
  method = methodArg.getValue();
  localPatchRadius = radiusArg.getValue();//(2r+1)*(2r+1)*(2r+1) is the volume of local patch

  outputCSVFileName = timeProfilingArg.getValue().c_str();

  if((method!=0) && (method!=1) && (method!=2))
  {
    std::cerr << "Error: Invalid value for method. Valid values = {0,1,2}" << std::endl;
    return EXIT_FAILURE;
  }

  if(maskFileNames.size() != inputFileNames.size() && inputRegFileNames.size() != inputFileNames.size())
  {
    std::cerr << "Error: Number of input images registered and masks must be equal to the number of input images." << std::endl;
    return EXIT_FAILURE;
  }

  prompt_start(inputFileNames,inputRegFileNames,maskFileNames,outputFileName,method);

  std::cout << std::endl << "Start multi-atlas segmentation fusion... \n" << std::endl;

  unsigned int numberOfImages = inputFileNames.size();

  std::vector< ReaderType::Pointer > inputReaders(numberOfImages);
  std::vector< InputImageType::Pointer > inputImages(numberOfImages);

  std::vector< ReaderType::Pointer > inputRegReaders(numberOfImages);
  std::vector< InputImageType::Pointer > inputRegImages(numberOfImages);

  std::vector< MaskReaderType::Pointer > maskReaders(numberOfImages);
  std::vector< InputMaskType::Pointer > inputMasks(numberOfImages);

  for (unsigned int i = 0; i < numberOfImages; i++)
  {
    // Load mask
    if(debug)
      std::cout << std::endl << "Load mask " << int2str(i) << " (" << maskFileNames[i].c_str() << ")" << std::endl;
    
    maskReaders[i] = MaskReaderType::New();
    maskReaders[i] -> SetFileName( maskFileNames[i].c_str() );
    
    try
    {
      maskReaders[i]->Update();
    }
    catch( itk::ExceptionObject & excp )
    {
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
    }

    inputMasks[i] = maskReaders[i]->GetOutput();

    try
    {
      inputMasks[i]->Update();
    }
    catch( itk::ExceptionObject & excp )
    {
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
    }

    if(debug)
    {
      std::cout << std::endl << "Input mask " << int2str(i) << " : " << std::endl << inputMasks[i] << std::endl;
    }

  }

  std::vector< SimilarityMetricType::MeasureType > measures(numberOfImages);

  std::vector< SimilarityMetricType::Pointer > metrics(numberOfImages);

  std::vector<ComputeNormalizedCrossCorrelationFilterType::Pointer> ncc(numberOfImages);

  std::vector< InterpolatorType::Pointer > interpolators(numberOfImages);

  std::vector< MaskFilterType::Pointer > maskFilters(numberOfImages);

  TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();

  for (unsigned int i = 0; i < numberOfImages; i++)
  {
      // Load input image
    if(debug)
      std::cout << std::endl <<"Load input image " << int2str(i) << " (" << inputFileNames[i].c_str() << ")" << std::endl;
    
    inputReaders[i] = ReaderType::New();
    inputReaders[i] -> SetFileName( inputFileNames[i].c_str() );

    try
    {
      inputReaders[i]->Update();
    }
    catch( itk::ExceptionObject & excp )
    {
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
    }

    maskFilters[i] = MaskFilterType::New();
    maskFilters[i]->SetInput(inputReaders[i]->GetOutput());
    maskFilters[i]->SetMaskImage(inputMasks[i].GetPointer());
    maskFilters[i]->Update();

    inputImages[i] = maskFilters[i]->GetOutput();

    try
    {
      inputImages[i]->Update();
    }
    catch( itk::ExceptionObject & excp )
    {
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
    }

      // Load input image registered
    if(debug)
      std::cout<<"Load input image registered " << int2str(i) << " (" << inputRegFileNames[i].c_str() << ")" << std::endl;
    
    inputRegReaders[i] = ReaderType::New();
    inputRegReaders[i] -> SetFileName( inputRegFileNames[i].c_str() );

    try
    {
      inputRegReaders[i]->Update();
    }
    catch( itk::ExceptionObject & excp )
    {
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
    }

    inputRegImages[i] = inputRegReaders[i]->GetOutput();

    try
    {
      inputRegImages[i]->Update();
    }
    catch( itk::ExceptionObject & excp )
    {
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
    }

    //Compute the global normalized correlation coefficient between the target and the template images 
    
    if(itkNCCArg.isSet())
    {
      //ITK Global ncc
      metrics[i] = SimilarityMetricType::New();
      metrics[i]->SetFixedImage(inputImages[i].GetPointer());
      metrics[i]->SetFixedImageRegion(inputImages[i]->GetBufferedRegion());
      metrics[i]->SetMovingImage(inputRegImages[i].GetPointer());

      interpolators[i] = InterpolatorType::New();
      interpolators[i]->SetInputImage(inputRegImages[i].GetPointer());

      metrics[i]->SetTransform(transform.GetPointer());
      metrics[i]->SetInterpolator(interpolators[i].GetPointer());

      metrics[i]->Initialize();

      measures[i] = metrics[i]->GetValue(transform->GetParameters());
    }
    else
    {
      //Gorthi's NCC
      ncc[i] = ComputeNormalizedCrossCorrelationFilterType::New();

      ncc[i]->SetInput(0, inputImages[i].GetPointer()); // read the target(fixed) image
      ncc[i]->SetInput(1, inputRegImages[i].GetPointer()); // read the atlas(moving) image

      ncc[i]->SetBlockHalfSizeX(5);
      ncc[i]->SetBlockHalfSizeY(5);
      ncc[i]->SetBlockHalfSizeZ(5);
      ncc[i]->SetSigmaThreshold(1e-6);

      ncc[i]->Update();
      measures[i] = ncc[i]->GetNormalizedCrossCorrelationValue();
    }    

    //Remap the normalized correlation from range [-1,1] to range [0,1]
    measures[i] = 0.5 * (measures[i] + 1.0);

    std::cout << "Normalized correlation with atlas #" << int2str(i) <<" = " << measures[i] << std::endl;

    if(debug)
    {
      std::cout << "Input mask " << int2str(i) << " : " << std::endl << inputMasks[i] << std::endl;
    }
  }

  //std::cout << measures << std::endl;

  ImageDuplicatorType::Pointer duplicator = ImageDuplicatorType::New();
  duplicator->SetInputImage(inputRegImages[0]);
  duplicator->Update();

  InputImageType::Pointer outputMask = duplicator->GetOutput();

  InputMaskType::SizeType size = inputMasks[0]->GetLargestPossibleRegion().GetSize();
  InputMaskType::IndexType inputIndex;
  OutputMaskType::IndexType outputIndex;

  //std::cout << std::endl << "Update mask...";

  std::vector<float> temp(numberOfImages);
  double pixelValue = 0.0;
  double inValue = 0.0;
  double outValue = 0.0;
  double totalWeight = 0.0;

  clock_t init,final;

  std::vector< InputImageType::Pointer > weightImages(numberOfImages);
  InputImageIteratorType* weightImagesIts = new InputImageIteratorType[numberOfImages];
  std::vector< ImageDuplicatorType::Pointer > duplicators(numberOfImages);

  if(method == 0)//Perform majority voting 
  {
    std::cout << std::endl << "Fusion by majority voting" << std::endl;
    init=clock();
    for(unsigned int k=0; k< size[2];k++)
      for(unsigned int j=0; j < size[1]; j++)
        for(unsigned int i=0; i< size[0]; i++)
        {
          inputIndex[0]=i;
          inputIndex[1]=j;
          inputIndex[2]=k;

          outputIndex[0]=i;
          outputIndex[1]=j;
          outputIndex[2]=k;

          for(unsigned int n=0; n<numberOfImages; n++)
          {
            temp[n] = inputMasks[n]->GetPixel(inputIndex);
          }
          
          majorityVoting(temp,pixelValue);
          outputMask->SetPixel(outputIndex,pixelValue);
        }
    final=clock()-init;
    std::cout << "Elapsed time = " << final << " clocks ("<< (double)final / ((double)CLOCKS_PER_SEC) << "s.) \n" << std::endl;

    if( strncmp( outputCSVFileName , "undefined" , sizeof(outputCSVFileName) - 1) )
    {
      std::cout << "Write evaluation to file" << outputCSVFileName << std::endl;

      std::ofstream fout(outputCSVFileName, std::ios_base::out | std::ios_base::app);

      fout << outputFileName << ',' << method << ',' << "basic" << ',';
      fout << final << ',' << (double)final / ((double)CLOCKS_PER_SEC) << std::endl;
      fout.close();
    }

  }
  else if(method == 1)//Perform global weighted voting based on the normalized correlation coefficient (image)
  {
    std::cout << std::endl << "Fusion by global weighted voting" << std::endl;
    init=clock();

    totalWeight=0.0;
    double  maxWeight = 0.0; //weights in [0,1]
    for(unsigned int n=0;n<numberOfImages;n++)
    {
        totalWeight += std::abs(measures[n]);
        
        if (measures[n] > maxWeight)
          maxWeight = measures[n];
    }

    //const char *outputFileNameIN = "/home/ch176971/Desktop/inWeightsGWV.csv";
    //std::cout << "Save weights to file " << outputFileNameIN << std::endl;
    //std::ofstream foutIN(outputFileNameIN);

    //const char *outputFileNameOUT = "/home/ch176971/Desktop/outWeightsGWV.csv";
    //std::cout << "Save weights to file " << outputFileNameOUT << std::endl;
    //std::ofstream foutOUT(outputFileNameOUT);

    int pixelCount = 1;

    for(unsigned int k=0; k< size[2];k++)
      for(unsigned int j=0; j < size[1]; j++)
        for(unsigned int i=0; i< size[0]; i++)
        {
          inputIndex[0]=i;
          inputIndex[1]=j;
          inputIndex[2]=k;

          outputIndex[0]=i;
          outputIndex[1]=j;
          outputIndex[2]=k;

          for(unsigned int n=0; n<numberOfImages; n++)
          {
            temp[n] = inputMasks[n]->GetPixel(inputIndex);
          }

          pixelValue=0;
          inValue=0;
          outValue=0;
          for(unsigned int n=0;n<numberOfImages;n++)
          {
            if(temp[n]==1.0)
              inValue += measures[n];
            else
              outValue += measures[n];
            //pixelValue += ( measures[n] / maxWeight ) * (temp[n]*255.0);
            //std::cout << "metric = " << measures[n] << " , maxWeight = " << maxWeight << std::endl;
          }

          if(inValue > outValue)
            pixelValue = 1.0;
          else
            pixelValue = 0.0;
          
          //pixelValue = pixelValue / numberOfImages;

          //foutIN << pixelCount << " (" << inValue << "/" << pixelValue << "),";
          //foutOUT << pixelCount << " (" << outValue << "/" << pixelValue << "),";

          //std::cout << "Set voxel value " << pixelValue << std::endl;
          outputMask->SetPixel(outputIndex,pixelValue);
          //std::cout << "done" << std::endl;

          pixelCount++;
        }

    //foutIN << std::endl;
    //foutIN.close();

    //foutOUT << std::endl;
    //foutOUT.close();

    final=clock()-init;
    std::cout << "Elapsed time = " << final << " clocks ("<< (double)final / ((double)CLOCKS_PER_SEC) << "s.) \n" << std::endl;

    if( strncmp( outputCSVFileName , "undefined" , sizeof(outputCSVFileName) - 1) )
    {
      std::cout << "Write evaluation to file" << outputCSVFileName << std::endl;

      std::ofstream fout(outputCSVFileName, std::ios_base::out | std::ios_base::app);

      fout << outputFileName << ',' << method << ',' << "basic" << ',';
      fout << final << ',' << (double)final / ((double)CLOCKS_PER_SEC) << std::endl;
      fout.close();
    }

  }  
  else if(method == 2)//Perform local weighted voting based on local correlation coefficient (patches)
  {
    std::cout << std::endl << "Fusion by local weighted voting (patch radius = " << int2str(localPatchRadius) << ")" << std::endl;
    
    //int localPatchRadius = 1;

          // InternalImageIteratorType* warpedImgIts = new InternalImageIteratorType[numberOfImages];
    InputImageIteratorType outputMaskIt(outputMask,outputMask->GetLargestPossibleRegion());

    InputMaskIteratorType* inputMaskIts = new InputMaskIteratorType[numberOfImages];

    InputImageNeighborhoodIteratorType* inputRegImageNeighborhoodIts = new InputImageNeighborhoodIteratorType[numberOfImages];

    InputImageType::SizeType radius;
    radius[0] = localPatchRadius; radius[1] = localPatchRadius; radius[2] = localPatchRadius;

    //InputImages corresponds to an array containing n duplicates of the target image (where n=numberOfImages) but we create only one iterator
    InputImageNeighborhoodIteratorType inputImageNeighborhoodIt(radius, inputImages[0], inputImages[0]->GetRequestedRegion() );
    InputImageNeighborhoodIteratorType inputImageNeighborIt(radius, inputImages[0], inputImages[0]->GetLargestPossibleRegion() );

    for( int i = 0; i < numberOfImages; i++ )
    {
      InputMaskIteratorType it2( inputMasks[i], inputMasks[i]->GetLargestPossibleRegion() );

      inputMaskIts[i] = it2;

      InputImageNeighborhoodIteratorType neighborIt(radius, inputRegImages[i], inputRegImages[i]->GetLargestPossibleRegion() );
      inputRegImageNeighborhoodIts[i] = neighborIt;
    }

    init=clock();
    localWeightedVoting(localPatchRadius, numberOfImages,inputImageNeighborIt,outputMaskIt,inputMaskIts,inputRegImageNeighborhoodIts);
    
    /*
    for(int i=0;i<numberOfImages;i++)
    {
      weightImages[i] = duplicators[i]->GetOutput();
      std::cout << weightImages[i] << std::endl;
      InputImageIteratorType wit( weightImages[i], weightImages[i]->GetLargestPossibleRegion() );
      weightImagesIts[i] = wit;
    }

    localWeightedVoting(localPatchRadius, numberOfImages,inputImageNeighborIt,outputMaskIt,inputMaskIts,inputRegImageNeighborhoodIts,weightImagesIts);
    */

    final=clock()-init;

    std::cout << "Elapsed time = " << final << " clocks ("<< (double)final / ((double)CLOCKS_PER_SEC) << "s.) \n" << std::endl;

    if( strncmp( outputCSVFileName , "undefined" , sizeof(outputCSVFileName) - 1) )
    {
      std::cout << "Write evaluation to file" << outputCSVFileName << std::endl;

      std::ofstream fout(outputCSVFileName, std::ios_base::out | std::ios_base::app);

      fout << outputFileName << ',' << method << ',' << "basic" << ',';
      fout << final << ',' << (double)final / ((double)CLOCKS_PER_SEC) << std::endl;
      fout.close();
    }

    //std::cout << "Done!" << std::endl;
  }

  
  /*
  WriterType::Pointer weightWriters[numberOfImages];
  for(int n=0;n<numberOfImages;n++)
  {
    std::cout << "loop" << int2str(n) << std::endl; 
    try
    {
      weightImages[n]->Update();
      std::string filename = std::string("/home/tourbier/Desktop/") + std::string("atlas") + int2str(n) + std::string("_weights.nii");
      std::cout << "Save weight image for atlas " << int2str(n) << " of " << int2str(numberOfImages) <<" as " << filename << std::endl;
      weightWriters[n]->SetFileName(filename);
      weightWriters[n]->SetInput(weightImages[n].GetPointer());
      weightWriters[n]->Update();
    }
    catch( itk::ExceptionObject & excp )
    {
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
    }
  }
  */

  try
  {
    outputMask->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
  }
  
  /*
  ThresholdingFilterType::Pointer thresholder = ThresholdingFilterType::New();
  thresholder->SetLowerThreshold(lower);
  thresholder->SetUpperThreshold(upper);
  thresholder->SetOutsideValue(outsideValue);
  thresholder->SetInsideValue(insideValue);
  thresholder->SetInput(outputMask.GetPointer());
  thresholder->Update();
  */  
  
  /*OtsuThresholdingFilterType::Pointer otsuThresholder = OtsuThresholdingFilterType::New();
  otsuThresholder->SetInput(outputMask.GetPointer());
  otsuThresholder->SetOutsideValue(outsideValue);
  otsuThresholder->SetInsideValue(insideValue);
  otsuThresholder->SetNumberOfHistogramBins(numberOfBins);*/

  CCFilterType::Pointer ccFilter = CCFilterType::New();
  ccFilter->SetInput(outputMask.GetPointer());
  ccFilter->FullyConnectedOff();

  RelabelType::Pointer relabelFilter = RelabelType::New();
  relabelFilter->SetInput(ccFilter->GetOutput());
  relabelFilter->SetMinimumObjectSize(minimumObjectSize);
  relabelFilter->Update();

  FillHolesFilterType::Pointer fillHolesFilter = FillHolesFilterType::New();
  OutputMaskType::SizeType indexRadius;

  indexRadius[0] = radiusX;
  indexRadius[1] = radiusY;
  indexRadius[2] = radiusZ;

  fillHolesFilter->SetInput(relabelFilter->GetOutput());
  fillHolesFilter->SetRadius(indexRadius);
  fillHolesFilter->SetBackgroundValue(0);
  fillHolesFilter->SetForegroundValue(1);
  fillHolesFilter->SetMajorityThreshold(2);

  fillHolesFilter->Update();

  CastFilterType::Pointer castFilter = CastFilterType::New();
  castFilter->SetInput(fillHolesFilter->GetOutput());

  //outputMask = castFilter->GetOutput();

  /*
  std::cout<<" Filling all holes"<<   std::endl;
  typedef itk::ImageRegionIteratorWithIndex<OutputMaskType>  IteratorWithIndex;
  IteratorWithIndex vfIter( relabelFilter->GetOutput(),relabelFilter->GetOutput()->GetLargestPossibleRegion() );

  for(  vfIter.GoToBegin(); !vfIter.IsAtEnd(); ++vfIter )
  {
    if (vfIter.Get()>  1 ) outputMask->SetPixel(vfIter.GetIndex(),1);
  }

  //outputMask->Update();
  */
  
  
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(outputFileName);
  writer->SetInput(outputMask.GetPointer());
  writer->Update();
  

  std::cout << std::endl << "Save output mask..." << std::endl;
  
  /*
  MaskWriterType::Pointer writer = MaskWriterType::New();
  writer->SetFileName(outputFileName);
  writer->SetInput(fillHolesFilter->GetOutput());
  writer->Update();
  */

  //int threshold = otsuThresholder->GetThreshold();
  //std::cout << "Threshold = " << threshold << std::endl;

  relabelFilter->Print(std::cout);
    
  /*WriterType::Pointer writer2 = WriterType::New();
  writer2->SetFileName(outputFileName);
  writer2->SetInput(outputMask.GetPointer());
  writer2->Update();*/

  std::cout << std::endl << "Done! " << std::endl << std::endl;

  return EXIT_SUCCESS;
};

void majorityVoting(std::vector<unsigned char> & values, unsigned char & outputValue)
{
  float thresh = 0.5f * values.size();
  float sum = 0.0f;

  //std::cout << "Begining of Maj. Voting" << std::endl;
  for(unsigned int i=0;i<values.size();i++)
  {
    sum+=values[i];
  }
  //std::cout << "After sum in Maj. Voting" << std::endl;
  if(sum >= thresh)
  {
    outputValue = 1;
  }
  else
  {
    outputValue = 0;
  }
  //std::cout << "End of Maj. Voting" << std::endl;
};

void majorityVoting(std::vector<float> & values, double & outputValue)
{
  float thresh = 0.5f * values.size();
  double sum = 0.0;

  //std::cout << "Begining of Maj. Voting" << std::endl;
  for(unsigned int i=0;i<values.size();i++)
  {
    sum+=values[i];
  }
  //std::cout << "After sum in Maj. Voting" << std::endl;
  if(sum >= thresh)
  {
    outputValue = 1.0;
  }
  else
  {
    outputValue = 0.0;
  }
  //std::cout << "End of Maj. Voting" << std::endl;
};

void localWeightedVoting(int patchRadius, int numberOfImages,InputImageNeighborhoodIteratorType &targetImageIt, InputImageIteratorType &outputMaskIt, InputMaskIteratorType* &templateRegMasksIts, InputImageNeighborhoodIteratorType* &templateRegImagesIts)
{
  //InputImageNeighborhoodIteratorType::NeighborhoodType::SizeType patchSizes = targetImageIt[0]->GetSize();

  int patchSize=2*patchRadius+1;
  int elementsInPatch=patchSize*patchSize*patchSize;
  int patchCenterIndex = round(0.5 * ( elementsInPatch - 1 ));

  std::cout << "patch center index : " << int2str(patchCenterIndex) << std::endl;  

  for(int i=0;i<numberOfImages;i++)
  {
    templateRegMasksIts[i].GoToBegin();
    templateRegImagesIts[i].GoToBegin();
  }

  InputImageType::IndexType index;
  index[0]=0;
  index[1]=0;
  index[2]=0;

  //const char *outputFileName = "/home/tourbier/Desktop/weights.csv";
  //std::cout << "Save weights to file " << outputFileName << std::endl;
  //std::ofstream fout(outputFileName);

  int pixelCount=0;

  //const char *outputFileNameIN = "/home/ch176971/Desktop/inWeightsLWV.csv";
  //std::cout << "Save weights to file " << outputFileNameIN << std::endl;
  //std::ofstream foutIN(outputFileNameIN);

  //const char *outputFileNameOUT = "/home/ch176971/Desktop/outWeightsLWV.csv";
  //std::cout << "Save weights to file " << outputFileNameOUT << std::endl;
  //std::ofstream foutOUT(outputFileNameOUT);


  for(targetImageIt.GoToBegin(),outputMaskIt.GoToBegin();!targetImageIt.IsAtEnd();++targetImageIt,++outputMaskIt)
  { 
    std::vector<float> targetPatch(elementsInPatch);   
    std::vector<float> templatePatch(elementsInPatch);  
    double pixelValue = 0.0;
    double inValue = 0.0;
    double outValue = 0.0;
    double totalWeight = 0.0;
    double maxWeight = 0.0;

    std::vector<float> temp(numberOfImages);
    std::vector<double> measures(numberOfImages);

    index = targetImageIt.GetIndex();

    for(int i=0;i<numberOfImages;i++)
    {
      for(int j=0;j<elementsInPatch;j++)
      {
        targetPatch[j]=targetImageIt.GetPixel(j);
        templatePatch[j]=templateRegImagesIts[i].GetPixel(j);

        if(j==patchCenterIndex)
          temp[i]=templateRegMasksIts[i].Get();
      }
      
      //Map the ncc from range [-1,1] to range [0,1].
      measures[i] = 0.5 * (mialsrtkNormalizedCorrelationCoef(targetPatch,templatePatch) + 1.0);
      
      /*
      if(((index[0]==39) && (index[1]==30) && (index[2]==8) )||((index[0]==47) && (index[1]==32) && (index[2]==13) ))
      {
        std::cout << "----------------------------------------------------------------------------------" << std::endl;
        std::cout << "----------------------------------------------------------------------------------" << std::endl;
        std::cout << "Voxel " << index[0] << " , " << index[1] << " , "<< index[2] << " ( template mask = " << temp[i] << " )" << std::endl;
        std::cout << "----------------------------------------------------------------------------------" << std::endl;

        std::cout << "Target patch:" << std::endl;
        for( std::vector<float>::iterator it = targetPatch.begin(); it != targetPatch.end(); ++it)
          std::cout << *it << ' ';

        std::cout << "" << std::endl;

        std::cout << "Template patch:" << std::endl;
        for( std::vector<float>::iterator it = templatePatch.begin(); it != templatePatch.end(); ++it)
          std::cout << *it << ' ';

        std::cout << "" << std::endl;
        std::cout << "----------------------------------------------------------------------------------" << std::endl;
        std::cout << "NCC = " << measures[i] << std::endl;
        std::cout << "----------------------------------------------------------------------------------" << std::endl;
        std::cout << "----------------------------------------------------------------------------------" << std::endl;

      }
      */

      //if(measures[i]>=0)
      //{
      //  totalWeight+=measures[i];
      //}
        
      //Compute normalized correlation coefficient
      /*metrics[i] = NeighborhoodSimilarityMetricType::New();
      metrics[i]->SetFixedImage(targetImageIt.GetNeighborhood());
      metrics[i]->SetFixedImageRegion(targetImageIt.GetNeighborhood()->GetBufferedRegion());
      metrics[i]->SetMovingImage(templateRegImagesIts[i].GetNeighborhood());

      interpolators[i] = NeighborhoodInterpolatorType::New();
      interpolators[i]->SetInputImage(templateRegImagesIts[i].GetNeighborhood());

      metrics[i]->SetTransform(transform.GetPointer());
      metrics[i]->SetInterpolator(interpolators[i].GetPointer());

      metrics[i]->Initialize();

      measures[i] = metrics[i]->GetValue(transform->GetParameters());
      */
      
      ++templateRegMasksIts[i];
      ++templateRegImagesIts[i];
    }

    //std::cout << "Totalweight = " << totalWeight << std::endl;

    maxWeight = *std::max_element(measures.begin(),measures.end());


    //TO BE CORRECTED
    if(maxWeight>0)
    {
      for(int i=0;i<numberOfImages;i++)
      {
        if(temp[i]==1)
          inValue += (double)measures[i];
        else
          outValue += (double)measures[i];
          //pixelValue+= ((double)measures[i]/(double)maxWeight) * (temp[i]*255.0);//*255.0);         
      }
      if(inValue > outValue)
        pixelValue = 1.0;
      else
        pixelValue = 0.0;

      //pixelValue = pixelValue / numberOfImages;
    }

    //foutIN << pixelCount << " (" << inValue << "/" << pixelValue << "),";
    //foutOUT << pixelCount << " (" << outValue << "/" << pixelValue << "),";

    //std::cout <<"Pixel #" << pixelCount<< ": LWV performed ("<< inValue <<"/"<< outValue <<"), pixel value = " << pixelValue << std::endl;
   

    /*
    if(((index[0]==39) && (index[1]==30) && (index[2]==8) )||((index[0]==47) && (index[1]==32) && (index[2]==13) ))
    {
      std::cout << "Target patch:" << std::endl;
      for( std::vector<float>::iterator it = targetPatch.begin(); it != targetPatch.end(); ++it)
        std::cout << *it << ' ';

      std::cout << "" << std::endl;

      std::cout << "Template patch:" << std::endl;
      for( std::vector<float>::iterator it = templatePatch.begin(); it != templatePatch.end(); ++it)
        std::cout << *it << ' ';

      std::cout << "" << std::endl;
    }
    */
    
    /*
    std::cout << "Set voxel value " << pixelValue;// << std::endl;

    std::vector<double>::iterator it;
    std::vector<float>::iterator it2;

    std::cout << " ( [weight/pix]:";
    for(it = measures.begin(), it2 = temp.begin(); it != measures.end(); ++it, ++it2)
      std::cout << *it << "/" << *it2 << " ";
    std::cout << ")" << std::endl;
    */

    //fout << measures[0] << ',';
      
    outputMaskIt.Set(pixelValue);

    pixelCount++;
  }

  //foutIN << std::endl;
  //foutIN.close();

  //foutOUT << std::endl;
  //foutOUT.close();

  //fout << std::endl;
  //fout.close();
};
