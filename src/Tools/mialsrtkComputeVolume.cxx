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
#include <sstream>  

#include <iostream>
#include <fstream> 
#include <string>
#include <stdlib.h> 

/* Itk includes */
#include "itkEuler3DTransform.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkImageRegionIterator.h"

int main( int argc, char *argv[] )
{

    try {

        const char *input = NULL;
    
        const char *test = "undefined";

        // Parse arguments
        TCLAP::CmdLine cmd("Compute the volume of binary image (For example the volume of the brain mask).", ' ', "Unversioned");

        // Input LR images
        TCLAP::ValueArg<std::string> inArg  ("i","input-mask","Input mask",true,"","string",cmd);

        // Parse the argv array.
        cmd.parse( argc, argv );
    
        input = inArg.getValue().c_str();

        // typedefs
        const   unsigned int    Dimension = 3;
        typedef unsigned char  PixelType;


        typedef itk::Image< PixelType, Dimension >  ImageMaskType;
        typedef itk::ImageFileReader< ImageMaskType > MaskReaderType;

        typedef ImageMaskType::RegionType               RegionType;
       
        typedef itk::ImageFileReader< ImageMaskType >   ImageMaskReaderType;
        
        typedef itk::ImageRegionIterator< ImageMaskType >  MaskIteratorType;
        
       
        std::cout<<"Reading mask image : "<<input<<std::endl;
        MaskReaderType::Pointer maskReader = MaskReaderType::New();
        maskReader -> SetFileName( input );
        maskReader -> Update();

        ImageMaskType::Pointer imageMask = maskReader  -> GetOutput();
        
        MaskIteratorType itMask(imageMask, imageMask->GetLargestPossibleRegion());
        
        //Count the number of voxels contained in the mask
        double numberOfVoxelsInMask = 0.0;
        for(itMask.GoToBegin(); !itMask.IsAtEnd(); ++itMask)
        {
            if(itMask.Get() > 0.0) numberOfVoxelsInMask++;
        }
        
        //Determine the volume of one voxel in mm3
        ImageMaskType::SpacingType spacing = imageMask->GetSpacing();
        
       double voxelVolume = spacing[0] * spacing[1] * spacing[2]; 
       
       //Compute volume in mm^3
       double maskVolumeInMM3 = numberOfVoxelsInMask * voxelVolume;
       
       //Conversion in mL(1 mL = 1000 mm^3)
       double maskVolumeInML = 0.001 * maskVolumeInMM3;
       
       std::cout << "Volume : " << maskVolumeInMM3 << " mm3 = " << maskVolumeInML << " mL" << std::endl;

    } catch (TCLAP::ArgException &e)  // catch any exceptions
    { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

    return EXIT_SUCCESS;
}
