/*==========================================================================

  © Université de Strasbourg - Centre National de la Recherche Scientifique

  Date: 11/04/2012
  Author(s): Marc Schweitzer (marc.schweitzer(at)unistra.fr)
             Julien Pontabry (pontabry@unistra.fr)

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

#ifndef BTK_IMAGE_HELPER_TXX
#define BTK_IMAGE_HELPER_TXX

#include "btkImageHelper.h"


// STL includes
#include "iostream"

// Local includes
#include "btkFileHelper.h"


namespace btk
{

template < class TImageInput, class TImageOutput >
void ImageHelper< TImageInput, TImageOutput >::WriteImage(typename TImageInput::Pointer image, const std::string &fileName)
{
    std::cout << "Writing \"" << fileName << "\"... " << std::flush;

    typename ImageWriter::Pointer writer = ImageWriter::New();
    writer->SetFileName(fileName);
    writer->SetInput(image);
    writer->Update();

    std::cout << "done." << std::endl;
}

//----------------------------------------------------------------------------------------

template < class TImageInput, class TImageOutput >
void ImageHelper< TImageInput, TImageOutput >::WriteImage(std::vector< typename TImageInput::Pointer > &images, std::vector<std::string> &fileNames)
{

    if(images.size() == fileNames.size())
    {
        for(int i = 0; i < images.size(); i++)
        {
            WriteImage(images[i], fileNames[i]);
        }
    }
    else
    {
        std::string err("vector of images and vector of names have not the same size !");
        throw err;

    }
}

//----------------------------------------------------------------------------------------

template < class TImageInput, class TImageOutput >
typename TImageInput::Pointer ImageHelper< TImageInput, TImageOutput >::ReadImage(const std::string &fileName)
{
    std::cout << "Reading image \"" << fileName << "\"... " << std::flush;

    typename ImageReader::Pointer reader = ImageReader::New();
    reader->SetFileName(fileName);
    reader->Update();

    std::cout << "done." << std::endl;

    return reader->GetOutput();
}

//----------------------------------------------------------------------------------------

template < class TImageInput, class TImageOutput >
typename TImageInput::ConstPointer ImageHelper< TImageInput, TImageOutput >::ReadConstImage(const std::string &fileName)
{
    return ReadImage(fileName).GetPointer();
}

//----------------------------------------------------------------------------------------

template < class TImageInput, class TImageOutput >
std::vector< typename TImageInput::Pointer > &ImageHelper< TImageInput, TImageOutput >::ReadImage(std::vector<std::string> &fileNames)
{
    //FIXME : memory leak about the returned vector (maybe use a itk::SmartPointer<>, or passing the vector per value)
    std::vector< typename TImageInput::Pointer > *ptrImages = new std::vector< typename TImageInput::Pointer >;
    std::vector< typename TImageInput::Pointer > &images = *ptrImages;
    images.resize(fileNames.size());

    for(int i = 0; i < fileNames.size(); i++)
    {
        images[i] = ReadImage(fileNames[i]);
    }

    return images;
}

//----------------------------------------------------------------------------------------

template < class TImageInput, class TImageOutput >
typename TImageOutput::Pointer ImageHelper< TImageInput, TImageOutput >::CreateNewImageFromPhysicalSpaceOfConst(typename TImageInput::ConstPointer image, typename TImageOutput::PixelType defaultValue)
{

    typename TImageOutput::Pointer newImage = TImageOutput::New();
    newImage->SetRegions(image->GetLargestPossibleRegion());
    newImage->SetOrigin(image->GetOrigin());
    newImage->SetSpacing(image->GetSpacing());
    newImage->SetDirection(image->GetDirection());
    newImage->Allocate();
    newImage->FillBuffer(defaultValue);

    return newImage;
}

//----------------------------------------------------------------------------------------

template < class TImageInput, class TImageOutput >
typename TImageOutput::Pointer ImageHelper< TImageInput, TImageOutput >::CreateNewImageFromPhysicalSpaceOf(typename TImageInput::Pointer image, typename TImageOutput::PixelType defaultValue)
{
    typename TImageOutput::Pointer newImage = TImageOutput::New();
    newImage->SetRegions(image->GetLargestPossibleRegion());
    newImage->SetOrigin(image->GetOrigin());
    newImage->SetSpacing(image->GetSpacing());
    newImage->SetDirection(image->GetDirection());
    newImage->Allocate();
    newImage->FillBuffer(defaultValue);

    return newImage;
}
//----------------------------------------------------------------------------------------

template < class TImageInput, class TImageOutput >
std::vector< typename TImageOutput::Pointer > &ImageHelper< TImageInput, TImageOutput >::CreateNewImageFromPhysicalSpaceOf(std::vector< typename TImageInput::Pointer > &images, typename TImageOutput::PixelType defaultValue)
{
    //FIXME : memory leak about the returned vector (maybe use a itk::SmartPointer<>, or passing the vector per value)
    std::vector< typename TImageOutput::Pointer > *ptrNewImages = new std::vector< typename TImageOutput::Pointer >;
    std::vector< typename TImageOutput::Pointer > &newImages = *ptrNewImages;

    for(typename std::vector< typename TImageInput::Pointer >::iterator it = images.begin(); it != images.end(); it++)
    {
        newImages.push_back(CreateNewImageFromPhysicalSpaceOf(*it, defaultValue));
    }

    return newImages;
}

//----------------------------------------------------------------------------------------

template < class TImageInput, class TImageOutput >
std::vector< typename TImageOutput::Pointer > &ImageHelper< TImageInput, TImageOutput >::CreateNewImageFromPhysicalSpaceOfConst(std::vector< typename TImageInput::ConstPointer > &images, typename TImageOutput::PixelType defaultValue)
{
    //FIXME : memory leak about the returned vector (maybe use a itk::SmartPointer<>, or passing the vector per value)
    std::vector< typename TImageOutput::Pointer > *ptrNewImages = new std::vector< typename TImageOutput::Pointer >;
    std::vector< typename TImageOutput::Pointer > &newImages = *ptrNewImages;

    for(typename std::vector< typename TImageInput::Pointer >::iterator it = images.begin(); it != images.end(); it++)
    {
        newImages.push_back(CreateNewImageFromPhysicalSpaceOfConst(*it, defaultValue));
    }

    return newImages;
}

//----------------------------------------------------------------------------------------

template < class TImageInput, class TImageOutput >
bool ImageHelper< TImageInput, TImageOutput >::IsInSamePhysicalSpace(typename TImageInput::Pointer firstImage, typename TImageOutput::Pointer secondImage, double epsilon)
{
    typename TImageInput::SizeType             firstSize = firstImage->GetLargestPossibleRegion().GetSize();
    typename TImageOutput::SizeType           secondSize = secondImage->GetLargestPossibleRegion().GetSize();
    typename TImageInput::SpacingType       firstSpacing = firstImage->GetSpacing();
    typename TImageOutput::SpacingType     secondSpacing = secondImage->GetSpacing();
    typename TImageInput::PointType          firstOrigin = firstImage->GetOrigin();
    typename TImageOutput::PointType        secondOrigin = secondImage->GetOrigin();
    typename TImageInput::DirectionType   firstDirection = firstImage->GetDirection();
    typename TImageOutput::DirectionType secondDirection = secondImage->GetDirection();

    bool SameSize, SameSpacing, SameOrigin, SameDirection, SameDimension;
    SameSize = SameSpacing = SameOrigin = SameDirection = SameDimension = true;

    SameDimension = (firstImage->GetImageDimension() == secondImage->GetImageDimension());

    const unsigned int Dim =  std::min(firstImage->GetImageDimension(), secondImage->GetImageDimension());

    SameSize = (firstSize == secondSize);

    for(unsigned int i = 0; i < Dim; i++)
    {
        SameSpacing = SameSpacing && (std::abs(firstSpacing[i] - secondSpacing[i]) < epsilon);

        SameOrigin = SameOrigin && (std::abs(firstOrigin[i] - secondOrigin[i]) < epsilon);


        for(unsigned int j = 0; j < Dim; j++)
        {
            SameDirection = SameDirection && (std::abs(firstDirection(i,j) - secondDirection(i,j)) < epsilon);
        }
    }

    return (SameDimension && SameSize && SameDirection && SameOrigin && SameSpacing);

}

//----------------------------------------------------------------------------------------

template < class TImageInput, class TImageOutput >
bool ImageHelper< TImageInput, TImageOutput >::IsInSamePhysicalSpace(const std::vector< typename TImageInput::Pointer > &images, double epsilon)
{
    bool isInSameSpace = true;

    if(images.size() > 0)
    {
        typename TImageInput::SizeType           firstSize = images[0]->GetLargestPossibleRegion().GetSize();
        typename TImageInput::SpacingType     firstSpacing = images[0]->GetSpacing();
        typename TImageInput::PointType        firstOrigin = images[0]->GetOrigin();
        typename TImageInput::DirectionType firstDirection = images[0]->GetDirection();

        unsigned int im = 1;

        bool SameSize, SameSpacing, SameOrigin, SameDirection;
        SameSize = SameSpacing = SameOrigin = SameDirection = true;

        while(im < images.size() && isInSameSpace)
        {
            typename TImageInput::SizeType           secondSize = images[im]->GetLargestPossibleRegion().GetSize();
            typename TImageInput::SpacingType     secondSpacing = images[im]->GetSpacing();
            typename TImageInput::PointType        secondOrigin = images[im]->GetOrigin();
            typename TImageInput::DirectionType secondDirection = images[im]->GetDirection();

            const unsigned int Dim =  TImageInput::ImageDimension;

            SameSize = (firstSize == secondSize);

            for(unsigned int i = 0; i<Dim; i++)
            {
                SameSpacing = SameSpacing && (std::abs(firstSpacing[i] - secondSpacing[i]) < epsilon);

                SameOrigin = SameOrigin && (std::abs(firstOrigin[i] - secondOrigin[i]) < epsilon);


                for(unsigned int j= 0; j<Dim; j++)
                {
                    SameDirection = SameDirection && (std::abs(firstDirection(i,j) - secondDirection(i,j)) < epsilon);
                }
            }

            isInSameSpace = (SameSize && SameDirection && SameOrigin && SameSpacing);

            im++;
        } // for each other image
    }

    return isInSameSpace;
}

//----------------------------------------------------------------------------------------

template < class TImageInput, class TImageOutput >
typename TImageOutput::Pointer ImageHelper< TImageInput, TImageOutput >::ReadOrCreateImage(const std::string &fileName, typename TImageInput::Pointer image, typename TImageOutput::PixelType defaultValue)
{
    typename TImageOutput::Pointer newImage = NULL;

    if(!fileName.empty() && btk::FileHelper::FileExist(fileName))
    {
        newImage = ReadImage(fileName);
    }
    else
    {
        std::cout << "Creating new image with pixel value set to " << defaultValue << std::endl;
        newImage = CreateNewImageFromPhysicalSpaceOf(image.GetPointer(), defaultValue);
    }

    return newImage;
}

//----------------------------------------------------------------------------------------

template < class TImageInput, class TImageOutput >
typename TImageOutput::Pointer ImageHelper< TImageInput, TImageOutput >::DeepCopy(typename TImageInput::ConstPointer image)
{
    typename TImageOutput::Pointer output = TImageOutput::New();
    output->SetRegions(image->GetLargestPossibleRegion());
    output->SetSpacing(image->GetSpacing());
    output->SetOrigin(image->GetOrigin());
    output->SetDirection(image->GetDirection());
    output->Allocate();

    itk::ImageRegionConstIterator< TImageInput > inputIt(image, image->GetLargestPossibleRegion());
    itk::ImageRegionIterator< TImageOutput >    outputIt(output, output->GetLargestPossibleRegion());

    for(inputIt.GoToBegin(), outputIt.GoToBegin(); !inputIt.IsAtEnd() && !outputIt.IsAtEnd(); ++inputIt, ++outputIt)
    {
        outputIt.Set(inputIt.Get());
    }

    return output;
}

//----------------------------------------------------------------------------------------
template < class TImageInput, class TImageOutput >
bool
ImageHelper< TImageInput, TImageOutput >::AreOrthos(typename TImageInput::Pointer image1, typename TImageInput::Pointer image2, float threshold)
{
    // TODO : This can be done more easily !
    itk::Vector<double,3> physicalZVector1, physicalZVector2;

    typename TImageInput::IndexType startIndex1, startIndex2;
    typename TImageInput::IndexType endIndex1, endIndex2;
    typename TImageInput::PointType startPoint1, startPoint2;
    typename TImageInput::PointType endPoint1, endPoint2;

    startIndex1[0] = startIndex2[0] = 0;
    startIndex1[1] = startIndex2[1] = 0;
    startIndex1[2] = startIndex2[2] = 0;

    image1->TransformIndexToPhysicalPoint(startIndex1,startPoint1);
    image2->TransformIndexToPhysicalPoint(startIndex2,startPoint2);

    endIndex1[0] = endIndex2[0] = 0;
    endIndex1[1] = endIndex2[1] = 0;
    endIndex1[2] = endIndex2[2] = 1;

    image1->TransformIndexToPhysicalPoint(endIndex1,endPoint1);
    image2->TransformIndexToPhysicalPoint(endIndex2,endPoint2);

    physicalZVector1[0] = startPoint1[0] - endPoint1[0];
    physicalZVector1[1] = startPoint1[1] - endPoint1[1];
    physicalZVector1[2] = startPoint1[2] - endPoint1[2];

    physicalZVector2[0] = startPoint2[0] - endPoint2[0];
    physicalZVector2[1] = startPoint2[1] - endPoint2[1];
    physicalZVector2[2] = startPoint2[2] - endPoint2[2];

    physicalZVector1.Normalize();
    physicalZVector2.Normalize();


    // Test if images are orthogonal or not
    double dotProduct = physicalZVector1 * physicalZVector2;

    if(dotProduct< threshold)
    {
        return true;
    }
    else
    {
        return false;
    }

}
//----------------------------------------------------------------------------------------
template < class TImageInput, class TImageOutput >
typename TImageOutput::Pointer
ImageHelper< TImageInput, TImageOutput >::CastImage(typename TImageInput::Pointer image)
{
    typename CastImageFilter::Pointer filter = CastImageFilter::New();
    filter->SetInput(image);
    filter->Update();

    typename TImageOutput::Pointer OutputImage = filter->GetOutput();

    return OutputImage;
}
//----------------------------------------------------------------------------------------
template < class TImageInput, class TImageOutput >
std::vector< typename TImageOutput::Pointer > &
ImageHelper< TImageInput, TImageOutput >::CastImage(std::vector< typename TImageInput::Pointer> &images)
{
    //FIXME : memory leak about the returned vector (maybe use a itk::SmartPointer<>, or passing the vector per value)
    std::vector< typename TImageOutput::Pointer > *ptrOutputImages = new std::vector< typename TImageOutput::Pointer >;
    std::vector< typename TImageOutput::Pointer > &OutputImages = *ptrOutputImages;
    OutputImages.resize(images.size());

    for(int i = 0; i < images.size(); i++)
    {
        OutputImages[i] = CastImage(images[i]);
    }

    return OutputImages;
}


} // namespace btk

#endif // BTK_IMAGE_HELPER_TXX
