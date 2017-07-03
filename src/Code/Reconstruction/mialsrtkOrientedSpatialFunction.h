/*=========================================================================

Copyright (c) 2017 Medical Image Analysis Laboratory (MIAL), Lausanne
  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
==========================================================================*/


#ifndef __mialsrtkOrientedSpatialFunction_h
#define __mialsrtkOrientedSpatialFunction_h

#include "itkSpatialFunction.h"
#include "itkFixedArray.h"
#include "itkPoint.h"
#include "itkMatrix.h"
#include "itkVector.h"
#include "itkGaussianSpatialFunction.h"
#include "itkFixedArray.h"

namespace mialsrtk
{

using namespace itk;

/** \class OrientedSpatialFunction
 * \brief N-dimensional oriented spatial function class
 *
 * This class implements a function oriented in a specific direction. This is
 * used to have the PSF in the correct orientation in world coordinates.
 *
 * \ingroup SpatialFunctions
 */
template <typename TOutput=double,
          unsigned int VImageDimension=3,
          typename TInput=Point<double, VImageDimension> >
class OrientedSpatialFunction
        : public SpatialFunction<TOutput, VImageDimension, TInput>
{
public:

    typedef enum {
        BOXCAR=0,
        GAUSSIAN=1,
        ANISOTROPIC=2,
        ISOTROPIC=3,
    } PSF_type;

    /** Standard class typedefs. */
    typedef OrientedSpatialFunction                                 Self;
    typedef SpatialFunction<TOutput, VImageDimension, TInput>       Superclass;
    typedef SmartPointer<Self>                                      Pointer;
    typedef SmartPointer<const Self>                                ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(OrientedSpatialFunction, SpatialFunction);

    /** Input type for the function. */
    typedef typename Superclass::InputType InputType;

    /** Output type for the function. */
    typedef typename Superclass::OutputType OutputType;

    /** Direction type. */
    typedef Matrix<double,VImageDimension,VImageDimension>  DirectionType;

    /** Point type */
    typedef Point<double,VImageDimension>  PointType;

    /** Spacing type */
    typedef Vector<double,VImageDimension>  SpacingType;

    /** Gaussian function type */
    typedef GaussianSpatialFunction< double,
    VImageDimension,
    PointType> GaussianFunctionType;

    /** Array type */
    typedef FixedArray<double,VImageDimension> ArrayType;


    /** Function value */
    OutputType Evaluate(const TInput& position) const;

    /** Sets spacing. This method changes the standard deviations of the Gaussian
   * function accordingly. */
    void SetSpacing(SpacingType spacing)
    {
        m_Spacing = spacing.GetVnlVector();

        ArrayType sigma;

        float cst = 8*sqrt(log(2));

        // 06 Feb 2015: PSF correction
        // sigma of Gaussian PSF should be equal to FWHM only in the slice select direction
        // Change Gaussian parameters (in case of inputs with different spaces)

        float sliceThickness = 0.0;
        unsigned int sliceThicknessIndex = 0;

        for(unsigned int i=0;i<3;i++)
        {
            if(spacing[i]>sliceThickness)
            {
                sliceThickness = spacing[i];
                sliceThicknessIndex = i;
            }
        }

        std::cout << "sliceThicknessIndex : " << sliceThicknessIndex << std::endl;
        std::cout << "sliceThickness : " << sliceThickness << std::endl;

        if(m_RES==2)//Anisotropic resolution
        {
            for(unsigned int i=0;i<3;i++)
            {
                if(i==sliceThicknessIndex)
                {
                    sigma[i]= spacing[i] / cst;
                }
                else
                {
                    //sigma[i]= (spacing[i] / cst);
                    sigma[i]= (1.2 * spacing[i] / cst);
                }

            }
        }

        else//Isotropic resolution
        {
            for(unsigned int i=0;i<3;i++)
            {
                //sigma[i]= (spacing[i] / cst);
                sigma[i]= (1.2*spacing[i] / cst);
            }
        }

        //sigma[0] = m_Spacing[0]/cst;
        //sigma[1] = m_Spacing[1]/cst;
        //sigma[2] = m_Spacing[2]/cst;

        //sigma[0] = sqrt(m_Spacing[0]*m_Spacing[0]/(8*log(2)));
        //sigma[1] = sqrt(m_Spacing[1]*m_Spacing[1]/(8*log(2)));
        //sigma[2] = sqrt(m_Spacing[2]*m_Spacing[2]/(8*log(2)));

        m_Gaussian -> SetSigma( sigma );

    }

    /** Sets direction of the PSF. */
    void SetDirection(DirectionType direction)
    {
        m_Direction = direction.GetVnlMatrix();
        m_idir = m_Direction.get_column(0);
        m_jdir = m_Direction.get_column(1);
        m_kdir = m_Direction.get_column(2);
    }

    /** Sets the position of the PSF. */
    void SetCenter(PointType center)
    {
        m_Center = center.GetVnlVector();
    }

    /** Sets the type of PSF (Boxcar, Gaussian). */
    itkSetMacro(PSF, unsigned int);

    /** Gets the type of PSF (Boxcar, Gaussian). */
    itkGetMacro(PSF, unsigned int);

    /** Sets the type of PSF (Boxcar, Gaussian). */
    itkSetMacro(RES, unsigned int);

    /** Gets the type of PSF (Boxcar, Gaussian). */
    itkGetMacro(RES, unsigned int);


protected:
    OrientedSpatialFunction();
    virtual ~OrientedSpatialFunction();
    void PrintSelf(std::ostream& os, Indent indent) const;

private:
    OrientedSpatialFunction(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    vnl_matrix<double> m_Direction;
    vnl_vector<double> m_Center;
    vnl_vector<double> m_Spacing;

    vnl_vector<double> m_idir;
    vnl_vector<double> m_jdir;
    vnl_vector<double> m_kdir;

    unsigned int m_PSF;

    unsigned int m_RES;

    typename GaussianFunctionType::Pointer m_Gaussian;


};

} // end namespace mialsrtk


// Define instantiation macro for this template.
#define ITK_TEMPLATE_OrientedSpatialFunction(_, EXPORT, x, y) namespace itk { \
    _(3(class EXPORT OrientedSpatialFunction< ITK_TEMPLATE_3 x >)) \
    namespace Templates { typedef OrientedSpatialFunction< ITK_TEMPLATE_3 x >\
    OrientedSpatialFunction##y; } \
    }

#if ITK_TEMPLATE_EXPLICIT
# include "Templates/itkOrientedSpatialFunction+-.h"
#endif

#if ITK_TEMPLATE_TXX
# include "mialsrtkOrientedSpatialFunction.txx"
#endif

#endif
