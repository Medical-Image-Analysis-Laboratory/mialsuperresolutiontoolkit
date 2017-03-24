/*=========================================================================

Program: Maths library (header)
Language: C++
Date: $Date: 2013-10-17 $
Version: $Revision: 1.0 $
Author: $Sebastien Tourbier$

==========================================================================*/

/* Standard includes */
#include <tclap/CmdLine.h>
#include <iostream>     // std::cout
#include <sstream>
#include <limits>       // std::numeric_limits
#include <algorithm>    // std::sort
#include <vector>
#include <cmath> // std::abs

#include "itkImage.h"
#include "itkPoint.h"

/************************************************************************/

typedef std::vector<float>::iterator floatIter;
typedef std::vector<double>::iterator doubleIter;


std::string int2str(int a)
{
    std::string str;
    std::ostringstream temp;
    temp<<a;
    return temp.str();
};

double mialtkMean(std::vector<float> &data)
{
    double sum=0.0;
    int counter=0;

    for(floatIter it = data.begin();it != data.end();++it)
    {
        if(*it>=0.0)
        {
            sum+=(double)*it;
            counter++;
        }

    }
    if(counter>0)
        return sum/(double)counter;
    else
        return 0.0;
};

double mialtkVariance(std::vector<float> &data)
{
    double mean = mialtkMean(data);
    double sum=0.0;
    int counter=0;

    for(floatIter it = data.begin();it != data.end();++it)
    {
        if(*it>=0.0)
        {
            sum+=(*it-mean)*(*it-mean);
            counter++;
        }

    }

    if(counter>0)
        return sum/(double)counter;
    else
        return 0.0;
};

double mialtkMedian(std::vector<double> data)
{
    double median = 0.0;
    std::size_t n = data.size() / 2;
    //nth_element (O(nlogn)) is faster than a simpler sort (O(n))
    std::nth_element(data.begin(),data.begin()+n,data.end());
    median = data[n];
    return median;
};

/**
 * Return the entropy as computed in Matlab (E=-sum(p * log(p)))
 */
double mialtkEntropy(std::vector<double> data, int nbins, double min, double max)
{
    double entropy = 0.0;
    std::size_t n = data.size();

    std::vector<int> histogram(nbins);
    double bin_width = (max - min) / (double)nbins;// Used for linear mapping in histogram bins

    //std::cout << "Histogram settings" << std::endl;
    //std::cout << "Min/Max" << min << "/" << max << std::endl;
    //std::cout << "Number of Bins : " << nbins << std::endl;
    //std::cout << "Bin width : " << bin_width << std::endl;

    //Initialize the vector
    for(int i=0; i<nbins; i++)
    {
        histogram[i]=0;
    }

    //Compute histogram
    for(int i=0; i<n; i++)
    {
        int bin_idx = (int)((data[i] - min ) / bin_width);// Linear mapping
        histogram[bin_idx]++;
    }

    //Print histogram for debug
    //for(int i=0; i<nbins; i++)
    //{
    //  std::cout << histogram[i] << ",";
    //}

    //Compute entropy
    double frequency = 0.0;
    for(int i=0; i<nbins; i++)
    {
        frequency = (double)histogram[i] / (double)n;

        if(frequency != 0.0)
            entropy-= frequency * (vcl_log(frequency) / vcl_log(2.0));
    }

    return entropy;
};

/**
 * Return the entropy as computed in Aksoy et al. 2012 (E=-sum(Ip/Itotal * ln(Ip/Itotal)))
 */
double mialtkEntropy2(std::vector<double> data)
{
    double entropy = 0.0;

    doubleIter it;

    // Compute Itotal (Itotal = sqrt(sum(Ip^2)))
    double i_total = 0.0;
    for(it = data.begin(); it != data.end(); ++it)
    {
        i_total+= (*it) * (*it);
    }
    i_total = sqrt(i_total);

    //std::cout << "Itotal = " << i_total << std::endl;

    //Compute the entropy (E=-sum(Ip/Itotal * ln(Ip/Itotal)))
    double temp = 0.0;
    for(it = data.begin(); it != data.end(); ++it)
    {
        if(*it != 0)
        {
            temp = *it / i_total;
            entropy -= temp * (vcl_log(temp) / vcl_log(vcl_exp(1.0)));
        }
    }

    return entropy;
};

/**
 * Return the entropy as computed in McGee 2000 (E=-sum(Abs(Ip))/Itotal * ln(Abs(Ip)/Itotal)))
 */
double mialtkEntropy3(std::vector<double> data)
{
    double entropy = 0.0;

    doubleIter it;

    // Compute Itotal (Itotal = sqrt(sum(Ip^2)))
    double i_total = 0.0;
    for(it = data.begin(); it != data.end(); ++it)
    {
        i_total+= vcl_abs(*it);
    }

    //std::cout << "Itotal = " << i_total << std::endl;

    //Compute the entropy (E=-sum(Ip/Itotal * ln(Ip/Itotal)))
    double temp = 0.0;
    for(it = data.begin(); it != data.end(); ++it)
    {
        if(*it != 0)
        {
            temp = vcl_abs(*it) / i_total;
            entropy -= temp * (vcl_log(temp) / vcl_log(vcl_exp(1.0)));
        }
    }

    return entropy;
};

double mialtkNormalizedCorrelationCoef(std::vector<float> &target, std::vector<float> &temp)
{
    double normCorrCoef = 0.0;

    double meanTarget = mialtkMean(target);
    double meanTemplate = mialtkMean(temp);

    double stdTarget = sqrt(mialtkVariance(target));
    double stdTemplate = sqrt(mialtkVariance(temp));

    floatIter targetIt;
    floatIter templateIt;

    for(targetIt = target.begin() , templateIt = temp.begin() ; targetIt != target.end(); ++targetIt , ++templateIt)
    {
        normCorrCoef+= ( *targetIt - meanTarget )*( *templateIt - meanTemplate );
    }

    //std::cout << 'Sum squared: ' << normCorrCoef << " , target ( mu=" << meanTarget << ", std=" << stdTarget << ") , template ( mu=" << meanTemplate << ", std=" << stdTemplate << ")" << std::endl;

    if( (stdTarget != 0.0) && (stdTemplate != 0.0) )
    {
        return normCorrCoef/((double)target.size()*stdTarget*stdTemplate);
    }
    else
    {
        // XXX GORTHI Modified
        if( std::fabs(stdTarget-stdTemplate) < 1e-6 )
            return 1.0; // maximum correlation
        else
            return 0.0; // no correlation
    }

};


/**
 * Return the SNR between the current image reconstructed x and a given reference image x_ref. x_ref is typically a ground truth image
 */
double mialtkComputeSNR(const vnl_vector<float>& x_ref, const vnl_vector<float>& x , float level = 0.0)
{
    //Compute Var(x)
    double mean = 0.0;
    double var = 0.0;
    double sum_sq = 0.0;

    int count1 = 0;

    for (int i=0;i<x_ref.size();i++)
    {
        if(x_ref[i] > level)
        {
            mean += x_ref[i];
            sum_sq += x_ref[i]*x_ref[i];
            count1++;
        }
    }

    mean/=count1;
    sum_sq/=count1;
    var = sum_sq - mean * mean;

    //Compute Var(xref - x)
    double mean_diff = 0.0;
    double var_diff = 0.0;
    double sum_sq_diff = 0.0;

    vnl_vector<float> x_diff;
    x_diff.set_size(x.size());
    x_diff = x - x_ref;

    int count2 = 0;

    for (int i=0;i<x.size();i++)
    {
        if( x_ref[i] > level )
        {
            mean_diff += x_diff[i];
            sum_sq_diff += x_diff[i]*x_diff[i];
            count2++;
        }
    }

    mean_diff/=count2;
    sum_sq_diff/=count2;
    var_diff = sum_sq_diff - mean_diff * mean_diff;

    double snr = 10 * log10( var / var_diff );
    return snr;
};

/**
 * Return the PSNR between the current image reconstructed x and a given reference image x_ref. x_ref is typically a ground truth image
 */
double mialtkComputePSNR(const vnl_vector<float>& x_ref, const vnl_vector<float>& x , float level = 0.0)
{
    double dyn = 0.0;
    double mse = 0.0;
    double psnr = 0.0;

    //Compute dynamic range of reference image
    dyn = x_ref.max_value() - x_ref.min_value();

    //Compute MSE between reference image x and current image m_x

    vnl_vector<float> x_diff;
    x_diff.set_size(x.size());
    x_diff = x - x_ref;

    int count2 = 0;

    for (int i=0;i<x.size();i++)
    {
        if( x_ref[i] > level)
        {
            mse += x_diff[i]*x_diff[i];
            count2++;
        }
    }

    mse/=count2;

    //Compute the PSNR value
    psnr = 10 * log10( ( dyn * dyn ) / mse );

    std::cout << "Pixels in ref : " << x_ref.size() << std::endl;
    std::cout << "Pixels in x : " << x.size() << std::endl;
    std::cout << "Pixels in mask : " << count2 << std::endl;

    return psnr;
}

/**
 * Return the MSE between the current image reconstructed x and a given reference image x_ref. x_ref is typically a ground truth image
 */
double mialtkComputeMSE(const vnl_vector<float>& x_ref, const vnl_vector<float>& x , float level = 0.0)
{
    //Compute MSE between reference image x and current image m_x
    double mse = 0.0;

    vnl_vector<float> x_diff;
    x_diff.set_size(x.size());
    x_diff = x - x_ref;

    int count = 0;

    for (int i=0;i<x.size();i++)
    {
        if( x_ref[i] > level)
        {
            mse += x_diff[i]*x_diff[i];
            count++;
        }
    }

    mse/=count;
    return mse;
}

/**
 * Return the Mean Absolute Error (MAE) between the current image reconstructed x and a given reference image x_ref. x_ref is typically a ground truth image
 */
double mialtkComputeMAE(const vnl_vector<float>& x_ref, const vnl_vector<float>& x , float level = 0.0)
{
    //Compute MAE  between reference image x and current image m_x
    double mae = 0.0;

    vnl_vector<float> x_diff;
    x_diff.set_size(x.size());
    x_diff = x - x_ref;

    int count = 0;

    for (int i=0;i<x.size();i++)
    {
        if( x_ref[i] > level)
        {
            mae += std::abs(x_diff[i]);
            count++;
        }
    }

    mae/=count;
    return mae;
}

/**
 * Return the Mean Absolute Error (MAE) between the current image reconstructed x and a given reference image x_ref. x_ref is typically a ground truth image
 */
float mialtkComputeMAE(vnl_vector<float>& x_ref, vnl_vector<float>& x , float level = 0.0)
{
    //Compute MAE  between reference image x and current image m_x
    float mae = 0.0;

    vnl_vector<float> x_diff;
    x_diff.set_size(x.size());
    x_diff = x - x_ref;

    int count = 0;

    for (int i=0;i<x.size();i++)
    {
        if( x_ref[i] > level)
        {
            mae += std::abs(x_diff[i]);
            count++;
        }
    }

    mae/=count;
    return mae;
}


/**
 * Return the Normalized Mutual Information (NMI) between the current image reconstructed y and a given reference image x. x is typically a ground truth image
 */
float mialtkComputeNMI(vnl_vector<float> x,  vnl_vector<float> y , float nbins = 32)
{
    //Rescale x and y between 0 and binSize-1
    x = ((x - x.min_value())/x.max_value()) * (nbins-1.0);
    y = ((y - y.min_value())/y.max_value()) * (nbins-1.0);

    //Iterators over x and y
    vnl_vector<float>::iterator itX;
    vnl_vector<float>::iterator itY;

    //std::cout << "x size : " << x.size() << " , y size : " << y.size() << std::endl;

    //Compute the joint histogram
    vnl_matrix<int> hist_xy(nbins,nbins);

    for(itX = x.begin(); itX != x.end(); ++itX)
    {
        //if(*(itX)>0)
        //{
            for(itY= y.begin(); itY != y.end(); ++itY)
            {
                //if(*(itY)>0) continue;

                float value_x = *(itX);
                float value_y = *(itY);

                value_x = floor(value_x);
                value_y = floor(value_y);

                hist_xy((int)value_x,(int)value_y)++;
            }
        //}
    }

    //Compute histograms of x and y
    vnl_vector<int> hist_x;
    hist_x.set_size(nbins);

    vnl_vector<int> hist_y;
    hist_y.set_size(nbins);

    int cnt = 0;

    for(int i = 1; i < nbins; i++)
    {
        for(int j = 1; j < nbins; j++)
        {
            hist_x[i] += hist_xy(i,j);
            hist_x[j] += hist_xy(i,j);
            cnt += hist_xy(i,j);
        }
    }

    //Compute entropies and the joint entropy
    float H_x = 0.0 , H_y = 0.0, H_xy = 0.0;

    for(int i = 1; i < nbins; i++)//Ignore zeros value (template is already mask, i.e. a lot of zeros against target image with few or no zeros)
    {
        if(hist_x[i]>0)
        {
            float p = float(hist_x[i]) / cnt;
            H_x -= p * log(p);
        }

        if(hist_y[i]>0)
        {
            float p = float(hist_y[i]) / cnt;
            H_y -= p * log(p);
        }

        for(int j = 1; j < nbins; j++)
        {

            if(hist_xy(i,j)>0)
            {
                float p = float(hist_xy(i,j)) / cnt;
                H_xy -= p * log(p);
            }
        }
    }

    float nmi = (H_x + H_y) / H_xy;

    //std::cout << "NMI : " << nmi << ", H_x: " << H_x << ", H_y: " << H_y <<  ", H_xy: " << H_xy << std::endl;
    return nmi;


}

/**
 * Return the Entropy of the squared differences between the current image reconstructed y and a given reference image x. x is typically a ground truth image
 */
float mialtkComputeESD(vnl_vector<float> x,  vnl_vector<float> y , float nbins = 32)
{
    vnl_vector<float> diff;
    diff.set_size(x.size());
    diff = y - x;
    diff = element_product(diff,diff);

    //Rescale diff between 0 and binSize-1
    diff = ((diff - diff.min_value())/diff.max_value()) * (nbins-1.0);

    //Iterators over x and diff
    vnl_vector<float>::iterator itDiff;
    vnl_vector<float>::iterator itX;

    //Compute the histogram of the image squared difference
    vnl_vector<int> hist_diff;
    hist_diff.set_size(nbins);

    int cnt = 0;

    for(itX = x.begin(), itDiff = diff.begin(); itX != x.end(); ++itX, ++itDiff)
    {
        if(*(itX)>0)
        {
            int value = *(itDiff);
            value = floor(value);
            hist_diff[value]++;
            cnt++;
        }
    }

    //Compute entropy
    float H_diff = 0.0;

    for(int i = 0; i < nbins; i++)
    {
        if(hist_diff[i]>0)
        {
            float p = float(hist_diff[i]) / cnt;
            H_diff -= p * log(p);
        }
    }

    return H_diff;
}


/**
 * Return the Normalized Cross Correlation coefficient between the current image reconstructed y and a given reference image x. x is typically a ground truth image
 */
float mialtkComputeNCC(vnl_vector<float> &x,  vnl_vector<float> &y, float thresh = 1e-1)
{
    /*
    vnl_vector<float>::iterator itX;
    vnl_vector<float>::iterator itY;

    int counter = 0;
    for(itX=x.begin(); itX!=x.end(); ++itX)
    {
        if(*(itX)>thresh) counter++;
    }

   // std::cout << "counter : " << counter << std::endl;

    vnl_vector<float> x_new;
    x_new.set_size(counter);

    vnl_vector<float> y_new;
    y_new.set_size(counter);

    counter = 0;
    for(itX=x.begin(), itY=y.begin(); itX != x.end(); ++itX, ++itY)
    {
        if(*(itX)>thresh)
        {
            x_new[counter] = *(itX);
            y_new[counter] = *(itY);
            counter++;
        }
    }


    vnl_vector<float> x_new_zeromean = x_new-x_new.mean();
    float mag_x = x_new_zeromean.magnitude();
    //if(mag < 1.0) continue;

    vnl_vector<float> nx_new_zeromean = x_new_zeromean/mag_x;

    vnl_vector<float> y_new_zeromean = y_new-y_new.mean();
    float mag_y = y_new_zeromean.magnitude();
    //if(mag < 1.0) continue;

    vnl_vector<float> ny_new_zeromean = y_new_zeromean/mag_y;

    //float varX = nx_new_zeromean.squared_magnitude() / nx_new_zeromean.size();
    //float varY = ny_new_zeromean.squared_magnitude() / ny_new_zeromean.size();

    //Normalise w.r.t standard deviation -> Inner product is then equivalent to computing the normalized cross correlation
    //nx_new_zeromean = nx_new_zeromean / sqrt(varX);
    //ny_new_zeromean = ny_new_zeromean / sqrt(varY);

    return inner_product(nx_new_zeromean, ny_new_zeromean) / ny_new_zeromean.size();
    */

    vnl_vector<float> x_zeromean = x-x.mean();
    float mag_x = x_zeromean.magnitude();

    //std::cout << "Mag x : " << mag_x << std::endl;

    //if(mag < 1.0) continue;

    vnl_vector<float> nx_zeromean = x_zeromean/mag_x;

    vnl_vector<float> y_zeromean = y-y.mean();
    float mag_y = y_zeromean.magnitude();

    //std::cout << "Mag y : " << mag_y << std::endl;

    //if(mag < 1.0) continue;

    vnl_vector<float> ny_zeromean = y_zeromean/mag_y;

    return inner_product(nx_zeromean, ny_zeromean) / ny_zeromean.size();

}
