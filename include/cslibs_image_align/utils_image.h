#ifndef ALIGNED_IMAGE_PYR_H
#define ALIGNED_IMAGE_PYR_H


#include <mm_malloc.h>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>

struct ImageRegResults {
    ImageRegResults()
    {
        numParams = 3;
        Clear();
    }

    ImageRegResults(int nParams)
    {
        numParams = nParams;
        Clear();

    }

    void Update(cv::Mat pars)
    {
        delta = 0;
        for (int i = 0; i < pars.rows;++i)
        {
            delta += std::abs(pars.at<double>(i,0));
            deltaParams.at<double>(i,0) = pars.at<double>(i,0);

        }
        params = params+deltaParams;
        iterations++;

    }

    void Clear()
    {
        delta = 1e10;
        error = 1e10;
        iterations = 0;
        pixels = 1e10;

        params = cv::Mat::zeros(numParams,1,CV_64F);
        deltaParams = cv::Mat::zeros(numParams,1,CV_64F);
        if (numParams == 8) warpMat = cv::Mat::zeros(3,3,CV_64F);
        else warpMat = cv::Mat::zeros(2,3,CV_64F);

    }

    void ResetStats()
    {
        delta = 1e10;
        error = 1e10;
        pixels = 1e10;

    }

    bool TestResults(cv::TermCriteria &termCrit)
    {
        if (delta < termCrit.epsilon) return false;
        if (iterations > termCrit.maxCount) return false;
        if (pixels == 0) return false;
        return true;
    }

    cv::Mat params;
    cv::Mat deltaParams;
    cv::Mat warpMat;
    float error;   ///< final total squared error
    double delta;   ///< norm of the last update
    float pixels;     ///< common pixels in last iteration
    int iterations; ///< number of iterations performed

    int numParams;

};



void Display32FImage(std::string wname, cv::Mat image, float imin, float imax)
{
    cv::namedWindow(wname,0);
    cv::Mat displayImg;


    switch (image.channels())
    {
    case 1:
        image.convertTo(displayImg,CV_8U,256.0/(double)(imax-imin),-imin*256.0/(double)(imax-imin));
        break;
    case 3:
        image.convertTo(displayImg,CV_8UC3,256.0/(double)(imax-imin),-imin*256.0/(double)(imax-imin));
        break;
    case 4:
        image.convertTo(displayImg,CV_8UC4,256.0/(double)(imax-imin),-imin*256.0/(double)(imax-imin));

    default:
         displayImg = cv::Mat::zeros(16,16,CV_8U);
    }

    cv::imshow(wname,displayImg);

}

void Display32FImage(std::string wname, cv::Mat &image)
{
    cv::namedWindow(wname,0);
    double imin,imax;
    int minIdx,maxIdx;
    cv::minMaxIdx(image,&imin,&imax,&minIdx,&maxIdx);

    cv::Mat displayImg;
    switch (image.channels())
    {
    case 1:
        image.convertTo(displayImg,CV_8U,256.0/(double)(imax-imin),-imin*256.0/(double)(imax-imin));
        break;
    case 3:
        image.convertTo(displayImg,CV_8UC3,256.0/(double)(imax-imin),-imin*256.0/(double)(imax-imin));
        break;
    case 4:
        image.convertTo(displayImg,CV_8UC4,256.0/(double)(imax-imin),-imin*256.0/(double)(imax-imin));

    default:
         displayImg = cv::Mat::zeros(16,16,CV_8U);
    }
    cv::imshow(wname,displayImg);

}




class ExpProc
{

/*    matrix exponential tool class
 *
 *  The function is taken from https://bitbucket.org/rricha1/naya/src/master/
    Copyright (c) Rogerio Richa
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
public:

    ExpProc()
    {
        //int type = CV_64F;
        update_auxA.create(3, 3, CV_64F);
        update_auxH.create(3, 3, CV_64F);
        aux1.create(3, 3, CV_64F);
        aux2.create(3, 3, CV_64F);
        aux3.create(3, 3, CV_64F);
        aux4.create(3, 3, CV_64F);
        aux5.create(3, 3, CV_64F);

    }

    cv::Mat update_auxA;
    cv::Mat update_auxH;
    cv::Mat aux1;
    cv::Mat aux2;
    cv::Mat aux3;
    cv::Mat aux4;
    cv::Mat aux5;



    double	Max(cv::Mat *M)
    {
        int i,j;
        double maximum = 0, sum;

        for ( i = 0; i < M->rows; i++)
        {
            sum = 0;
            for ( j = 0; j < M->cols; j++)
            {
                sum += M->at<double>(i, j);
            }
            if (sum < 0)
            {
                sum = sum * -1;
            }
            if (sum > maximum)
            {
                maximum = sum;
                sum = 0;
            }
        }
        return maximum;
    }

    void Expm(cv::Mat *input)
    {
        int e;

        double n = Max(input);
        double f = frexp(n, &e);
        double s = (e + 1) > 0 ? (e + 1) : 0;

        aux1.setTo(pow(2,s));
        cv::divide(*input, aux1, aux3);

        // Pade approx
        double c = 0.5;
        aux2.setTo(c);
        cv::multiply(aux3, aux2, aux1);

        aux2 = aux2.eye(3,3,CV_64F);

        aux4 = aux2 + aux1;
        aux5 = aux2 - aux1;

        double q = 6;
        char tf = 1;

        for (int k = 2; k < q; k++)
        {
            c = c * (q-k + 1) / (k*(2*q-k+1));
            aux1 = aux3*aux3;
            aux1.copyTo(aux3);
            aux1.setTo(c);
            cv::multiply(aux3, aux1, aux2);
            aux1 = aux2 + aux4;
            aux1.copyTo(aux4);

            if (tf)
            {
                aux1 = aux5 + aux2;
                aux1.copyTo(aux5);
                tf = 0;
            }
            else
            {
                aux1 = aux5-aux2;
                aux1.copyTo(aux5);
                tf = 1;
            }
        }

        cv::invert(aux5, aux1, CV_LU);
        aux1.copyTo(aux5);

        aux1 = aux5*aux4;
        aux1.copyTo(aux4);

        // undoing scaling
        for (int k = 1; k <= s; k++)
        {
            aux4.copyTo(aux1);
            aux2 = aux4*aux1;
            aux2.copyTo(aux4);
        }

        aux4.copyTo(*input);

    }

};

#define AlignedMat_Alignment 32

class AlignedMat
{

public:
    typedef std::shared_ptr<AlignedMat > ptr;

    static AlignedMat::ptr Create(int width, int height, int cvType){ return std::make_shared< AlignedMat >(width,height,cvType) ; }
    static AlignedMat::ptr Create(cv::Size imgSize, int cvType){ return std::make_shared< AlignedMat >(imgSize,cvType) ; }
    static AlignedMat::ptr Create(cv::Mat input){ return std::make_shared< AlignedMat >(input) ; }

    AlignedMat(int width, int height, int cvType)
    {
        Allocate(width,height,cvType);
    }

    AlignedMat(cv::Size imgSize, int cvType)
    {
        Allocate(imgSize.width,imgSize.height,cvType);
    }

    AlignedMat(const cv::Mat &input)
    {
        Allocate(input.cols,input.rows,input.type());
        int ByteSize = GetPixelSizeForCVTypes(input.type());

        for (int y = 0; y < input.rows;y++)
        {
            memcpy(mat_.ptr(y),input.ptr(y),input.cols*ByteSize);
        }
    }

    void Allocate(int width, int height, int cvType)
    {
        int ByteSize = GetPixelSizeForCVTypes(cvType);
        int allocateWidth = width;
        if ((allocateWidth*ByteSize) % AlignedMat_Alignment != 0)
        {
            allocateWidth += (AlignedMat_Alignment - (width*ByteSize)% AlignedMat_Alignment)/ByteSize;
        }
        void* p =(void*) _mm_malloc (allocateWidth*height*ByteSize, AlignedMat_Alignment);
        memset(p,0,allocateWidth*height*ByteSize);
        cv::Mat resMat(cv::Size(width,height),cvType,p,allocateWidth*ByteSize);
        mat_ = resMat;
    }

    void SetZero()
    {
        memset(mat_.data,0,mat_.step*mat_.rows);
    }

    void CopyDataFrom(const cv::Mat &input)
    {
        memset(mat_.data,0,mat_.step*mat_.rows);
        int ByteSize = GetPixelSizeForCVTypes(input.type());

        for (int y = 0; y < input.rows;y++)
        {
            memcpy(mat_.ptr(y),input.ptr(y),input.cols*ByteSize);
        }
    }

    void CopyDataFrom(const void* input)
    {
        int ByteSize = GetPixelSizeForCVTypes(mat_.type());


        memcpy(mat_.ptr(),input,mat_.step*ByteSize*mat_.rows);

    }


    ~AlignedMat(){
        _mm_free((void*)mat_.data);
        //std::cout << "destroyMat!!" << std::endl;
    }
    cv::Mat mat_;

    int GetPixelSizeForCVTypes(int cvType)
    {
    switch(cvType){
    case CV_8U  :
        return 1;
        break; //optional
    case CV_8UC3  :
        return 3;
        break; //optional
    case CV_8UC4  :
        return 4;
        break; //optional

    case CV_16S  :
        return 2;
        break; //optional
    case CV_16SC3  :
        return 6;
        break; //optional
    case CV_16SC4  :
        return 8;
        break; //optional

    case CV_16U  :
        return 2;
        break; //optional
    case CV_16UC3  :
        return 6;
        break; //optional
    case CV_16UC4  :
        return 8;
        break; //optional


    case CV_32F  :
        return 4;
        break; //optional
    case CV_32FC3  :
        return 12;
        break; //optional
    case CV_32FC4  :
        return 16;
        break; //optional

    case CV_64F  :
        return 8;
        break; //optional
    case CV_64FC3  :
        return 24;
        break; //optional
    case CV_64FC4  :
        return 32;
        break; //optional
    }
    return 1;

    }

};

class AlignedImagePyr
{

public:
    enum PYR_TYPES { LINEAR, GAUSS, MASK};


    typedef std::shared_ptr<AlignedImagePyr > ptr;

    static AlignedImagePyr::ptr Create(cv::Mat input, int numLevels, int pyrType){ return std::make_shared< AlignedImagePyr >(input,numLevels,pyrType) ; }

    AlignedImagePyr(cv::Mat input, int numLevels, int pyrType)
    {
        switch (pyrType) {
                case GAUSS: CreateGaussPyramid(input,numLevels); break;
                case MASK: CreateMaskPyramid(input,numLevels); break;
                case LINEAR: CreatePyramid(input,numLevels); break;
                default: CreatePyramid(input,numLevels); //there are no applicable constant_expressions
                                           //therefore default is executed
            }
    }



    void CreatePyramid(cv::Mat img, int levels)
    {
        levels_.clear();
        levels_.push_back(AlignedMat::Create(img));

        for (int tl = 1; tl < levels;tl++)
        {
            cv::Mat timg;

            cv::resize(levels_[tl-1]->mat_,timg,cv::Size(levels_[tl-1]->mat_.cols/2, levels_[tl-1]->mat_.rows/2));
            levels_.push_back(AlignedMat::Create(timg));

        }

    }

    void CreateMaskPyramid(cv::Mat img, int levels)
    {
        levels_.clear();
        levels_.push_back(AlignedMat::Create(img));

        for (int tl = 1; tl < levels;tl++)
        {
            cv::Mat timg;
            cv::Mat threshImg;

            cv::resize(levels_[tl-1]->mat_,timg,cv::Size(levels_[tl-1]->mat_.cols/2, levels_[tl-1]->mat_.rows/2));
            cv::threshold(timg,threshImg,0.9999,1.0,CV_THRESH_BINARY);
            levels_.push_back(AlignedMat::Create(threshImg));

        }

    }


    void CreateGaussPyramid(cv::Mat img, int levels)
    {
        levels_.clear();
        levels_.push_back(AlignedMat::Create(img));

        for (int tl = 1; tl < levels;tl++)
        {
            cv::Mat timg;

            cv::pyrDown(levels_[tl-1]->mat_,timg,cv::Size(levels_[tl-1]->mat_.cols/2, levels_[tl-1]->mat_.rows/2));
            levels_.push_back(AlignedMat::Create(timg));

        }

    }

    AlignedMat::ptr Get(int i)
    {
        return levels_[i];
    }

    const AlignedMat::ptr GetConst(int i) const
    {
        return levels_[i];
    }

    AlignedMat::ptr GetInv(int i)
    {
        return levels_[levels_.size()-1-i];
    }

    std::vector<AlignedMat::ptr > levels_;

};



#endif //CV_ALIGNED_MAT_H
