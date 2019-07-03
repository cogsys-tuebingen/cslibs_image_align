#ifndef UTILS_SIMD_H
#define UTILS_SIMD_H


#include <mm_malloc.h>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


// AVX is required
#include <immintrin.h>


namespace Utils_SIMD {

inline float HSumAvxFlt(const __m256 &val)
{
    float sumAVX = 0;

    __m256 hsum = _mm256_hadd_ps(val, val);

    hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, 0x1));
    _mm_store_ss(&sumAVX, _mm_hadd_ps( _mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum) ) );

    return sumAVX;
}

inline void LoopZero(__m256 *vals, const int &n)
{
    for (int x = 0; x < n; ++x) vals[x] = _mm256_set1_ps(0.0f);
}

inline void ProcessJacDiff(const int &n, const __m256 &diff, const __m256 *jvals,  __m256 *sumJ, __m256 *sumH )
{
    int ic = 0;
    for (int i = 0; i < n;++i)
    {
        sumJ[i] = _mm256_add_ps(sumJ[i],_mm256_mul_ps(jvals[i],diff));
        for (int j = 0; j <= i;++j)
        {
            sumH[ic] = _mm256_add_ps(sumH[ic],_mm256_mul_ps(jvals[i],jvals[j]));

        }
    }


}

inline void  CalcErrorSqrAVX(const cv::Mat &ref,const cv::Mat &refmask, const cv::Mat &temp, const cv::Mat &tempMask, const cv::Point2i offset, float &error, float &numPixels)
{
    int x,y;
    const float *refPtr,*refMaskPtr,*tempPtr,*tempMaskPtr;

    __m256 refA;
    __m256 refMaskA;
    __m256 tempA;
    __m256 tempMaskA;

    __m256 maskV;

    __m256 diff;

    __m256 maskDiff;


    __m256 resSqr;

    //const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    __m256 errorSum =  _mm256_set1_ps(0.0f);
    __m256 pixelSum =  _mm256_set1_ps(0.0f);


    int yStart = 0,yEnd = temp.rows,xStart = 0,xEnd = temp.cols;
    if (offset.y < 0) yStart = -offset.y;
    if (offset.x < 0) xStart = -offset.x;
    if (temp.rows+offset.y > ref.rows) yEnd = ref.rows-offset.y;
    if (temp.cols+offset.x > ref.cols) xEnd = ref.cols-offset.x;



    for(y = yStart; y < yEnd; ++y)
    {

        refPtr = ref.ptr<float>(y+offset.y)+offset.x;
        refMaskPtr = refmask.ptr<float>(y+offset.y)+offset.x;
        tempPtr = temp.ptr<float>(y);
        tempMaskPtr = tempMask.ptr<float>(y);



        for(x = xStart; x < xEnd; x+=8)
        {
            refA = _mm256_load_ps((refPtr));
            refMaskA = _mm256_load_ps((refMaskPtr));
            tempA = _mm256_load_ps((tempPtr));
            tempMaskA = _mm256_load_ps((tempMaskPtr));

            maskV = _mm256_mul_ps(refMaskA,tempMaskA);

            diff = _mm256_sub_ps(refA,tempA);

            maskDiff = _mm256_mul_ps(diff,maskV);


            //res = _mm256_and_ps(maskDiff,absmask);
            resSqr = _mm256_mul_ps(maskDiff,maskDiff);

            errorSum = _mm256_add_ps(errorSum,resSqr);
            pixelSum = _mm256_add_ps(pixelSum,maskV);

            refPtr += 8;
            refMaskPtr += 8;
            tempPtr += 8;
            tempMaskPtr += 8;

        }

    }

    numPixels = HSumAvxFlt(pixelSum);
    error = (HSumAvxFlt(errorSum))/(float) numPixels;

}


inline void  CalcErrorSqrResidualAVX(const cv::Mat &ref,const cv::Mat &refmask, const cv::Mat &temp, const cv::Mat &tempMask, cv::Mat &residualImg, const cv::Point2i offset, float &error, float &numPixels)
{
    int x,y;
    const float *refPtr,*refMaskPtr,*tempPtr,*tempMaskPtr;
    float *residualPtr;

    __m256 refA;
    __m256 refMaskA;
    __m256 tempA;
    __m256 tempMaskA;

    __m256 maskV;

    __m256 diff;

    __m256 maskDiff;


    __m256 res;
    __m256 resSqr;

    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    __m256 errorSum =  _mm256_set1_ps(0.0f);
    __m256 pixelSum =  _mm256_set1_ps(0.0f);


    int yStart = 0,yEnd = temp.rows,xStart = 0,xEnd = temp.cols;
    if (offset.y < 0) yStart = -offset.y;
    if (offset.x < 0) xStart = -offset.x;
    if (temp.rows+offset.y > ref.rows) yEnd = ref.rows-offset.y;
    if (temp.cols+offset.x > ref.cols) xEnd = ref.cols-offset.x;



    for(y = yStart; y < yEnd; ++y)
    {

        refPtr = ref.ptr<float>(y+offset.y)+offset.x;
        refMaskPtr = refmask.ptr<float>(y+offset.y)+offset.x;
        tempPtr = temp.ptr<float>(y);
        tempMaskPtr = tempMask.ptr<float>(y);
        residualPtr = residualImg.ptr<float>(y);


        for(x = xStart; x < xEnd; x+=8)
        {
            refA = _mm256_load_ps((refPtr));
            refMaskA = _mm256_load_ps((refMaskPtr));
            tempA = _mm256_load_ps((tempPtr));
            tempMaskA = _mm256_load_ps((tempMaskPtr));

            maskV = _mm256_mul_ps(refMaskA,tempMaskA);

            diff = _mm256_sub_ps(refA,tempA);

            maskDiff = _mm256_mul_ps(diff,maskV);


            res = _mm256_and_ps(maskDiff,absmask);

            _mm256_store_ps(residualPtr,maskDiff);

            resSqr = _mm256_mul_ps(res,res);

            errorSum = _mm256_add_ps(errorSum,resSqr);
            pixelSum = _mm256_add_ps(pixelSum,maskV);

            refPtr += 8;
            refMaskPtr += 8;
            tempPtr += 8;
            tempMaskPtr += 8;
            residualPtr += 8;
        }

    }

    error = HSumAvxFlt(errorSum);
    numPixels = HSumAvxFlt(pixelSum);

}


cv::Point2f GetInitialTrans(const cv::Mat &ref, const cv::Mat &refMask, const cv::Mat &tmp, const cv::Mat &tmpMask, int stepSize, int numSteps)
{

    float bestError = 9999999;
    cv::Point2i bestPos;
    float numPixels;
    float curError;

    for (int y = -stepSize*numSteps; y <= stepSize*numSteps;y+=stepSize)
    {
        for (int x = -stepSize*numSteps; x <= stepSize*numSteps;x+=stepSize)
        {
            Utils_SIMD::CalcErrorSqrAVX(ref,refMask,tmp,tmpMask,cv::Point2i(x,y),curError,numPixels);
            if (curError < bestError)
            {
                bestError = curError;
                bestPos.x = x;
                bestPos.y = y;
            }
        }
    }

    return bestPos;

}


}




#endif // UTILS_SIMD_H
