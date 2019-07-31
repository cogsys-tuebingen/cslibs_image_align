#ifndef IMAGEREGBASE_H
#define IMAGEREGBASE_H

#include <cslibs_image_align/utils_image.h>
#include <cslibs_image_align/utils_simd.h>
#include <iostream>
#include <opencv2/video/video.hpp>




/*!
 * \brief The image registration interface
 */
class IImageReg
{
public:

    typedef std::shared_ptr<IImageReg > ptr;

    virtual void AlignImage(const cv::Mat &refImg, const cv::Mat &refMask, const cv::Mat &tmpImg, const cv::Mat &tmpMask,ImageRegResults &result) = 0;
    virtual void Setup(cv::Size refImgSize, cv::Size tmpImgSize, int imageType, int numLevels) = 0;
    virtual void SetTermCrit(cv::TermCriteria termCrit) = 0;
    virtual ImageRegResults CreateResult() = 0;
    virtual void SetStepFactor(float factor) = 0;

};


/*!
 *  Base Class containing matrix operations
 */
template <int N>
class RegProc
{
public:

    RegProc()
    {

    }

    /// Number of parameters and number of entries of upper diagonal matrix
    static constexpr int numParams = N;
    static constexpr int numParamsSq = (N*N+N)/2;


    /// Calculate ATA and Ad from jacobian and difference
    static inline void CalcMat(const __m256 (&mJVals)[numParams], const __m256& diff,__m256 (&sumHesRows)[numParamsSq], __m256 (&sumJacDifs)[numParams])
    {
        int c = 0;

        for (int i = 0; i < N;++i)
        {
            for (int j = i; j < N;++j)
            {
                sumHesRows[c] = _mm256_add_ps(sumHesRows[c],_mm256_mul_ps( mJVals[i], mJVals[j]));
                ++c;
            }
        }


        for (int i = 0; i < N;++i)
        {
            sumJacDifs[i] = _mm256_add_ps(sumJacDifs[i],_mm256_mul_ps(mJVals[i],diff));
        }

    }

    /// Sum avx vectors to corresponding matrix entries
    static inline void WriteToMatrix(const __m256* sumHesRows,const __m256* sumJacDifs, cv::Mat &jac, cv::Mat &hess,const float &numPixels)
    {
        float *jacDifPtr= jac.ptr<float>(0);

        float *hesRowPtr[N];

        for (int i = 0; i < N;++i)
        {
            hesRowPtr[i] = hess.ptr<float>(i);

        }

        /// Jacobian*Residual matrix
        for (int i = 0; i < N;++i)
        {
            jacDifPtr[i] = Utils_SIMD::HSumAvxFlt(sumJacDifs[i]);

        }

        /// upper triangle of J^TJ
        int c = 0;
        for (int i = 0; i < N;++i)
        {
            for (int j = i; j < N;++j)
            {
                hesRowPtr[i][j] =  Utils_SIMD::HSumAvxFlt(sumHesRows[c]);
                ++c;
            }
        }

        /// lower triangle
        for (int i = 1; i < N;++i)
        {
            for (int j = 0; j < i;++j)
            {
                hesRowPtr[i][j] =  hesRowPtr[j][i];
                ++c;
            }
        }

    }


    static inline void LoopZero(__m256 *vals, int n)
    {
        for (int x = 0; x < n; ++x) vals[x] = _mm256_set1_ps(0.0f);
    }


    static inline bool TestNotZero(const __m256 &val)
    {
        return _mm256_movemask_ps(_mm256_cmp_ps(val, _mm256_setzero_ps(), _CMP_EQ_OQ)) != 0xff;
    }


    static ImageRegResults CreateResult()
    {
        return ImageRegResults(N);
    }

    static cv::Mat CreateHessF()
    {
        return cv::Mat::zeros(N,N,CV_32F);
    }
    static cv::Mat CreateHessD()
    {
        return cv::Mat::zeros(N,N,CV_64F);
    }
    static cv::Mat CreateJacF()
    {
        return cv::Mat::zeros(N,1,CV_32F);
    }
    static cv::Mat CreateJacD()
    {
        return cv::Mat::zeros(N,1,CV_64F);
    }

    /// For pyramide proc only
    static void ParamScaleUp(cv::Mat &p)
    {
        p.at<double>(0,0) = p.at<double>(0,0)*2.0;
        p.at<double>(1,0) = p.at<double>(1,0)*2.0;

    }

    static void ParamScale(cv::Mat &p, float scale)
    {
        p.at<double>(0,0) = p.at<double>(0,0)*scale;
        p.at<double>(1,0) = p.at<double>(1,0)*scale;

    }

    static int GetNumParams() { return numParams;}
    static int GetNumParamsSq() { return numParamsSq;}

};


/*!
 * \brief Utility class for creating Euclidean Warp matrix
 */
class EuclidWarp
{
public:
    static cv::Mat CreateMat(const cv::Mat &params)
    {
        cv::Mat rot_mat(2,3,CV_64F);
        rot_mat.at<double>(0,2) = params.at<double>(0,0);
        rot_mat.at<double>(1,2) = params.at<double>(1,0);
        rot_mat.at<double>(0,0) = std::cos(params.at<double>(2,0));
        rot_mat.at<double>(1,1) = std::cos(params.at<double>(2,0));
        rot_mat.at<double>(1,0) = std::sin(params.at<double>(2,0));
        rot_mat.at<double>(0,1) = -std::sin(params.at<double>(2,0));
        return rot_mat;
    }

};
/*!
 * \brief Utility class for creating Homography Warp matrix
 */
class HOMWarp
{
public:
    cv::Mat CreateMat(const cv::Mat &params)
    {
        cv::Mat G = cv::Mat::zeros(3,3,CV_64F);

        G.at<double>(2,1) = params.at<double>(7,0);;
        G.at<double>(2,0) = params.at<double>(6,0);;
        G.at<double>(0,1) = params.at<double>(5,0);;
        G.at<double>(1,0) = params.at<double>(5,0);;
        G.at<double>(0,0) = params.at<double>(4,0);;
        G.at<double>(1,1) = -params.at<double>(4,0);;
        G.at<double>(0,0) += params.at<double>(3,0);;
        G.at<double>(1,1) += params.at<double>(3,0);;
        G.at<double>(2,2) += -2*params.at<double>(3,0);;
        G.at<double>(0,1) -= params.at<double>(2,0);;
        G.at<double>(1,0) += params.at<double>(2,0);;
        G.at<double>(1,2) = params.at<double>(1,0);;
        G.at<double>(0,2) = params.at<double>(0,0);;

        expProc.Expm(&G);

        return G;
    }

    ExpProc expProc;


};

/*!
 * \brief ESM with euclidean warp
 */
class ESM_Euclidean : public RegProc<3>, public EuclidWarp
{
public:

    /// gradient scaling for ESM 1/4 for sobel gradient times 0.5
    static constexpr float gradMultiplier = 0.125f;
    /// Step size, larger values for faster convergence, but too large values prohibit proper convergence
    static constexpr double stepFactor = 2.0;
    /// Uses ESM method
    static constexpr bool useESMJac = true;


    /// Create the jacobian matrix
    static inline void AddJacobians(__m256* mJVals , const __m256 &mPosX, const __m256 &mPosY, const __m256 &refGradX, const __m256 &refGradY, const __m256 &tempGradX, const __m256 &tempGradY, const __m256 &refData, const __m256 &tempData, const __m256 &mask, const __m256 &gradMul)
    {
        const __m256 maskGradMul = _mm256_mul_ps(mask,gradMul);

        /// jacobian x direction
        mJVals[0] = _mm256_mul_ps(_mm256_add_ps(refGradX,tempGradX),maskGradMul);

        /// jacobian y direction
        mJVals[1] = _mm256_mul_ps(_mm256_add_ps(refGradY,tempGradY),maskGradMul);

        /// jacobian rotation
        mJVals[2] = _mm256_sub_ps(_mm256_mul_ps(mPosX,mJVals[1]),_mm256_mul_ps(mPosY,mJVals[0]));




    }

    /// Warp image data, includes gradients for ESM
    static cv::Mat DoWarp(const cv::Mat &tmpImg, const cv::Mat &tmpMask, const cv::Mat &tmpGradX, const cv::Mat &tmpGradY, const cv::Mat &params,
                          const cv::Mat &wtmpImg, const cv::Mat &wtmpMask, const cv::Mat &wtmpGradX, const cv::Mat &wtmpGradY)
    {
        cv::Mat rot_mat = CreateMat(params);

        cv::warpAffine(tmpImg,wtmpImg,rot_mat,wtmpImg.size());
        cv::warpAffine(tmpGradX,wtmpGradX,rot_mat,wtmpImg.size());
        cv::warpAffine(tmpGradY,wtmpGradY,rot_mat,wtmpImg.size());

        cv::warpAffine(tmpMask,wtmpMask,rot_mat,wtmpImg.size(),cv::INTER_NEAREST);


        return rot_mat;
    }

    /// Calculate the warp parameters
    static cv::Mat toDeltaPars(const cv::Mat &paramUpdate, double stepFactor)
    {
        cv::Mat res(3,1,CV_64F);
        res.at<double>(0,0) = -stepFactor*paramUpdate.at<double>(0,0);
        res.at<double>(1,0) = -stepFactor*paramUpdate.at<double>(1,0);
        res.at<double>(2,0) = -std::asin(stepFactor*paramUpdate.at<double>(2,0));
        return res;
    }
};


/*!
 * \brief ESM with euclidean warp with image intensity offset
 */
class ESM_Euclidean_IO : public RegProc<4>, public EuclidWarp
{
public:
    static constexpr float gradMultiplier = 0.125f;
    static constexpr double stepFactor = 2.0;
    static constexpr bool useESMJac = true;

    static inline void AddJacobians(__m256* mJVals , const __m256 &mPosX, const __m256 &mPosY, const __m256 &refGradX, const __m256 &refGradY, const __m256 &tempGradX, const __m256 &tempGradY, const __m256 &refData, const __m256 &tempData, const __m256 &mask, const __m256 &gradMul)
    {
        const __m256 maskGradMul = _mm256_mul_ps(mask,gradMul);

        mJVals[0] = _mm256_mul_ps(_mm256_add_ps(refGradX,tempGradX),maskGradMul);

        mJVals[1] = _mm256_mul_ps(_mm256_add_ps(refGradY,tempGradY),maskGradMul);

        mJVals[2] = _mm256_sub_ps(_mm256_mul_ps(mPosX,mJVals[1]),_mm256_mul_ps(mPosY,mJVals[0]));

        /// jacobian for intensity
        mJVals[3] = mask;


    }


    static cv::Mat DoWarp(const cv::Mat &tmpImg, const cv::Mat &tmpMask, const cv::Mat &tmpGradX, const cv::Mat &tmpGradY, const cv::Mat &params,
                          const cv::Mat &wtmpImg, const cv::Mat &wtmpMask, const cv::Mat &wtmpGradX, const cv::Mat &wtmpGradY)
    {
        cv::Mat rot_mat = CreateMat(params);

        cv::warpAffine(tmpImg,wtmpImg,rot_mat,wtmpImg.size());
        cv::warpAffine(tmpMask,wtmpMask,rot_mat,wtmpImg.size(),cv::INTER_NEAREST);
        cv::warpAffine(tmpGradX,wtmpGradX,rot_mat,wtmpImg.size());
        cv::warpAffine(tmpGradY,wtmpGradY,rot_mat,wtmpImg.size());

        wtmpImg -= (float)params.at<double>(3,0);
        return rot_mat;
    }

    static cv::Mat toDeltaPars(const cv::Mat &paramUpdate, double stepFactor)
    {
        cv::Mat res(4,1,CV_64F);
        res.at<double>(0,0) = -stepFactor*paramUpdate.at<double>(0,0);
        res.at<double>(1,0) = -stepFactor*paramUpdate.at<double>(1,0);
        res.at<double>(2,0) = -std::asin(stepFactor*paramUpdate.at<double>(2,0));
        res.at<double>(3,0) = -paramUpdate.at<double>(3,0);
        return res;
    }
};

/*!
 * \brief ESM with homography warp with image intensity offset
 */
class ESM_HOM : public RegProc<8>, public HOMWarp
{
public:

    static constexpr float gradMultiplier = 0.25f;
    static constexpr double stepFactor = 2.0;
    static constexpr bool useESMJac = true;

    static inline __m256 Dot2P(const __m256 &u1, const __m256 &u2, const __m256 &v1, const __m256 &v2)
    {
        return _mm256_add_ps(_mm256_mul_ps(u1,v1),_mm256_mul_ps(u2,v2));

    }

    /// jacobian of the homograpy warp
    static inline void AddJacobians(__m256* mJVals , const __m256 &mPosX, const __m256 &mPosY, const __m256 &refGradX, const __m256 &refGradY, const __m256 &tempGradX, const __m256 &tempGradY, const __m256 &refData, const __m256 &tempData, const __m256 &mask, const __m256 &gradMul)
    {
        const __m256 maskGradMul = _mm256_mul_ps(mask,gradMul);
        const __m256 mulGradX = _mm256_mul_ps(_mm256_add_ps(refGradX,tempGradX),maskGradMul);
        const __m256 mulGradY = _mm256_mul_ps(_mm256_add_ps(refGradY,tempGradY),maskGradMul);
        const __m256 dotGP = Dot2P(mPosX,mPosY,mulGradX,mulGradY);
        const __m256 mdotGP = _mm256_mul_ps(dotGP,_mm256_set1_ps(-1.0f));

        mJVals[0] = mulGradX;
        mJVals[1] = mulGradY;

        mJVals[2] = _mm256_sub_ps(_mm256_mul_ps(mPosX,mulGradY),_mm256_mul_ps(mPosY,mulGradX));

        mJVals[3] = (_mm256_mul_ps(dotGP,_mm256_set1_ps(3.0f)));

        mJVals[4] = _mm256_sub_ps(_mm256_mul_ps(mulGradX,mPosX),_mm256_mul_ps(mulGradY,mPosY));

        mJVals[5] = _mm256_add_ps(_mm256_mul_ps(mulGradX,mPosY),_mm256_mul_ps(mulGradY,mPosX));

        mJVals[6] = (_mm256_mul_ps(mPosX,mdotGP));

        mJVals[7] = (_mm256_mul_ps(mPosY,mdotGP));


    }

    /// Do Homography warp with warp perspective.
    cv::Mat DoWarp(const cv::Mat &tmpImg, const cv::Mat &tmpMask, const cv::Mat &tmpGradX, const cv::Mat &tmpGradY, const cv::Mat &params,
                          const cv::Mat &wtmpImg, const cv::Mat &wtmpMask, const cv::Mat &wtmpGradX, const cv::Mat &wtmpGradY)
    {
        cv::Mat G = CreateMat(params);

        cv::warpPerspective(tmpImg,wtmpImg,G,wtmpImg.size());
        cv::warpPerspective(tmpMask,wtmpMask,G,wtmpImg.size(),cv::INTER_NEAREST);
        cv::warpPerspective(tmpGradX,wtmpGradX,G,wtmpImg.size());
        cv::warpPerspective(tmpGradY,wtmpGradY,G,wtmpImg.size());

        return G;
    }

    static cv::Mat toDeltaPars(const cv::Mat &paramUpdate, double stepFactor)
    {
        cv::Mat res(8,1,CV_64F);
        res = -stepFactor*paramUpdate;

        return res;
    }
};


class ESM_HOM_IO : public RegProc<9>, public HOMWarp
{
public:

    static constexpr float gradMultiplier = 0.25f;
    static constexpr double stepFactor = 2.0;
    static constexpr bool useESMJac = true;

    static inline __m256 Dot2P(const __m256 &u1, const __m256 &u2, const __m256 &v1, const __m256 &v2)
    {
        return _mm256_add_ps(_mm256_mul_ps(u1,v1),_mm256_mul_ps(u2,v2));

    }

    static inline void AddJacobians(__m256* mJVals , const __m256 &mPosX, const __m256 &mPosY, const __m256 &refGradX, const __m256 &refGradY, const __m256 &tempGradX, const __m256 &tempGradY, const __m256 &refData, const __m256 &tempData, const __m256 &mask, const __m256 &gradMul)
    {
        const __m256 maskGradMul = _mm256_mul_ps(mask,gradMul);
        const __m256 mulGradX = _mm256_mul_ps(_mm256_add_ps(refGradX,tempGradX),maskGradMul);
        const __m256 mulGradY = _mm256_mul_ps(_mm256_add_ps(refGradY,tempGradY),maskGradMul);
        const __m256 dotGP = Dot2P(mPosX,mPosY,mulGradX,mulGradY);
        const __m256 mdotGP = _mm256_mul_ps(dotGP,_mm256_set1_ps(-1.0f));

        mJVals[0] = mulGradX;
        mJVals[1] = mulGradY;

        mJVals[2] = _mm256_sub_ps(_mm256_mul_ps(mPosX,mulGradY),_mm256_mul_ps(mPosY,mulGradX));

        mJVals[3] = (_mm256_mul_ps(dotGP,_mm256_set1_ps(3.0f)));

        mJVals[4] = _mm256_sub_ps(_mm256_mul_ps(mulGradX,mPosX),_mm256_mul_ps(mulGradY,mPosY));

        mJVals[5] = _mm256_add_ps(_mm256_mul_ps(mulGradX,mPosY),_mm256_mul_ps(mulGradY,mPosX));

        mJVals[6] = (_mm256_mul_ps(mPosX,mdotGP));

        mJVals[7] = (_mm256_mul_ps(mPosY,mdotGP));

        mJVals[8] = mask;


    }

    cv::Mat DoWarp(const cv::Mat &tmpImg, const cv::Mat &tmpMask, const cv::Mat &tmpGradX, const cv::Mat &tmpGradY, const cv::Mat &params,
                          const cv::Mat &wtmpImg, const cv::Mat &wtmpMask, const cv::Mat &wtmpGradX, const cv::Mat &wtmpGradY)
    {
        cv::Mat G = CreateMat(params);

        cv::warpPerspective(tmpImg,wtmpImg,G,wtmpImg.size());
        cv::warpPerspective(tmpMask,wtmpMask,G,wtmpImg.size(),cv::INTER_NEAREST);
        cv::warpPerspective(tmpGradX,wtmpGradX,G,wtmpImg.size());
        cv::warpPerspective(tmpGradY,wtmpGradY,G,wtmpImg.size());

        wtmpImg -= (float)params.at<double>(8,0);

        return G;
    }

    static cv::Mat toDeltaPars(const cv::Mat &paramUpdate, double stepFactor)
    {
        cv::Mat res(9,1,CV_64F);
        res = -stepFactor*paramUpdate;

        res.at<double>(8,0) = -paramUpdate.at<double>(8,0);



        return res;
    }
};

/*!
 * \brief The Inverse Compositional implementation for Euclidean warp
 */
class IC_Euclidean : public RegProc<3>, public EuclidWarp
{
public:

    static constexpr float gradMultiplier = 0.25f;
    static constexpr double stepFactor = 2.0;
    static constexpr bool useESMJac = false;


    static inline void AddJacobians(__m256* mJVals , const __m256 &mPosX, const __m256 &mPosY, const __m256 &refGradX, const __m256 &refGradY, const __m256 &tempGradX, const __m256 &tempGradY, const __m256 &refData, const __m256 &tempData, const __m256 &mask, const __m256 &gradMul)
    {
        const __m256 maskGradMul = _mm256_mul_ps(mask,gradMul);
        const __m256 mulGradX = _mm256_mul_ps(refGradX,maskGradMul);
        const __m256 mulGradY = _mm256_mul_ps(refGradY,maskGradMul);


        mJVals[0] = mulGradX;

        mJVals[1] = mulGradY;

        mJVals[2] = _mm256_sub_ps(_mm256_mul_ps(mPosX,mulGradY),_mm256_mul_ps(mPosY,mulGradX));




    }

    /// warping of gradients is not required for IC
    static cv::Mat DoWarp(const cv::Mat &tmpImg, const cv::Mat &tmpMask, const cv::Mat &tmpGradX, const cv::Mat &tmpGradY, const cv::Mat &params,
                          const cv::Mat &wtmpImg, const cv::Mat &wtmpMask, const cv::Mat &wtmpGradX, const cv::Mat &wtmpGradY)
    {
        cv::Mat rot_mat = CreateMat(params);

        cv::warpAffine(tmpImg,wtmpImg,rot_mat,wtmpImg.size());
        cv::warpAffine(tmpMask,wtmpMask,rot_mat,wtmpImg.size(),cv::INTER_NEAREST);

        return rot_mat;
    }

    static cv::Mat toDeltaPars(const cv::Mat &paramUpdate, double stepFactor)
    {
        cv::Mat res(3,1,CV_64F);
        res.at<double>(0,0) = -stepFactor*paramUpdate.at<double>(0,0);
        res.at<double>(1,0) = -stepFactor*paramUpdate.at<double>(1,0);
        res.at<double>(2,0) = -std::asin(stepFactor*paramUpdate.at<double>(2,0));
        return res;
    }
};


class IC_Euclidean_IO : public RegProc<4>, public EuclidWarp
{
public:
    static constexpr float gradMultiplier = 0.25f;
    static constexpr double stepFactor = 2.0;
    static constexpr bool useESMJac = false;

    static inline void AddJacobians(__m256* mJVals , const __m256 &mPosX, const __m256 &mPosY, const __m256 &refGradX, const __m256 &refGradY, const __m256 &tempGradX, const __m256 &tempGradY, const __m256 &refData, const __m256 &tempData, const __m256 &mask, const __m256 &gradMul)
    {
        const __m256 maskGradMul = _mm256_mul_ps(mask,gradMul);
        const __m256 mulGradX = _mm256_mul_ps(refGradX,maskGradMul);
        const __m256 mulGradY = _mm256_mul_ps(refGradY,maskGradMul);


        mJVals[0] = mulGradX;

        mJVals[1] = mulGradY;

        mJVals[2] = _mm256_sub_ps(_mm256_mul_ps(mPosX,mulGradY),_mm256_mul_ps(mPosY,mulGradX));

        mJVals[3] = mask;


    }


    static cv::Mat DoWarp(const cv::Mat &tmpImg, const cv::Mat &tmpMask, const cv::Mat &tmpGradX, const cv::Mat &tmpGradY, const cv::Mat &params,
                          const cv::Mat &wtmpImg, const cv::Mat &wtmpMask, const cv::Mat &wtmpGradX, const cv::Mat &wtmpGradY)
    {
        cv::Mat rot_mat = CreateMat(params);

        cv::warpAffine(tmpImg,wtmpImg,rot_mat,wtmpImg.size());
        cv::warpAffine(tmpMask,wtmpMask,rot_mat,wtmpImg.size(),cv::INTER_NEAREST);

        wtmpImg -= (float)params.at<double>(3,0);
        return rot_mat;
    }

    static cv::Mat toDeltaPars(const cv::Mat &paramUpdate, double stepFactor)
    {
        cv::Mat res(4,1,CV_64F);
        res.at<double>(0,0) = -stepFactor*paramUpdate.at<double>(0,0);
        res.at<double>(1,0) = -stepFactor*paramUpdate.at<double>(1,0);
        res.at<double>(2,0) = -std::asin(stepFactor*paramUpdate.at<double>(2,0));
        res.at<double>(3,0) = -paramUpdate.at<double>(3,0);
        return res;
    }
};


class IC_HOM : public RegProc<8>, public HOMWarp
{
public:

    static constexpr float gradMultiplier = 0.25f;
    static constexpr double stepFactor = 2.0;
    static constexpr bool useESMJac = false;


    static inline __m256 Dot2P(const __m256 &u1, const __m256 &u2, const __m256 &v1, const __m256 &v2)
    {
        return _mm256_add_ps(_mm256_mul_ps(u1,v1),_mm256_mul_ps(u2,v2));

    }

    static inline void AddJacobians(__m256* mJVals , const __m256 &mPosX, const __m256 &mPosY, const __m256 &refGradX, const __m256 &refGradY, const __m256 &tempGradX, const __m256 &tempGradY, const __m256 &refData, const __m256 &tempData, const __m256 &mask, const __m256 &gradMul)
    {
        const __m256 maskGradMul = _mm256_mul_ps(mask,gradMul);
        const __m256 mulGradX = _mm256_mul_ps(refGradX,maskGradMul);
        const __m256 mulGradY = _mm256_mul_ps(refGradY,maskGradMul);
        const __m256 dotGP = Dot2P(mPosX,mPosY,mulGradX,mulGradY);
        const __m256 mdotGP = _mm256_mul_ps(dotGP,_mm256_set1_ps(-1.0f));

        mJVals[0] = mulGradX;
        mJVals[1] = mulGradY;

        mJVals[2] = _mm256_sub_ps(_mm256_mul_ps(mPosX,mulGradY),_mm256_mul_ps(mPosY,mulGradX));

        mJVals[3] = (_mm256_mul_ps(dotGP,_mm256_set1_ps(3.0f)));

        mJVals[4] = _mm256_sub_ps(_mm256_mul_ps(mulGradX,mPosX),_mm256_mul_ps(mulGradY,mPosY));

        mJVals[5] = _mm256_add_ps(_mm256_mul_ps(mulGradX,mPosY),_mm256_mul_ps(mulGradY,mPosX));

        mJVals[6] = (_mm256_mul_ps(mPosX,mdotGP));

        mJVals[7] = (_mm256_mul_ps(mPosY,mdotGP));


    }

    cv::Mat DoWarp(const cv::Mat &tmpImg, const cv::Mat &tmpMask, const cv::Mat &tmpGradX, const cv::Mat &tmpGradY, const cv::Mat &params,
                          const cv::Mat &wtmpImg, const cv::Mat &wtmpMask, const cv::Mat &wtmpGradX, const cv::Mat &wtmpGradY)
    {
        cv::Mat G = CreateMat(params);

        cv::warpPerspective(tmpImg,wtmpImg,G,wtmpImg.size());
        cv::warpPerspective(tmpMask,wtmpMask,G,wtmpImg.size(),cv::INTER_NEAREST);

        return G;
    }

    static cv::Mat toDeltaPars(const cv::Mat &paramUpdate, double stepFactor)
    {
        cv::Mat res(8,1,CV_64F);
        res = -stepFactor*paramUpdate;

        return res;
    }
};

class IC_HOM_IO : public RegProc<9>, public HOMWarp
{
public:

    static constexpr float gradMultiplier = 0.25f;
    static constexpr double stepFactor = 2.0;
    static constexpr bool useESMJac = false;

    static inline __m256 Dot2P(const __m256 &u1, const __m256 &u2, const __m256 &v1, const __m256 &v2)
    {
        return _mm256_add_ps(_mm256_mul_ps(u1,v1),_mm256_mul_ps(u2,v2));

    }

    static inline void AddJacobians(__m256* mJVals , const __m256 &mPosX, const __m256 &mPosY, const __m256 &refGradX, const __m256 &refGradY, const __m256 &tempGradX, const __m256 &tempGradY, const __m256 &refData, const __m256 &tempData, const __m256 &mask, const __m256 &gradMul)
    {
        const __m256 maskGradMul = _mm256_mul_ps(mask,gradMul);
        const __m256 mulGradX = _mm256_mul_ps(refGradX,maskGradMul);
        const __m256 mulGradY = _mm256_mul_ps(refGradY,maskGradMul);
        const __m256 dotGP = Dot2P(mPosX,mPosY,mulGradX,mulGradY);
        const __m256 mdotGP = _mm256_mul_ps(dotGP,_mm256_set1_ps(-1.0f));

        mJVals[0] = mulGradX;
        mJVals[1] = mulGradY;

        mJVals[2] = _mm256_sub_ps(_mm256_mul_ps(mPosX,mulGradY),_mm256_mul_ps(mPosY,mulGradX));

        mJVals[3] = (_mm256_mul_ps(dotGP,_mm256_set1_ps(3.0f)));

        mJVals[4] = _mm256_sub_ps(_mm256_mul_ps(mulGradX,mPosX),_mm256_mul_ps(mulGradY,mPosY));

        mJVals[5] = _mm256_add_ps(_mm256_mul_ps(mulGradX,mPosY),_mm256_mul_ps(mulGradY,mPosX));

        mJVals[6] = (_mm256_mul_ps(mPosX,mdotGP));

        mJVals[7] = (_mm256_mul_ps(mPosY,mdotGP));

        mJVals[8] = mask;


    }

    cv::Mat DoWarp(const cv::Mat &tmpImg, const cv::Mat &tmpMask, const cv::Mat &tmpGradX, const cv::Mat &tmpGradY, const cv::Mat &params,
                          const cv::Mat &wtmpImg, const cv::Mat &wtmpMask, const cv::Mat &wtmpGradX, const cv::Mat &wtmpGradY)
    {
        cv::Mat G = CreateMat(params);

        cv::warpPerspective(tmpImg,wtmpImg,G,wtmpImg.size());
        cv::warpPerspective(tmpMask,wtmpMask,G,wtmpImg.size(),cv::INTER_NEAREST);

        wtmpImg -= (float)params.at<double>(8,0);

        return G;
    }

    static cv::Mat toDeltaPars(const cv::Mat &paramUpdate, double stepFactor)
    {
        cv::Mat res(9,1,CV_64F);
        res = -stepFactor*paramUpdate;

        res.at<double>(8,0) = -paramUpdate.at<double>(8,0);



        return res;
    }
};


/*!
 *Main image registration class.
 *
 */
template <typename IR_PROC>
class ImageReg : public IImageReg
{


public:

    ImageReg<IR_PROC>()
    {
        stepFactor_ = proc_.stepFactor;

    }
    /// Enforce shared pointer usage
    typedef std::shared_ptr<ImageReg > ptr;
    static ImageReg::ptr Create(){ return std::make_shared< ImageReg >() ; }

    IR_PROC proc_;

    double stepFactor_;


    /// Setup buffer matrices
    void Setup(cv::Size refImgSize, cv::Size tmpImgSize, int imageType, int numLevels)
    {
        refImg_ = AlignedMat::Create(refImgSize,imageType);
        refMask_ = AlignedMat::Create(refImgSize,imageType);
        refGradX_ = AlignedMat::Create(refImgSize,imageType);
        refGradY_ = AlignedMat::Create(refImgSize,imageType);

        tmpImg_ = AlignedMat::Create(tmpImgSize,imageType);
        tmpMask_ = AlignedMat::Create(tmpImgSize,imageType);
        tmpGradX_ = AlignedMat::Create(tmpImgSize,imageType);
        tmpGradY_ = AlignedMat::Create(tmpImgSize,imageType);
        wTmpImg_ = AlignedMat::Create(tmpImgSize,imageType);
        wTmpMask_ = AlignedMat::Create(tmpImgSize,imageType);
        wTmpGradX_ = AlignedMat::Create(tmpImgSize,imageType);
        wTmpGradY_ = AlignedMat::Create(tmpImgSize,imageType);


    }

    /// Set termination criteria
    void SetTermCrit(cv::TermCriteria termCrit) {termCrit_ = termCrit;}
    void SetStepFactor(float factor) {stepFactor_ = factor;}

    ImageRegResults CreateResult(){return proc_.CreateResult();}


    /// Basic function for calculating the sum over all pixels, ESM specific since both gradients are used
    int CalcHesJacDifESMAVX(const cv::Mat &refImage, const cv::Mat &refGradX,const cv::Mat &refGradY, const cv::Mat &refMask, const cv::Mat &templateImage, const cv::Mat &tempGradX, const cv::Mat &tempGradY, const cv::Mat &tempMask, const cv::Point2i &offset, const cv::Point2i &tempPos, const cv::Point2i &size, cv::Mat &hess, cv::Mat &jacDiff, int &numPixels)
    {

        __m256 jVals[proc_.numParams];
        __m256 sumJacDifs[proc_.numParams];
        __m256 sumHesRows[proc_.numParamsSq];
        IR_PROC::LoopZero(jVals,proc_.numParams);
        IR_PROC::LoopZero(sumJacDifs,proc_.numParams);
        IR_PROC::LoopZero(sumHesRows,proc_.numParamsSq);


        int x,y;

        const float *refImgPtr,*refGradPtrX, *refGradPtrY,*tempImgPtr,*tempGradPtrX, *tempGradPtrY,*refMaskPtr, *tempMaskPtr;

        __m256 mDiffVal;


        __m256 mPosX;
        __m256 mPosY;

        __m256 refArr;
        __m256 refGradXArr;
        __m256 refGradYArr;
        __m256 tmpArr;
        __m256 tempGradXArr;
        __m256 tempGradYArr;
        __m256 maskArr;
        __m256 refMaskArr;
        __m256 tempMaskArr;

        __m256 sumPixels = _mm256_set1_ps(0.0);;

        __m256 gradMul = _mm256_set1_ps(proc_.gradMultiplier);

        const __m256 mYStep = _mm256_set1_ps(1.0);

        __m256 mXSteps[3]; // = _mm256_set1_ps(8.0);
        __m256 mXStep; // = _mm256_set1_ps(8.0);

        __m256 *mXStepCur;
        __m256 *mXStepEnd;

        int channels = refImage.channels();


        switch (channels)
        {
        case 4:
            mXStep = _mm256_set1_ps(2.0f);
            mPosX = _mm256_set_ps(0.0f,0.0f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f);
            break;
        case 3:
            mXSteps[0] = _mm256_set_ps(2.0f,3.0f,3.0f,2.0f,3.0f,3.0f,2.0f,3.0f);
            mXSteps[1] = _mm256_set_ps(3.0f,2.0f,3.0f,3.0f,2.0f,3.0f,3.0f,2.0f);
            mXSteps[2] = _mm256_set_ps(3.0f,3.0f,2.0f,3.0f,3.0f,2.0f,3.0f,3.0f);
            mXStepCur = &mXSteps[0];
            mXStepEnd = &mXSteps[2];
            mPosX = _mm256_set_ps(0.0f,0.0f,0.0f,1.0f,1.0f,1.0f,2.0f,2.0f);
            break;
        case 2:
            mXStep = _mm256_set1_ps(4.0f);
            mPosX = _mm256_set_ps(0.0f,0.0f,1.0f,1.0f,2.0f,2.0f,3.0f,3.0f);
            break;
        default:
            mXStep = _mm256_set1_ps(8.0f);
            mPosX = _mm256_set_ps(0.0f,1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f);

        }


        int yStart = tempPos.y,yEnd = yStart+size.y,xStart = tempPos.x*channels,xEnd = xStart+size.x*channels;


        const __m256 mXStart= _mm256_add_ps(mPosX,_mm256_set1_ps((float)xStart));
        mPosY = _mm256_set1_ps((float)yStart);

        for ( y = yStart; y < yEnd; ++y)
        {

            refImgPtr = refImage.ptr<float>(offset.y+y)+offset.x;
            refGradPtrX = refGradX.ptr<float>(offset.y+y)+offset.x;
            refGradPtrY = refGradY.ptr<float>(offset.y+y)+offset.x;
            refMaskPtr = refMask.ptr<float>(offset.y+y)+offset.x;

            tempImgPtr = templateImage.ptr<float>(y);
            tempGradPtrX = tempGradX.ptr<float>(y);
            tempGradPtrY = tempGradY.ptr<float>(y);
            tempMaskPtr = tempMask.ptr<float>(y);


            mPosX = mXStart;


            for ( x = xStart; x < xEnd; x+= 8)
            {
                refMaskArr = _mm256_load_ps(refMaskPtr+x);
                tempMaskArr = _mm256_load_ps(tempMaskPtr+x);
                maskArr = _mm256_mul_ps(refMaskArr,tempMaskArr);
                if (proc_.TestNotZero(maskArr))
                {
                    refArr = _mm256_load_ps(refImgPtr+x);
                    refGradXArr = _mm256_load_ps(refGradPtrX+x);
                    refGradYArr = _mm256_load_ps(refGradPtrY+x);
                    tmpArr = _mm256_load_ps(tempImgPtr+x);
                    tempGradXArr = _mm256_load_ps(tempGradPtrX+x);
                    tempGradYArr = _mm256_load_ps(tempGradPtrY+x);


                    sumPixels = _mm256_add_ps(sumPixels,maskArr);

                    mDiffVal = _mm256_sub_ps(refArr,tmpArr);

                    proc_.AddJacobians(jVals,mPosX,mPosY,refGradXArr,refGradYArr, tempGradXArr, tempGradYArr,refArr,tmpArr,maskArr,gradMul);
                    proc_.CalcMat(jVals,mDiffVal,sumHesRows,sumJacDifs);

                }


                if (refImage.channels() != 3) mPosX = _mm256_add_ps(mPosX,mXStep);
                else
                {
                    mPosX = _mm256_add_ps(mPosX,*mXStepCur);
                    mXStepCur = mXStepCur == mXStepEnd? &mXSteps[0] : mXStepCur+1;
                }

            }

            mPosY = _mm256_add_ps(mPosY,mYStep);
        }

        numPixels = (int)Utils_SIMD::HSumAvxFlt(sumPixels);

        proc_.WriteToMatrix(sumHesRows,sumJacDifs, jacDiff,hess,numPixels);


        return 0;
    }


    /// IC only requires reference templates
    int CalcHesJacDifICAVX(const cv::Mat &refImage, const cv::Mat &refGradX,const cv::Mat &refGradY, const cv::Mat &refMask, const cv::Mat &templateImage, const cv::Mat &tempMask, const cv::Point2i &offset, const cv::Point2i &tempPos, const cv::Point2i &size, cv::Mat &hess, cv::Mat &jacDiff, int &numPixels)
    {

        __m256 jVals[proc_.numParams];
        __m256 sumJacDifs[proc_.numParams];
        __m256 sumHesRows[proc_.numParamsSq];
        IR_PROC::LoopZero(jVals,proc_.numParams);
        IR_PROC::LoopZero(sumJacDifs,proc_.numParams);
        IR_PROC::LoopZero(sumHesRows,proc_.numParamsSq);

        int x,y;

        const float *refImgPtr,*refGradPtrX, *refGradPtrY,*tempImgPtr,*refMaskPtr, *tempMaskPtr;

        __m256 mDiffVal;


        __m256 mPosX;
        __m256 mPosY;

        __m256 refArr;
        __m256 refGradXArr;
        __m256 refGradYArr;
        __m256 tmpArr;
        __m256 maskArr;
        __m256 refMaskArr;
        __m256 tempMaskArr;

        __m256 sumPixels = _mm256_set1_ps(0.0);;

        __m256 gradMul = _mm256_set1_ps(proc_.gradMultiplier);

        const __m256 mYStep = _mm256_set1_ps(1.0);

        __m256 mXSteps[3]; // = _mm256_set1_ps(8.0);
        __m256 mXStep; // = _mm256_set1_ps(8.0);

        __m256 *mXStepCur;
        __m256 *mXStepEnd;

        int channels = refImage.channels();


        switch (channels)
        {
        case 4:
            mXStep = _mm256_set1_ps(2.0f);
            mPosX = _mm256_set_ps(0.0f,0.0f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f);
            break;
        case 3:
            mXSteps[0] = _mm256_set_ps(2.0f,3.0f,3.0f,2.0f,3.0f,3.0f,2.0f,3.0f);
            mXSteps[1] = _mm256_set_ps(3.0f,2.0f,3.0f,3.0f,2.0f,3.0f,3.0f,2.0f);
            mXSteps[2] = _mm256_set_ps(3.0f,3.0f,2.0f,3.0f,3.0f,2.0f,3.0f,3.0f);
            mXStepCur = &mXSteps[0];
            mXStepEnd = &mXSteps[2];
            mPosX = _mm256_set_ps(0.0f,0.0f,0.0f,1.0f,1.0f,1.0f,2.0f,2.0f);
            break;
        case 2:
            mXStep = _mm256_set1_ps(4.0f);
            mPosX = _mm256_set_ps(0.0f,0.0f,1.0f,1.0f,2.0f,2.0f,3.0f,3.0f);
            break;
        default:
            mXStep = _mm256_set1_ps(8.0f);
            mPosX = _mm256_set_ps(0.0f,1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f);

        }

        int yStart = tempPos.y,yEnd = yStart+size.y,xStart = tempPos.x*channels,xEnd = xStart+size.x*channels;

        const __m256 mXStart= _mm256_add_ps(mPosX,_mm256_set1_ps((float)xStart));
        mPosY = _mm256_set1_ps((float)yStart);

        for ( y = yStart; y < yEnd; ++y)
        {

            refImgPtr = refImage.ptr<float>(offset.y+y)+offset.x;
            refGradPtrX = refGradX.ptr<float>(offset.y+y)+offset.x;
            refGradPtrY = refGradY.ptr<float>(offset.y+y)+offset.x;
            refMaskPtr = refMask.ptr<float>(offset.y+y)+offset.x;

            tempImgPtr = templateImage.ptr<float>(y);
            tempMaskPtr = tempMask.ptr<float>(y);


            mPosX = mXStart;


            for ( x = xStart; x < xEnd; x+= 8)
            {
                refMaskArr = _mm256_load_ps(refMaskPtr+x);
                tempMaskArr = _mm256_load_ps(tempMaskPtr+x);
                maskArr = _mm256_mul_ps(refMaskArr,tempMaskArr);
                if (proc_.TestNotZero(maskArr))
                {
                    refArr = _mm256_load_ps(refImgPtr+x);
                    refGradXArr = _mm256_load_ps(refGradPtrX+x);
                    refGradYArr = _mm256_load_ps(refGradPtrY+x);
                    tmpArr = _mm256_load_ps(tempImgPtr+x);


                    sumPixels = _mm256_add_ps(sumPixels,maskArr);

                    mDiffVal = _mm256_sub_ps(refArr,tmpArr);

                    proc_.AddJacobians(jVals,mPosX,mPosY,refGradXArr,refGradYArr, mYStep, mYStep,refArr,tmpArr,maskArr,gradMul);
                    proc_.CalcMat(jVals,mDiffVal,sumHesRows,sumJacDifs);

                }


                if (refImage.channels() != 3) mPosX = _mm256_add_ps(mPosX,mXStep);
                else
                {
                    mPosX = _mm256_add_ps(mPosX,*mXStepCur);
                    mXStepCur = mXStepCur == mXStepEnd? &mXSteps[0] : mXStepCur+1;
                }

            }

            mPosY = _mm256_add_ps(mPosY,mYStep);
        }

        numPixels = (int)Utils_SIMD::HSumAvxFlt(sumPixels);

        proc_.WriteToMatrix(sumHesRows,sumJacDifs, jacDiff,hess,numPixels);


        return 0;
    }


    /// Perform actual alignment
    void AlignImage(const cv::Mat &refImg, const cv::Mat &refMask, const cv::Mat &tmpImg, const cv::Mat &tmpMask, ImageRegResults &result)
    {


        refImg_->CopyDataFrom(refImg);
        refMask_->CopyDataFrom(refMask);

        tmpImg_->CopyDataFrom(tmpImg);
        tmpMask_->CopyDataFrom(tmpMask);

        cv::Sobel(refImg,refGradX_->mat_,-1,1,0);
        cv::Sobel(refImg,refGradY_->mat_,-1,0,1);

        if (proc_.useESMJac)
        {
            cv::Sobel(tmpImg,tmpGradX_->mat_,-1,1,0);
            cv::Sobel(tmpImg,tmpGradY_->mat_,-1,0,1);

        }
        cv::Mat jacF = IR_PROC::CreateJacF();
        cv::Mat hessF = IR_PROC::CreateHessF();
        //cv::Mat hessFInv = IR_PROC::CreateHessF();
        cv::Mat jacD = IR_PROC::CreateJacD();
        cv::Mat hessD = IR_PROC::CreateHessD();
        cv::Mat hessDInv = IR_PROC::CreateHessD();
        int numPixels = 0;

        cv::Point2i offset(0,0);
        cv::Point2i tmpPos(0,0);
        cv::Point2i tmpSize(tmpImg.cols,tmpImg.rows);




        while (result.TestResults(termCrit_))
        {

            result.warpMat = IR_PROC::DoWarp(tmpImg_->mat_,tmpMask_->mat_,tmpGradX_->mat_,tmpGradY_->mat_,result.params,wTmpImg_->mat_,wTmpMask_->mat_,wTmpGradX_->mat_,wTmpGradY_->mat_);

            if (proc_.useESMJac) CalcHesJacDifESMAVX(refImg_->mat_,refGradX_->mat_,refGradY_->mat_,refMask_->mat_,
                                                     wTmpImg_->mat_,wTmpGradX_->mat_,wTmpGradY_->mat_,wTmpMask_->mat_,
                                                     offset,tmpPos,tmpSize,hessF,jacF,numPixels);
            else CalcHesJacDifICAVX(refImg_->mat_,refGradX_->mat_,refGradY_->mat_,refMask_->mat_,
                                    wTmpImg_->mat_,wTmpMask_->mat_,
                                    offset,tmpPos,tmpSize,hessF,jacF,numPixels);

            hessF.convertTo(hessD,CV_64F);
            jacF.convertTo(jacD,CV_64F);

            //std::cout << "Hess: " << hessD << std::endl;
            //std::cout << "Jac: " << jacD << std::endl;

            cv::invert(hessD,hessDInv );
            cv::Mat resPars = hessDInv*jacD;
            cv::Mat deltaPars = IR_PROC::toDeltaPars(resPars,stepFactor_);

            result.Update(deltaPars);


            Utils_SIMD::CalcErrorSqrAVX(refImg_->mat_,refMask_->mat_,wTmpImg_->mat_,wTmpMask_->mat_,offset,result.error,result.pixels);

            //results.push_back(std::move(result));

            //PrintResult(result);

        }

        PrintResult(result);



    }


    void PrintResult(ImageRegResults &res)
    {
        std::cout << "Iter: " << res.iterations << " Error: " << res.error << " del: " << res.delta << " pars: " << res.params << " delPars: " << res.deltaParams << " Pixels: " << res.pixels << std::endl;
    }

    AlignedMat::ptr refImg_,refMask_;
    AlignedMat::ptr refGradX_,refGradY_;

    AlignedMat::ptr tmpImg_,tmpMask_;
    AlignedMat::ptr tmpGradX_,tmpGradY_;
    AlignedMat::ptr wTmpImg_,wTmpMask_;
    AlignedMat::ptr wTmpGradX_,wTmpGradY_;

    cv::TermCriteria termCrit_;


};

/*!
 * \brief Class for wrapping image alignment using image pyramid
 */
template <typename IR_PROC>
class ImageRegPyr : public IImageReg
{
public:

    typedef std::shared_ptr<ImageRegPyr > ptr;

    static ImageRegPyr::ptr Create(){ return std::make_shared< ImageRegPyr >() ; }


    ImageRegPyr()
    {
    }

    void SetTermCrit(cv::TermCriteria termCrit)
    {
        for (unsigned int i = 0; i < procs_.size();++i)
        {
            procs_[i]->SetTermCrit(termCrit);
        }
    }

    void SetStepFactor(float factor)
    {
        for (unsigned int i = 0; i < procs_.size();++i)
        {
            procs_[i]->SetStepFactor(factor);
        }
    }
    ImageRegResults CreateResult(){return IR_PROC::CreateResult();}


    void Setup(cv::Size refImgSize, cv::Size tmpImgSize, int imageType, int numLevels)
    {
        numLevels_ = numLevels;
        procs_.clear();
        levelScale_.clear();

        typename ImageReg<IR_PROC>::ptr curReg = ImageReg<IR_PROC>::Create();


        float curScale = 1.0/(std::pow(2.0, (numLevels_-1)));

        for (int tl = 0; tl < numLevels_;tl++)
        {
            cv::Size curImgSize;
            cv::Size curTmpSize;
            curImgSize.width = refImgSize.width*curScale;
            curImgSize.height = refImgSize.height*curScale;
            curTmpSize.width = tmpImgSize.width*curScale;
            curTmpSize.height = tmpImgSize.height*curScale;


            curReg = ImageReg<IR_PROC>::Create();
            curReg->Setup(curImgSize,curTmpSize,imageType,0);
            levelScale_.push_back(curScale);
            procs_.push_back(curReg);

            curScale *= 2;

        }
    }

    void AlignImage(const cv::Mat &refImg, const cv::Mat &refMask, const cv::Mat &tmpImg, const cv::Mat &tmpMask, ImageRegResults &result)
    {

        /// scale input parameter to the first (smallest) level
        IR_PROC::ParamScale(result.params,levelScale_[0]);

        /// loop over all levels
        for (int tl = 0; tl < numLevels_;++tl)
        {
            cv::Mat refImgS,refMasks;
            cv::Mat tmpImgS,tmpMaskS;

            cv::resize(refImg,refImgS,cv::Size(),levelScale_[tl],levelScale_[tl]);
            cv::resize(refMask,refMasks,cv::Size(),levelScale_[tl],levelScale_[tl]);
            cv::resize(tmpImg,tmpImgS,cv::Size(),levelScale_[tl],levelScale_[tl]);
            cv::resize(tmpMask,tmpMaskS,cv::Size(),levelScale_[tl],levelScale_[tl]);

            cv::threshold(refMasks,refMasks,0.9999,1.0,CV_THRESH_BINARY);
            cv::threshold(tmpMaskS,tmpMaskS,0.9999,1.0,CV_THRESH_BINARY);

            if (tl != 0) IR_PROC::ParamScaleUp(result.params);

            procs_[tl]->AlignImage(refImgS, refMasks, tmpImgS, tmpMaskS, result);

            std::cout << std::endl << std::endl;
            std::cout << "PyrLevel: " << tl << " scale: " <<  levelScale_[tl] << " cc: " << result.error <<  std::endl;
            std::cout << " Mat: " << result.warpMat <<  std::endl;
            std::cout << std::endl << std::endl;




            result.ResetStats();
            result.iterations = 0;


        }


    }
    void PrintResult(ImageRegResults &res)
    {
        std::cout << "Iter: " << res.iterations << " Error: " << res.error << " del: " << res.delta << " pars: " << res.params << " delPars: " << res.deltaParams << std::endl;
    }


    std::vector< float > levelScale_;
    std::vector< typename ImageReg<IR_PROC>::ptr > procs_;
    int numLevels_;


};






#endif // IMAGEREGBASE_H
