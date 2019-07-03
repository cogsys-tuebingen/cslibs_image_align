
//#define __AVX__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


//#include <opencv2/imgcodecs.hpp>
#include <opencv2/video/video.hpp>
//#include <opencv2/core/utility.hpp>

#include <sys/time.h>
#include <iostream>

#include <cslibs_image_align/imagereg.h>
#include <cslibs_image_align/blockreg.h>

#include <iostream>
#include <iomanip>
#include <fstream>

#include <unistd.h>
#include <stdio.h>
#include <limits.h>


#if CV_MAJOR_VERSION < 3

namespace cv {
enum MOTION_MODE { MOTION_AFFINE = 0, MOTION_EUCLIDEAN = 0, MOTION_HOMOGRAPHY};

static double findTransformECC(cv::Mat refImgS,cv::Mat tmpImgS,cv::Mat warpMat,int warpMode_,cv::TermCriteria termCrit)
{
    std::cout << "not support by this opencv version";
}


}

#endif


class ImageRegEcc : public IImageReg
{
public:

    typedef std::shared_ptr<ImageRegEcc > ptr;

    static ImageRegEcc::ptr Create(){ return std::make_shared< ImageRegEcc >() ; }
    static ImageRegEcc::ptr Create(int warpMode){ return std::make_shared< ImageRegEcc >(warpMode) ; }



    ImageRegEcc()
    {
        warpMode_ = cv::MOTION_EUCLIDEAN;

    }

    ImageRegEcc(int warpMode)
    {
        warpMode_ = warpMode;
    }

    void SetStepFactor(float factor) {}


    void SetTermCrit(cv::TermCriteria termCrit)
    {
        termCrit_ = termCrit;
    }

    ImageRegResults CreateResult(){return ImageRegResults(8);}


    void Setup(cv::Size imageSize, cv::Size templateSize, int imageType, int numLevels)
    {

    }

    void AlignImage(const cv::Mat &refImg, const cv::Mat &refMask, const cv::Mat &tmpImg, const cv::Mat &tmpMask, ImageRegResults &result)
    {

        if (result.warpMat.type() != CV_32F) result.warpMat.convertTo(result.warpMat,CV_32F);

        if (warpMode_ == cv::MOTION_HOMOGRAPHY && result.warpMat.rows < 3)
        {
            cv::Mat row = cv::Mat::zeros(1,3,CV_32F);
            row.at<float>(0,2) = 1.0f;

            result.warpMat.push_back(row);

        }
        if (warpMode_ != cv::MOTION_HOMOGRAPHY && result.warpMat.rows >= 3)
        {
            result.warpMat = result.warpMat.rowRange(0,2);

        }

        result.error = cv::findTransformECC(refImg,tmpImg,result.warpMat,warpMode_,termCrit_);


        cv::Mat inWarp;

        if ( warpMode_ == cv::MOTION_HOMOGRAPHY )
        {
            cv::invert(result.warpMat,inWarp);
            result.warpMat = inWarp;

        }
        else
        {
            cv::Mat row = cv::Mat::zeros(1,3,CV_32F);
            row.at<float>(0,2) = 1.0f;
            result.warpMat.push_back(row);
            cv::invert(result.warpMat,inWarp);
            result.warpMat = inWarp.rowRange(0,2);

        }

        //PrintResult(result);



    }
    void PrintResult(ImageRegResults &res)
    {
        std::cout << "Iter: " << res.iterations << " Error: " << res.error << " del: " << res.delta << " pars: " << res.params << " delPars: " << res.deltaParams << std::endl;
    }


    //std::vector< float > levelScale_;
    //int numLevels_;
    cv::TermCriteria termCrit_;
    int warpMode_;


};


class ImageRegPyrEcc : public IImageReg
{
public:

    typedef std::shared_ptr<ImageRegPyrEcc > ptr;

    static ImageRegPyrEcc::ptr Create(){ return std::make_shared< ImageRegPyrEcc >() ; }
    static ImageRegPyrEcc::ptr Create(int warpMode){ return std::make_shared< ImageRegPyrEcc >(warpMode) ; }



    ImageRegPyrEcc()
    {
        warpMode_ = cv::MOTION_EUCLIDEAN;
    }

    ImageRegPyrEcc(int warpMode)
    {
        warpMode_ = warpMode;
    }

    void SetStepFactor(float factor) {}


    void SetTermCrit(cv::TermCriteria termCrit)
    {
        termCrit_ = termCrit;
    }

    ImageRegResults CreateResult(){return ImageRegResults(8);}


    void Setup(cv::Size imageSize, cv::Size templateSize, int imageType, int numLevels)
    {
        numLevels_ = numLevels;
        levelScale_.clear();



        float curScale = 1.0/(std::pow(2.0, (numLevels_-1)));

        //int curScale = 1;

        for (int tl = 0; tl < numLevels_;tl++)
        {
            cv::Size curImgSize;
            cv::Size curTmpSize;
            curImgSize.width = imageSize.width*curScale;
            curImgSize.height = imageSize.height*curScale;
            curTmpSize.width = templateSize.width*curScale;
            curTmpSize.height = templateSize.height*curScale;


            levelScale_.push_back(curScale);

            curScale *= 2.0;

        }
    }

    void AlignImage(const cv::Mat &refImg, const cv::Mat &refMask, const cv::Mat &tmpImg, const cv::Mat &tmpMask, ImageRegResults &result)
    {
        if (result.warpMat.type() != CV_32F) result.warpMat.convertTo(result.warpMat,CV_32F);

        if (warpMode_ == cv::MOTION_HOMOGRAPHY && result.warpMat.rows < 3)
        {
            cv::Mat row = cv::Mat::zeros(1,3,CV_32F);
            row.at<float>(0,2) = 1.0f;
            result.warpMat.push_back(row);

        }
        if (warpMode_ != cv::MOTION_HOMOGRAPHY && result.warpMat.rows >= 3)
        {
            result.warpMat = result.warpMat.rowRange(0,2);

        }

        result.warpMat.at<float>(0,2) = result.warpMat.at<float>(0,2)*levelScale_[0];
        result.warpMat.at<float>(1,2) = result.warpMat.at<float>(1,2)*levelScale_[0];

        for (int tl = 0; tl < numLevels_;++tl)
        {
            cv::Mat refImgS;//,refMasks;
            cv::Mat tmpImgS,tmpMaskS,tmpMaskI;

            cv::resize(refImg,refImgS,cv::Size(),levelScale_[tl],levelScale_[tl]);
            cv::resize(tmpImg,tmpImgS,cv::Size(),levelScale_[tl],levelScale_[tl]);
            cv::resize(tmpMask,tmpMaskS,cv::Size(),levelScale_[tl],levelScale_[tl]);

            cv::threshold(tmpMaskS,tmpMaskS,0.9999,1.0,CV_THRESH_BINARY);
            tmpMaskS.convertTo(tmpMaskI,CV_8U);

            if (tl != 0)
            {
                result.warpMat.at<float>(0,2) = result.warpMat.at<float>(0,2)*2.0;
                result.warpMat.at<float>(1,2) = result.warpMat.at<float>(1,2)*2.0;
            }
            //PrintResult(result);


            result.error = cv::findTransformECC(refImgS,tmpImgS,result.warpMat,warpMode_,termCrit_);

            //bool AlignImage(const cv::Mat &refImage, const cv::Mat &refMask, const cv::Mat &templateImage, const cv::Mat &templateMask, const cv::Point2i &offset, ImageRegResults &result)
            std::cout << std::endl << std::endl;
            std::cout << "PyrLevel: " << tl << " scale: " <<  levelScale_[tl] << " cc: " << result.error <<  std::endl;
            std::cout << " Mat: " << result.warpMat <<  std::endl;
            std::cout << std::endl << std::endl;






            result.ResetStats();
            result.iterations = 0;


        }

        cv::Mat inWarp;

        if ( warpMode_ == cv::MOTION_HOMOGRAPHY )
        {
            cv::invert(result.warpMat,inWarp);
            result.warpMat = inWarp;

        }
        else
        {
            cv::Mat row = cv::Mat::zeros(1,3,CV_32F);
            row.at<float>(0,2) = 1.0f;
            result.warpMat.push_back(row);
            cv::invert(result.warpMat,inWarp);
            result.warpMat = inWarp.rowRange(0,2);

        }

        //PrintResult(result);



    }
    void PrintResult(ImageRegResults &res)
    {
        std::cout << "Iter: " << res.iterations << " Error: " << res.error << " del: " << res.delta << " pars: " << res.params << " delPars: " << res.deltaParams << std::endl;
    }


    std::vector< float > levelScale_;
    int numLevels_;
    cv::TermCriteria termCrit_;
    int warpMode_;


};




void Display32FImage(std::string wname, cv::Mat image, float imin, float imax)
{
    cv::namedWindow(wname,0);
    cv::Mat displayImg;
    if (image.channels() == 1)
        image.convertTo(displayImg,CV_8U,256.0/(double)(imax-imin),-imin*256.0/(double)(imax-imin));
    if (image.channels() == 3)
        image.convertTo(displayImg,CV_8UC3,256.0/(double)(imax-imin),-imin*256.0/(double)(imax-imin));
    if (image.channels() == 4)
        image.convertTo(displayImg,CV_8UC4,256.0/(double)(imax-imin),-imin*256.0/(double)(imax-imin));

    cv::imshow(wname,displayImg);

}

void Display32FImage(std::string wname, cv::Mat &image)
{
    cv::namedWindow(wname,0);
    double imin,imax;
    int minIdx,maxIdx;
    cv::minMaxIdx(image,&imin,&imax,&minIdx,&maxIdx);

    cv::Mat displayImg;
    if (image.channels() == 1)
        image.convertTo(displayImg,CV_8U,256.0/(double)(imax-imin),-imin*256.0/(double)(imax-imin));
    if (image.channels() == 3)
        image.convertTo(displayImg,CV_8UC3,256.0/(double)(imax-imin),-imin*256.0/(double)(imax-imin));

    cv::imshow(wname,displayImg);

}



void ReadImage(std::string imgName, int erosion_size, cv::Mat &img, cv::Mat &mask)
{

    cv::Mat refImg = cv::imread(imgName);
    cv::Mat refImgF;
    refImg.convertTo(refImgF,CV_32F);
    if (refImgF.channels() > 1) cv::cvtColor(refImgF,refImg, cv::COLOR_RGB2GRAY); else refImg = refImgF;

    img = refImg;
    mask = cv::Mat::ones(refImgF.size(),CV_32F );

    if (erosion_size > 0)
    {
        int erosion_type = cv::MORPH_RECT;
        cv::Mat element = cv::getStructuringElement(erosion_type, cv::Size(2*erosion_size + 1, 2*erosion_size+1));


        cv::Mat etmpMask;
        cv::erode(mask,etmpMask, element,cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0));
        //refMask = erefMask;
        mask = etmpMask;
    }


}


int main(int argc, char *argv[])
{

    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("Current working dir: %s\n", cwd);
    } else {
        perror("getcwd() error");
        return 1;
    }
    ///Read Images

    cv::Mat refImg;
    cv::Mat refMask;

    ReadImage("img3.png",1,refImg,refMask);



    cv::Mat tmpImg;
    cv::Mat tmpMask;

    ReadImage("img4.png",1,tmpImg,tmpMask);


    AlignedMat::ptr refImgA = AlignedMat::Create(refImg);
    AlignedMat::ptr refMaskA = AlignedMat::Create(refMask);
    AlignedMat::ptr tmpImgA = AlignedMat::Create(tmpImg);
    AlignedMat::ptr tmpMaskA = AlignedMat::Create(tmpMask);

    cv::Point2f initial = Utils_SIMD::GetInitialTrans(refImgA->mat_,refMaskA->mat_,tmpImgA->mat_,tmpMaskA->mat_,8,20);

    ///Setup ImageReg
    int number_of_iterations = 50;
    double termination_eps = 0.05;
    int numPyrLevels = 4;
    cv::TermCriteria term_criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, number_of_iterations, termination_eps);

    auto regBlock = BlockReg<ESM_Euclidean,BS_EuclidDist>::Create();
    IImageReg::ptr regESM_PE = regBlock;
    regBlock->SetBlockParams(refImg.size(),cv::Size(80,80),cv::Point2i(40,40));

    //IImageReg::ptr regESM_PE = ImageRegPyr<ESM_HOM>::Create();

    regESM_PE->Setup(refImg.size(),refImg.size(),refImg.type(),numPyrLevels);
    regESM_PE->SetTermCrit(term_criteria);
    ImageRegResults resEuc = regESM_PE->CreateResult();


    resEuc.params.at<double>(0,0) = initial.x;
    resEuc.params.at<double>(1,0) = initial.y;

    timeval tStart;
     gettimeofday(&tStart, NULL);


    regESM_PE->AlignImage(refImg,refMask,tmpImg,tmpMask,resEuc);


    timeval tZend;
    gettimeofday(&tZend, NULL);
    double tE= (double)(tZend.tv_sec - tStart.tv_sec)*1000.0+ (double)(tZend.tv_usec - tStart.tv_usec)/1000.0;



    cv::Mat aligned,alignedMask;
    cv::Mat warp_matrix = resEuc.warpMat;

    if (warp_matrix.rows == 2)
    {

        cv::warpAffine(tmpImg, aligned, warp_matrix, tmpImg.size(), cv::INTER_LINEAR);
        cv::warpAffine(tmpMask, alignedMask, warp_matrix, tmpImg.size(), cv::INTER_NEAREST);

    }
    else
    {
        cv::warpPerspective(tmpImg, aligned, warp_matrix, tmpImg.size(), cv::INTER_LINEAR);
        cv::warpPerspective(tmpMask, alignedMask, warp_matrix, tmpImg.size(), cv::INTER_NEAREST);
    }
    // Show final output

    cv::Mat diffImg;
    cv::absdiff(aligned,refImg,diffImg);
    cv::multiply(diffImg , alignedMask,diffImg);
    cv::multiply(diffImg , refMask,diffImg);

    Display32FImage("diff", diffImg,0.0f,100.0f);

    Display32FImage("Ref", refImg,0.0f,250.0f);
    Display32FImage("Tmp", tmpImg,0.0f,250.0f);
    Display32FImage("Warped", aligned,0.0f,250.0f);
    Display32FImage("TmpMask", tmpMask,0.0f,1.0f);
    Display32FImage("WarpedMask", alignedMask,0.0f,1.0f);

    cv::Mat sqrdDiff;
    cv::pow(diffImg,2.0,sqrdDiff);

    double ferror = cv::sum(sqrdDiff)[0]/cv::sum(alignedMask)[0];
    std::cout << " Error: " << ferror << std::endl;
    std::cout << " Time: " << tE << "ms" << std::endl;

    cv::waitKey(0);



    return 1;
}



int main2(int argc, char *argv[])
{
    ///Read Images
    ///

    /*
    int number_of_iterations = 50;
    double termination_eps = 0.05;
    int numPyrLevels = 4;
    cv::TermCriteria term_criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                   number_of_iterations, termination_eps);




    IImageReg::ptr regESM_PE = ImageRegPyr<ESM_Euclidean>::Create();
    //IImageReg::ptr regESM_PE = ImageRegEcc::Create();
    regESM_PE->Setup(imgSize,imgSize,cvMatType,numPyrLevels);
    regESM_PE->SetTermCrit(term_criteria);
    regESM_PE->SetStepFactor(3.0f);
    ImageRegResults resEuc = regESM_PE->CreateResult();

    cv::TermCriteria term_criteria_blur(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                        200, 0.03);


    IImageReg::ptr regESM_BL = ImageReg<ESM_Euclidean>::Create();
    //IImageReg::ptr regESM_PE = ImageRegEcc::Create();
    regESM_BL->Setup(imgSize,imgSize,cvMatType,numPyrLevels);
    regESM_BL->SetTermCrit(term_criteria_blur);
    regESM_BL->SetStepFactor(3.0f);



    cv::TermCriteria term_criteria_hom(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                       20, termination_eps);


    IImageReg::ptr regESM_PH = ImageReg<ESM_HOM_IO>::Create();
    //IImageReg::ptr regESM_PE = ImageRegEcc::Create();
    regESM_PH->Setup(imgSize,imgSize,cvMatType,numPyrLevels);
    regESM_PH->SetTermCrit(term_criteria_hom);
    regESM_PH->SetStepFactor(3.0f);
    ImageRegResults resHom = regESM_PH->CreateResult();


    cv::Mat curWarp = cv::Mat::eye(3,3,CV_64F);


    cv::Mat refImg;
    cv::Mat refMask;

    ReadImage("img1.png",1,refImg,refMask);



    cv::Mat tmpImg;
    cv::Mat tmpMask;

    ReadImage("img2.png",1,tmpImg,tmpMask);


    resEuc = regESM_PE->CreateResult();

    resEuc.params.at<double>(2,0) = diffYaw;
    resEuc.params.at<double>(0,0) = (trotMat.at<double>(0,2))/initTransScale;
    resEuc.params.at<double>(1,0) = (trotMat.at<double>(1,2))/initTransScale;

    regESM_PE->AlignImage(refImg,refMask,tmpImg,tmpMask,resEuc);

    regESM_BL->AlignImage(brefImg,refMask,btmpImg,tmpMask,resEuc);


    resHom = regESM_PH->CreateResult();

    resHom.params.at<double>(0,0) = resEuc.params.at<double>(0,0);
    resHom.params.at<double>(1,0) = resEuc.params.at<double>(1,0);
    resHom.params.at<double>(2,0) = resEuc.params.at<double>(2,0);

    regESM_PH->AlignImage(brefImg,refMask,btmpImg,tmpMask,resHom);


    timeval tZend;
    gettimeofday(&tZend, NULL);
    double tE= (double)(tZend.tv_sec - tStart.tv_sec)*1000.0+ (double)(tZend.tv_usec - tStart.tv_usec)/1000.0;



    cv::Mat aligned,alignedMask;
    cv::Mat warp_matrix = resHom.warpMat;

    if (resEuc.error < resHom.error)
    {
        warp_matrix = resEuc.warpMat;
    }

    if (warp_matrix.rows == 2)
    {

        cv::warpAffine(tmpImg, aligned, warp_matrix, tmpImg.size(), cv::INTER_LINEAR);
        cv::warpAffine(tmpMask, alignedMask, warp_matrix, tmpImg.size(), cv::INTER_NEAREST);

    }
    else
    {
        cv::warpPerspective(tmpImg, aligned, warp_matrix, tmpImg.size(), cv::INTER_LINEAR);
        cv::warpPerspective(tmpMask, alignedMask, warp_matrix, tmpImg.size(), cv::INTER_NEAREST);
    }
    // Show final output

    cv::Mat diffImg;
    cv::absdiff(aligned,refImg,diffImg);
    cv::multiply(diffImg , alignedMask,diffImg);
    cv::multiply(diffImg , refMask,diffImg);
    Display32FImage("RotTmp", tmpIsRot,0.0f,250.0f);

    Display32FImage("diff", diffImg,0.0f,100.0f);

    Display32FImage("Ref", refImg,0.0f,250.0f);
    Display32FImage("Tmp", tmpImg,0.0f,250.0f);
    Display32FImage("Warped", aligned,0.0f,250.0f);
    Display32FImage("TmpMask", tmpMask,0.0f,1.0f);
    Display32FImage("WarpedMask", alignedMask,0.0f,1.0f);

    cv::Mat sqrdDiff;
    cv::pow(diffImg,2.0,sqrdDiff);

    double ferror = cv::sum(sqrdDiff)[0]/cv::sum(alignedMask)[0];
    std::cout << " Error: " << ferror << std::endl;
    std::cout << " Time: " << tE << "ms" << std::endl;
    std::cout << " RefYaw: " << refYaw << " TmpYaw: " << tmpYaw << " Diff: " << diffYaw << std::endl;

    cv::waitKey(0);

*/

    return 1;
}









