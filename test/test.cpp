
//#define __AVX__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


//#include <opencv2/imgcodecs.hpp>
#include <opencv2/video/video.hpp>
//#include <opencv2/core/utility.hpp>

#include <sys/time.h>
#include <iostream>

//#include <imageregpyr.h>
//#include <imageregesm.h>
#include "imagereg.h"
#include "blockreg.h"

#include <iostream>
#include <iomanip>
#include <fstream>

#if CV_MAJOR_VERSION < 3

/*
namespace cv {
enum MOTION_MODE { MOTION_AFFINE = 0, MOTION_HOMOGRAPHY};

}

double alignImageECC(const cv::Mat &src, const cv::Mat &dst, const int &warp_mode, const cv::TermCriteria &criteria, cv::Mat &warpMat)
{
    return 0;
}*/
#else
double alignImageECC(const cv::Mat &refImg, const cv::Mat &refMsk , const cv::Mat &tmpImg, const cv::Mat &tmpMask, int warpMode, int numAngles, const cv::TermCriteria &criteria, cv::Mat &warpMat)
{
    double cc = 0;


    bool usePyr = true;


    if (!usePyr)
    {

        for (int tl = 0; tl < numAngles;++tl)
        {
            if ( warpMode == cv::MOTION_HOMOGRAPHY )
                warpMat = cv::Mat::eye(3, 3, CV_32F);
            else
                warpMat = cv::Mat::eye(2, 3, CV_32F);


            cc = cv::findTransformECC ( refImg, tmpImg, warpMat, warpMode, criteria );

        }

    }
    else
    {
        IImageReg::ptr imgReg;
        imgReg = ImageRegPyrEcc::Create(warpMode);
        imgReg->Setup(refImg.size(),tmpImg.size(),0,4);
        imgReg->SetTermCrit(criteria);

        ImageRegResults result = imgReg->CreateResult();


        for (int tl = 0; tl < numAngles;++tl)
        {
            result = imgReg->CreateResult();
            if ( warpMode == cv::MOTION_HOMOGRAPHY )
                result.warpMat = cv::Mat::eye(3, 3, CV_32F);
            else
                result.warpMat = cv::Mat::eye(2, 3, CV_32F);


            imgReg->AlignImage(refImg,refMsk,tmpImg,tmpMask,result);

        }
        warpMat = result.warpMat;
        //return cc;

    }

    cv::Mat inWarp;

    if ( warpMode == cv::MOTION_HOMOGRAPHY )
    {
        cv::invert(warpMat,inWarp);
        warpMat = inWarp;

    }
    else
    {
        cv::Mat row = cv::Mat::zeros(1,3,CV_32F);
        row.at<float>(0,2) = 1.0f;
        warpMat.push_back(row);
        cv::invert(warpMat,inWarp);
        warpMat = inWarp.rowRange(0,2);

    }




    return cc;

}

#endif




double alignImageESM(const cv::Mat &refImg, const cv::Mat &refMsk , const cv::Mat &tmpImg, const cv::Mat &tmpMask, int warpMode, int numAngles, const cv::TermCriteria &criteria, cv::Mat &warpMat)
{
    IImageReg::ptr imgReg;

    bool usePyr = true;
    bool useESM = true;

    if (!useESM)
    {
        if (usePyr)
        {
            if ( warpMode != cv::MOTION_HOMOGRAPHY )
                imgReg = ImageRegPyr<IC_Euclidean>::Create();
            else
                imgReg = ImageRegPyr<IC_HOM>::Create();
        }
        else
        {


            if ( warpMode != cv::MOTION_HOMOGRAPHY )
                imgReg = ImageReg<IC_Euclidean>::Create();
            else
                imgReg = ImageReg<IC_HOM>::Create();

        }
    }
    else
    {
        if (usePyr)
        {
            if ( warpMode != cv::MOTION_HOMOGRAPHY )
                imgReg = ImageRegPyr<ESM_Euclidean>::Create();
            else
                imgReg = ImageRegPyr<ESM_HOM>::Create();
        }
        else
        {


            if ( warpMode != cv::MOTION_HOMOGRAPHY )
                imgReg = ImageReg<ESM_Euclidean>::Create();
            else
                imgReg = ImageReg<ESM_HOM>::Create();

        }

    }



    imgReg->Setup(refImg.size(),tmpImg.size(),refImg.type(),4);
    imgReg->SetTermCrit(criteria);


    cv::Point2f center(128,128);
    cv::Point2i offset(0,0);

    cv::Mat imageTrans;

    ImageRegResults result = imgReg->CreateResult();

    for (int tl = 0; tl < numAngles;++tl)
    {
        result = imgReg->CreateResult();

        cv::Mat rot_mat = cv::getRotationMatrix2D( center, (double)tl, 1.0 );

        imgReg->AlignImage(refImg,refMsk,tmpImg,tmpMask,result);

        //ImageRegResults result =  imgReg.AlignImage(refImg,refMsk,tmpImg,tmpMask);
        //resVals.push_back(result.RMSE());
    }

    warpMat = result.warpMat;

}

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



void CreateMaps(const cv::Mat &img, const cv::Mat warpMat, cv::Mat &mapx, cv::Mat &mapy)
{
    cv::Mat wmat = warpMat;
    wmat.convertTo(wmat,CV_32F);

    if (wmat.rows == 2)
    {
        cv::Mat row(1,3,CV_32F);
        wmat.push_back(row);
    }
    mapx = cv::Mat::zeros(img.size(),CV_32F);
    mapy = cv::Mat::zeros(img.size(),CV_32F);

    cv::Mat p(3,1,CV_32F);
    float* pPtr = p.ptr<float>();
    pPtr[0] = 0;
    pPtr[1] = 0;
    pPtr[2] = 1.0f;

    float *mapxPtr,*mapyPtr;

    for (int y = 0; y < img.rows;++y)
    {
        mapxPtr = mapx.ptr<float>(y);
        mapyPtr = mapy.ptr<float>(y);
        for (int x = 0; x < img.cols;++x)
        {
            pPtr[0]+=1.0f;

            const cv::Mat resP = wmat*p;
            const float* resPtr = resP.ptr<float>();

            mapxPtr[x] = resPtr[0];
            mapyPtr[x] = resPtr[1];


        }
        pPtr[1]+=1.0f;

    }


}


void TestRotRemap(const cv::Mat &img, const cv::Mat &warpMat)
{
    cv::Mat warp1;

    timeval tStart;
    gettimeofday(&tStart, NULL);

    for (int tl = 0; tl < 100;++tl)
    {
        cv::warpPerspective(img,warp1,warpMat,img.size());
        cv::warpPerspective(img,warp1,warpMat,img.size());
        cv::warpPerspective(img,warp1,warpMat,img.size());
    }

    timeval tZend;
    gettimeofday(&tZend, NULL);
    double tEWP= (double)(tZend.tv_sec - tStart.tv_sec)*1000.0+ (double)(tZend.tv_usec - tStart.tv_usec)/1000.0;


    cv::Mat mapx,mapy;
    cv::Mat imap1,imap2;
    cv::Mat warp2;


    gettimeofday(&tStart, NULL);

    for (int tl = 0; tl < 100;++tl)
    {
        CreateMaps(img,warpMat,mapx,mapy);
        cv::convertMaps(mapx,mapy,imap1,imap2, CV_16SC2);


        cv::remap(img,warp2,imap1,imap2,CV_INTER_LINEAR);
        cv::remap(img,warp2,imap1,imap2,CV_INTER_LINEAR);
        cv::remap(img,warp2,imap1,imap2,CV_INTER_LINEAR);
    }


    gettimeofday(&tZend, NULL);
    double tERM= (double)(tZend.tv_sec - tStart.tv_sec)*1000.0+ (double)(tZend.tv_usec - tStart.tv_usec)/1000.0;

    Display32FImage("warp",warp1,0,250);
    Display32FImage("remmap",warp2,0,250);

    std::cout << "TimeWarp: " << tEWP << " TimeMap: " << tERM << std::endl;


}





std::string EvalImageReg(std::string name, IImageReg::ptr imgReg, int warpMode, const cv::Mat &refI, const cv::Mat &refM, const cv::Mat &tmpI, const cv::Mat &tmpM)
{
    int numTests = 10;
    ImageRegResults result = imgReg->CreateResult();

    timeval tStart;
    gettimeofday(&tStart, NULL);

    for (int tl = 0; tl < numTests;++tl)
    {
        result = imgReg->CreateResult();
        if ( warpMode == cv::MOTION_HOMOGRAPHY )
            result.warpMat = cv::Mat::eye(3, 3, CV_32F);
        else
            result.warpMat = cv::Mat::eye(2, 3, CV_32F);


        imgReg->AlignImage(refI,refM,tmpI,tmpM,result);

    }

    timeval tZend;
    gettimeofday(&tZend, NULL);
    double tE= (double)(tZend.tv_sec - tStart.tv_sec)*1000.0+ (double)(tZend.tv_usec - tStart.tv_usec)/1000.0;


    cv::Mat warp_matrix  = result.warpMat;

    cv::Mat aligned,alignedMask;

    if (warpMode == cv::MOTION_HOMOGRAPHY)
    {
        // Use Perspective warp when the transformation is a Homography
        cv::warpPerspective (tmpI, aligned, warp_matrix, tmpI.size(), cv::INTER_LINEAR);
        cv::warpPerspective (tmpM, alignedMask, warp_matrix, tmpM.size(), cv::INTER_NEAREST);
    }
    else
    {    // Use Affine warp when the transformation is not a Homography
        cv:: warpAffine(tmpI, aligned, warp_matrix, tmpI.size(), cv::INTER_LINEAR);
        cv:: warpAffine(tmpM, alignedMask, warp_matrix, tmpM.size(), cv::INTER_NEAREST);

    }
    // Show final output
    //cv::namedWindow(name,0);
    //cv::imshow(name, aligned);

    cv::Mat diffImg;
    cv::absdiff(aligned,refI,diffImg);
    cv::multiply(diffImg , alignedMask,diffImg);
    Display32FImage(name, diffImg,0.0f,100.0f);

    cv::Mat sqrdDiff;
    cv::pow(diffImg,2.0,sqrdDiff);

    double ferror = cv::sum(sqrdDiff)[0]/cv::sum(alignedMask)[0];

    std::ostringstream oss;

    oss << name<< " Error: " << ferror << std::endl;
    oss << name << " Time: " << tE << "ms" << " avg: " << tE / (double)numTests << std::endl;
    oss << warp_matrix << std::endl << std::endl;
    return oss.str();
}


int main2(int argc, char *argv[])
{

    ///Read Images
    std::string workingFolder = "/localhome/jordan/Data/temp/MicaSense/Img2";

    std::string imgName = "image";
    std::string maskName = "mask";
    std::string fileExtension = ".png";


    std::string fileName = "IMG_0153_1.tif";
    std::string fileName2 = "IMG_0154_1.tif";


    /// Image scale
    float curScale = 1.0f;

    cv::Mat refImg = cv::imread(workingFolder+"/"+fileName);
    cv::Mat refImgF;
    refImg.convertTo(refImgF,CV_32F);
    cv::cvtColor(refImgF,refImg, cv::COLOR_RGB2GRAY);
    cv::Mat refS;
    cv::resize(refImg,refS,cv::Size(),curScale,curScale);

    cv::Mat refMask;
    cv::threshold(refS,refMask,1,1,cv::THRESH_BINARY);


    cv::Mat tmpImg = cv::imread(workingFolder+"/"+fileName2);
    cv::Mat tmpImgF;
    tmpImg.convertTo(tmpImgF,CV_32F);
    cv::cvtColor(tmpImgF,tmpImg, cv::COLOR_RGB2GRAY);
    cv::Mat tmpS;
    cv::resize(tmpImg,tmpS,cv::Size(),curScale,curScale);

    cv::Mat tmpMask;
    cv::threshold(tmpS,tmpMask,1,1,cv::THRESH_BINARY);


    /// Set registration parameters
    int warp_mode = cv::MOTION_HOMOGRAPHY;

    int number_of_iterations = 50;
    double termination_eps = 0.01;
    int numPyrLevels = 4;
    cv::Size imgSize = refS.size();

    cv::TermCriteria term_criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                   number_of_iterations, termination_eps);


    /// Evaluate
    IImageReg::ptr regESM_PH = ImageRegPyr<ESM_HOM>::Create();
    regESM_PH->Setup(imgSize,imgSize,refS.type(),numPyrLevels);
    regESM_PH->SetTermCrit(term_criteria);

    std::string esmph_res = EvalImageReg("ESM Pyr Hom",regESM_PH,warp_mode,refS,refMask,tmpS,tmpMask);


    IImageReg::ptr regIC_PH = ImageRegPyr<IC_HOM>::Create();
    regIC_PH->Setup(imgSize,imgSize,refS.type(),numPyrLevels);
    regIC_PH->SetTermCrit(term_criteria);

    std::string icph_res = EvalImageReg("IC Pyr Hom",regIC_PH,warp_mode,refS,refMask,tmpS,tmpMask);



    IImageReg::ptr regECC_PH = ImageRegPyrEcc::Create(warp_mode);
    regECC_PH->Setup(imgSize,imgSize,refS.type(),numPyrLevels);
    regECC_PH->SetTermCrit(term_criteria);

    //std::string eccph_res = EvalImageReg("ECC Pyr Hom",regECC_PH,warp_mode,refS,refMask,tmpS,tmpMask);




    warp_mode = cv::MOTION_EUCLIDEAN;

    IImageReg::ptr regESM_PE = ImageRegPyr<ESM_Euclidean>::Create();
    regESM_PE->Setup(imgSize,imgSize,refS.type(),numPyrLevels);
    regESM_PE->SetTermCrit(term_criteria);

    std::string esmpe_res = EvalImageReg("ESM Pyr Euc",regESM_PE,warp_mode,refS,refMask,tmpS,tmpMask);

    IImageReg::ptr regIC_PE = ImageRegPyr<IC_Euclidean>::Create();
    regIC_PE->Setup(imgSize,imgSize,refS.type(),numPyrLevels);
    regIC_PE->SetTermCrit(term_criteria);

    std::string icpe_res = EvalImageReg("IC Pyr Euc",regIC_PE,warp_mode,refS,refMask,tmpS,tmpMask);


    IImageReg::ptr regECC_PE = ImageRegPyrEcc::Create(warp_mode);
    regECC_PE->Setup(imgSize,imgSize,refS.type(),numPyrLevels);
    regECC_PE->SetTermCrit(term_criteria);

    //std::string eccpe_res = EvalImageReg("ECC Pyr Euc",regECC_PE,warp_mode,refS,refMask,tmpS,tmpMask);


    std::cout << esmph_res;
    std::cout << icph_res;
    //std::cout << eccph_res;


    std::cout << esmpe_res;
    std::cout << icpe_res;
    //std::cout << eccpe_res;



    cv::waitKey(0);


    return 1;
}


int mainMica(int argc, char *argv[])
{
    ///Read Images
    //std::string workingFolder = "/localhome/jordan/Data/temp/MicaSense/Img2";

    //    std::string imgName = "image";
    //    std::string maskName = "mask";
    //    std::string fileExtension = ".png";


    //std::string fileName = "IMG_0153_1.tif";
    //std::string fileName2 = "IMG_0154_1.tif";


    std::string workingFolder = "/localhome/jordan/Data/temp/MicaSense";

    std::string fileName = "IMG_0128_1.tif";
    std::string fileName2 = "IMG_0129_1.tif";

    /// Image scale
    float curScale = 1.0f;

    cv::Mat refImg = cv::imread(workingFolder+"/"+fileName);
    cv::Mat refImgF;
    refImg.convertTo(refImgF,CV_32F);
    cv::cvtColor(refImgF,refImg, cv::COLOR_RGB2GRAY);
    cv::Mat refS;
    cv::resize(refImg,refS,cv::Size(),curScale,curScale);

    cv::Mat refMask;
    cv::threshold(refS,refMask,1,1,cv::THRESH_BINARY);


    cv::Mat tmpImg = cv::imread(workingFolder+"/"+fileName2);
    cv::Mat tmpImgF;
    tmpImg.convertTo(tmpImgF,CV_32F);
    cv::cvtColor(tmpImgF,tmpImg, cv::COLOR_RGB2GRAY);
    cv::Mat tmpS;
    cv::resize(tmpImg,tmpS,cv::Size(),curScale,curScale);

    cv::Mat tmpMask;
    cv::threshold(tmpS,tmpMask,1,1,cv::THRESH_BINARY);



    int erosion_size = 1;
    int erosion_type = cv::MORPH_RECT;
    cv::Mat element = cv::getStructuringElement(erosion_type, cv::Size(2*erosion_size + 1, 2*erosion_size+1));
    cv::Mat erefMask;
    cv::erode(refMask,erefMask, element,cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0));
    cv::Mat etmpMask;
    cv::erode(tmpMask,etmpMask, element,cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0));
    refMask = erefMask;
    tmpMask = etmpMask;


    int blurSize = 5;
    cv::Mat brefS;
    cv::blur(refS,brefS,cv::Size(blurSize,blurSize));
    //refS = brefS;
    cv::Mat btmpS;
    cv::blur(tmpS,btmpS,cv::Size(blurSize,blurSize));
    //tmpS = btmpS;


    /// Set registration parameters
    int warp_mode = cv::MOTION_HOMOGRAPHY;

    int number_of_iterations = 50;
    double termination_eps = 0.01;
    int numPyrLevels = 4;
    cv::Size imgSize = refS.size();

    cv::TermCriteria term_criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                   number_of_iterations, termination_eps);


    IImageReg::ptr regESM_PE = ImageRegPyr<ESM_Euclidean>::Create();
    regESM_PE->Setup(imgSize,imgSize,refS.type(),numPyrLevels);
    regESM_PE->SetTermCrit(term_criteria);
    regESM_PE->SetStepFactor(3.0f);
    ImageRegResults resEuc = regESM_PE->CreateResult();


    IImageReg::ptr regESM_PH = ImageReg<ESM_HOM_IO>::Create();
    regESM_PH->Setup(imgSize,imgSize,refS.type(),numPyrLevels);
    term_criteria.epsilon = 0.05;
    regESM_PH->SetTermCrit(term_criteria);
    regESM_PE->SetStepFactor(3.0f);
    ImageRegResults resHom = regESM_PH->CreateResult();


    timeval tStart;
    gettimeofday(&tStart, NULL);


    regESM_PE->AlignImage(refS,refMask,tmpS,tmpMask,resEuc);

    resHom.params.at<double>(0,0) = resEuc.params.at<double>(0,0);
    resHom.params.at<double>(1,0) = resEuc.params.at<double>(1,0);
    resHom.params.at<double>(2,0) = resEuc.params.at<double>(2,0);


    regESM_PH->AlignImage(brefS,refMask,btmpS,tmpMask,resHom);

    timeval tZend;
    gettimeofday(&tZend, NULL);
    double tE= (double)(tZend.tv_sec - tStart.tv_sec)*1000.0+ (double)(tZend.tv_usec - tStart.tv_usec)/1000.0;


    cv::Mat aligned,alignedMask;
    cv::Mat warp_matrix = resHom.warpMat;

    cv::warpPerspective (tmpS, aligned, warp_matrix, tmpS.size(), cv::INTER_LINEAR);
    cv::warpPerspective (tmpMask, alignedMask, warp_matrix, tmpS.size(), cv::INTER_NEAREST);

    // Show final output

    cv::Mat diffImg;
    cv::absdiff(aligned,refS,diffImg);
    cv::multiply(diffImg , alignedMask,diffImg);
    Display32FImage("diff", diffImg,0.0f,100.0f);

    Display32FImage("Ref", refS,0.0f,250.0f);
    Display32FImage("Tmp", tmpS,0.0f,250.0f);
    Display32FImage("Warped", aligned,0.0f,250.0f);
    //Display32FImage("TmpMask", tmpMask,0.0f,1.0f);
    //Display32FImage("WarpedMask", alignedMask,0.0f,1.0f);


    cv::Mat sqrdDiff;
    cv::pow(diffImg,2.0,sqrdDiff);

    double ferror = cv::sum(sqrdDiff)[0]/cv::sum(alignedMask)[0];
    std::cout << " Error: " << ferror << std::endl;
    std::cout << " Time: " << tE << "ms" << std::endl;

    cv::waitKey(0);




    return 1;
}




int main4(int argc, char *argv[])
{
    ///Read Images
    //    std::string workingFolder = "/localhome/jordan/Data/temp/MicaSense/Img2";

    //    std::string fileName = "IMG_0153_1.tif";
    //    std::string fileName2 = "IMG_0154_1.tif";




    /*
    std::string workingFolder = "/localhome/jordan/Misc/MiscProjects/Diss/data";

    std::string fileName = "image0939.png";
    std::string fileName2 = "image0944.png";
    */



    std::string workingFolder = "/localhome/jordan/Data/temp/MicaSense";

    std::string fileName = "IMG_0128_1.tif";
    std::string fileName2 = "IMG_0129_1.tif";


    /// Image scale
    float curScale = 0.5f;

    cv::Mat refImg = cv::imread(workingFolder+"/"+fileName);
    cv::Mat refImgF;
    refImg.convertTo(refImgF,CV_32F);
    if (refImgF.channels() > 1) cv::cvtColor(refImgF,refImg, cv::COLOR_RGB2GRAY); else refImg = refImgF;
    //    refImg = refImgF;
    cv::Mat refS;
    cv::resize(refImg,refS,cv::Size(),curScale,curScale);

    cv::Mat refMask;
    cv::threshold(refS,refMask,1,1,cv::THRESH_BINARY);


    cv::Mat tmpImg = cv::imread(workingFolder+"/"+fileName2);
    cv::Mat tmpImgF;
    tmpImg.convertTo(tmpImgF,CV_32F);
    if (tmpImgF.channels() > 1) cv::cvtColor(tmpImgF,tmpImg, cv::COLOR_RGB2GRAY); else tmpImg = tmpImgF;
    //    tmpImg = tmpImgF;
    cv::Mat tmpS;
    cv::resize(tmpImg,tmpS,cv::Size(),curScale,curScale);

    cv::Mat tmpMask;
    cv::threshold(tmpS,tmpMask,1,1,cv::THRESH_BINARY);

    int erosion_size = 2;
    int erosion_type = cv::MORPH_RECT;
    cv::Mat element = cv::getStructuringElement(erosion_type, cv::Size(2*erosion_size + 1, 2*erosion_size+1));
    cv::Mat erefMask;
    cv::erode(refMask,erefMask, element,cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0));
    cv::Mat etmpMask;
    cv::erode(tmpMask,etmpMask, element,cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0));
    refMask = erefMask;
    tmpMask = etmpMask;


    int blurSize = 5;
    cv::Mat brefS;
    cv::blur(refS,brefS,cv::Size(blurSize,blurSize));
    //refS = brefS;
    cv::Mat btmpS;
    cv::blur(tmpS,btmpS,cv::Size(blurSize,blurSize));
    //tmpS = btmpS;
    //tmpS += 52.0f;

    /// Set registration parameters
    int warp_mode = cv::MOTION_HOMOGRAPHY;

    int number_of_iterations = 100;
    double termination_eps = 0.01;
    int numPyrLevels = 3;
    cv::Size imgSize = refS.size();

    cv::TermCriteria term_criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                   number_of_iterations, termination_eps);



    IImageReg::ptr regESM_PE = ImageReg<IC_Euclidean>::Create();
    //IImageReg::ptr regESM_PE = ImageRegEcc::Create();
    regESM_PE->Setup(imgSize,imgSize,refS.type(),numPyrLevels);
    regESM_PE->SetTermCrit(term_criteria);
    regESM_PE->SetStepFactor(3.0f);
    ImageRegResults resEuc = regESM_PE->CreateResult();

    timeval tStart;
    gettimeofday(&tStart, NULL);


    //for (int tl = 0; tl < 100; ++tl)
    {
        resEuc = regESM_PE->CreateResult();
        regESM_PE->AlignImage(brefS,refMask,btmpS,tmpMask,resEuc);
        //std::cout << "Iter: " << tl << std::endl;
    }

    timeval tZend;
    gettimeofday(&tZend, NULL);
    double tE= (double)(tZend.tv_sec - tStart.tv_sec)*1000.0+ (double)(tZend.tv_usec - tStart.tv_usec)/1000.0;


    cv::Mat aligned,alignedMask;
    cv::Mat warp_matrix = resEuc.warpMat;

    if (warp_matrix.rows == 2)
    {

        cv::warpAffine(tmpS, aligned, warp_matrix, tmpS.size(), cv::INTER_LINEAR);
        cv::warpAffine(tmpMask, alignedMask, warp_matrix, tmpS.size(), cv::INTER_NEAREST);

    }
    else
    {
        cv::warpPerspective(tmpS, aligned, warp_matrix, tmpS.size(), cv::INTER_LINEAR);
        cv::warpPerspective(tmpMask, alignedMask, warp_matrix, tmpS.size(), cv::INTER_NEAREST);
    }
    // Show final output

    cv::Mat diffImg;
    cv::absdiff(aligned,refS,diffImg);
    cv::multiply(diffImg , alignedMask,diffImg);
    cv::multiply(diffImg , refMask,diffImg);
    Display32FImage("diff", diffImg,0.0f,100.0f);

    Display32FImage("Ref", refS,0.0f,250.0f);
    Display32FImage("Tmp", tmpS,0.0f,250.0f);
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



int main5(int argc, char *argv[])
{
    ///Read Images
    ///

    //    std::string workingFolder = "/localhome/jordan/Data/temp/MicaSense/Img2";

    //    std::string fileName = "IMG_0153_1.tif";
    //    std::string fileName2 = "IMG_0154_1.tif";



    //    std::string workingFolder = "/localhome/jordan/Misc/MiscProjects/Diss/data";

    //    std::string fileName = "image0939.png";
    //    std::string fileName2 = "image0944.png";

    //    std::string workingFolder = "/localhome/jordan/Data/temp/MicaSense";

    //    std::string fileName = "IMG_0128_1.tif";
    //    std::string fileName2 = "IMG_0129_1.tif";



    std::string workingFolder = "/rahome/robotlog/rajappa/MicaSense/captures/swap-300319-2-who/0002SET";

    std::string fileName = "IMG_0076_1.tif";
    std::string fileName2 = "IMG_0077_1.tif";




    /// Image scale
    float curScale = 1.0;

    cv::Mat refImg = cv::imread(workingFolder+"/"+fileName);
    cv::Mat refImgF;
    refImg.convertTo(refImgF,CV_32F);
    if (refImgF.channels() > 1) cv::cvtColor(refImgF,refImg, cv::COLOR_RGB2GRAY); else refImg = refImgF;
    //    refImg = refImgF;
    cv::Mat refS;
    cv::resize(refImg,refS,cv::Size(),curScale,curScale);

    cv::Mat refMask;
    cv::threshold(refS,refMask,1,1,cv::THRESH_BINARY);


    cv::Mat tmpImg = cv::imread(workingFolder+"/"+fileName2);
    /*cv::Mat transMat = cv::Mat::eye(3,3,CV_64F);
    transMat.at<double>(0,2) = 1;
    transMat.at<double>(1,2) = 1;
    cv::warpPerspective(refImg, tmpImg, transMat, refImg.size(), cv::INTER_LINEAR);*/
    cv::Mat tmpImgF;
    tmpImg.convertTo(tmpImgF,CV_32F);
    if (tmpImgF.channels() > 1) cv::cvtColor(tmpImgF,tmpImg, cv::COLOR_RGB2GRAY); else tmpImg = tmpImgF;
    //    tmpImg = tmpImgF;
    cv::Mat tmpS;
    cv::resize(tmpImg,tmpS,cv::Size(),curScale,curScale);

    cv::Mat tmpMask;
    cv::threshold(tmpS,tmpMask,1,1,cv::THRESH_BINARY);

    int erosion_size = 2;
    int erosion_type = cv::MORPH_RECT;
    cv::Mat element = cv::getStructuringElement(erosion_type, cv::Size(2*erosion_size + 1, 2*erosion_size+1));
    cv::Mat erefMask;
    cv::erode(refMask,erefMask, element,cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0));
    cv::Mat etmpMask;
    cv::erode(tmpMask,etmpMask, element,cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0));
    refMask = erefMask;
    tmpMask = etmpMask;


    int blurSize = 5;
    cv::Mat brefS;
    cv::blur(refS,brefS,cv::Size(blurSize,blurSize));
    //refS = brefS;
    cv::Mat btmpS;
    cv::blur(tmpS,btmpS,cv::Size(blurSize,blurSize));
    //tmpS = btmpS;
    //tmpS += 52.0f;

    /// Set registration parameters
    int warp_mode = cv::MOTION_HOMOGRAPHY;

    int number_of_iterations = 200;
    double termination_eps = 0.05;
    int numPyrLevels = 4;
    cv::Size imgSize = refS.size();

    cv::TermCriteria term_criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                   number_of_iterations, termination_eps);



    IImageReg::ptr regESM_PE = ImageRegPyr<ESM_Euclidean>::Create();
    //IImageReg::ptr regESM_PE = ImageRegEcc::Create();
    regESM_PE->Setup(imgSize,imgSize,refS.type(),numPyrLevels);
    regESM_PE->SetTermCrit(term_criteria);
    regESM_PE->SetStepFactor(3.0f);
    ImageRegResults resEuc = regESM_PE->CreateResult();

    IImageReg::ptr regESM_PH = ImageReg<ESM_HOM_IO>::Create();
    //IImageReg::ptr regESM_PE = ImageRegEcc::Create();
    regESM_PH->Setup(imgSize,imgSize,refS.type(),numPyrLevels);
    regESM_PH->SetTermCrit(term_criteria);
    regESM_PH->SetStepFactor(3.0f);
    ImageRegResults resHom = regESM_PH->CreateResult();


    timeval tStart;
    gettimeofday(&tStart, NULL);

    resEuc = regESM_PE->CreateResult();

    resEuc.params.at<double>(2,0) = 0.4;

    regESM_PE->AlignImage(refS,refMask,tmpS,tmpMask,resEuc);

    resHom.params.at<double>(0,0) = resEuc.params.at<double>(0,0);
    resHom.params.at<double>(1,0) = resEuc.params.at<double>(1,0);
    resHom.params.at<double>(2,0) = resEuc.params.at<double>(2,0);

    regESM_PH->AlignImage(refS,refMask,tmpS,tmpMask,resHom);


    timeval tZend;
    gettimeofday(&tZend, NULL);
    double tE= (double)(tZend.tv_sec - tStart.tv_sec)*1000.0+ (double)(tZend.tv_usec - tStart.tv_usec)/1000.0;

    BlockReg<ESM_HOM,BS_EuclidDist>::ptr blockReg = BlockReg<ESM_HOM,BS_EuclidDist>::Create();
    //IImageReg::ptr regESM_PE = ImageRegEcc::Create();
    blockReg->Setup(imgSize,imgSize,refS.type(),numPyrLevels);
    term_criteria.epsilon = 0.3;
    blockReg->SetTermCrit(term_criteria);
    blockReg->SetStepFactor(3.0f);
    blockReg->SetBlockParams(imgSize,cv::Size(128,128),cv::Point2i(64,64));
    blockReg->SetDistThreshold(1.0f);
    ImageRegResults resBlock = blockReg->CreateResult();

    resBlock.params.at<double>(0,0) = resHom.params.at<double>(0,0);
    resBlock.params.at<double>(1,0) = resHom.params.at<double>(1,0);
    resBlock.params.at<double>(2,0) = resHom.params.at<double>(2,0);
    resBlock.params.at<double>(3,0) = resHom.params.at<double>(3,0);
    resBlock.params.at<double>(4,0) = resHom.params.at<double>(4,0);
    resBlock.params.at<double>(5,0) = resHom.params.at<double>(5,0);
    resBlock.params.at<double>(6,0) = resHom.params.at<double>(6,0);
    resBlock.params.at<double>(7,0) = resHom.params.at<double>(7,0);


    //blockReg->AlignImage(refS,refMask,tmpS,tmpMask,resBlock);


    cv::Mat aligned,alignedMask;
    cv::Mat warp_matrix = resEuc.warpMat;

    if (warp_matrix.rows == 2)
    {

        cv::warpAffine(tmpS, aligned, warp_matrix, tmpS.size(), cv::INTER_LINEAR);
        cv::warpAffine(tmpMask, alignedMask, warp_matrix, tmpS.size(), cv::INTER_NEAREST);

    }
    else
    {
        cv::warpPerspective(tmpS, aligned, warp_matrix, tmpS.size(), cv::INTER_LINEAR);
        cv::warpPerspective(tmpMask, alignedMask, warp_matrix, tmpS.size(), cv::INTER_NEAREST);
    }
    // Show final output

    cv::Mat diffImg;
    cv::absdiff(aligned,refS,diffImg);
    cv::multiply(diffImg , alignedMask,diffImg);
    cv::multiply(diffImg , refMask,diffImg);
    Display32FImage("diff", diffImg,0.0f,100.0f);

    Display32FImage("Ref", refS,0.0f,250.0f);
    Display32FImage("Tmp", tmpS,0.0f,250.0f);
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


template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}
std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

double ReadYaw(std::string fname)
{
    std::ifstream inFile;

    inFile.open(fname);
    if (!inFile) {
        std::cout << "Unable to open file";
        exit(1); // terminate with error
    }

    std::string tstring;

    inFile >> tstring;

    std::vector<std::string> tokens = split(tstring,',');


    return std::atof(tokens[2].c_str());

}

void ReadImage(std::string workingFolder, int i, int erosion_size, cv::Mat &img, cv::Mat &mask, double &yaw)
{
    int imageType = 3;
    std::ostringstream ossr;
    ossr << "IMG_" <<  std::setfill('0') << std::setw(4) << i << "_" << imageType << ".tif";

    std::ostringstream ossrc;
    ossrc << "IMG_" <<  std::setfill('0') << std::setw(4) << i << "_" << imageType << ".csv";

    std::string imgName = workingFolder+"/"+ossr.str();
    std::string csvName = workingFolder+"/"+ossrc.str();

    cv::Mat refImg = cv::imread(imgName);
    cv::Mat refImgF;
    refImg.convertTo(refImgF,CV_32F);
    if (refImgF.channels() > 1) cv::cvtColor(refImgF,refImg, cv::COLOR_RGB2GRAY); else refImg = refImgF;
    cv::Mat refMask;
    cv::threshold(refImg,refMask,1,1,cv::THRESH_BINARY);

    img = refImg;
    mask = refMask;
    yaw = ReadYaw(csvName);


    if (erosion_size > 0)
    {
        int erosion_type = cv::MORPH_RECT;
        cv::Mat element = cv::getStructuringElement(erosion_type, cv::Size(2*erosion_size + 1, 2*erosion_size+1));


        cv::Mat etmpMask;
        cv::erode(refMask,etmpMask, element,cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0));
        //refMask = erefMask;
        mask = etmpMask;
    }


}


void ReadImageMC(std::string workingFolder, int i, int erosion_size, cv::Mat &img, cv::Mat &mask, double &yaw)
{

    std::vector<int> imgTypes{ 2,3,4,5};


    std::vector<cv::Mat> refImages;
    std::vector<cv::Mat> maskImages;

    for (int tl = 0; tl < imgTypes.size();tl++)
    {

        std::ostringstream ossr;
        ossr << "IMG_" <<  std::setfill('0') << std::setw(4) << i << "_" << imgTypes[tl] << ".tif";


        std::ostringstream ossrc;
        ossrc << "IMG_" <<  std::setfill('0') << std::setw(4) << i << "_" << imgTypes[tl] << ".csv";

        std::string imgName = workingFolder+"/"+ossr.str();
        std::string csvName = workingFolder+"/"+ossrc.str();

        cv::Mat refImg = cv::imread(imgName);
        cv::Mat refImgF;
        refImg.convertTo(refImgF,CV_32F);
        if (refImgF.channels() > 1) cv::cvtColor(refImgF,refImg, cv::COLOR_RGB2GRAY); else refImg = refImgF;
        //cv::Mat refMask;
        //cv::threshold(refImg,refMask,1,1,cv::THRESH_BINARY);
        cv::Mat refMask = cv::Mat::ones(refImg.size(),CV_32F);

        //img = refImg;
        //mask = refMask;
        yaw = ReadYaw(csvName);


        if (erosion_size > 0)
        {
            int erosion_type = cv::MORPH_RECT;
            cv::Mat element = cv::getStructuringElement(erosion_type, cv::Size(2*erosion_size + 1, 2*erosion_size+1));


            cv::Mat etmpMask;
            cv::erode(refMask,etmpMask, element,cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0));
            //refMask = erefMask;
            mask = etmpMask;
        }

        refImages.emplace_back(refImg);
        maskImages.emplace_back(refMask);

    }

    cv::merge(refImages,img);
    cv::merge(maskImages,mask);


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

cv::Mat GetYawInitialParam(double yaw, cv::Size imgSize)
{
    cv::Mat rmat = cv::Mat::eye(3,3,CV_64F);
    rmat.at<double>(0,0) = cos(yaw);
    rmat.at<double>(1,1) = cos(yaw);
    rmat.at<double>(0,1) = -sin(yaw);
    rmat.at<double>(1,0) = sin(yaw);

    cv::Mat tmat = cv::Mat::eye(3,3,CV_64F);
    cv::Mat tmati = cv::Mat::eye(3,3,CV_64F);
    tmat.at<double>(0,2) = imgSize.width/2.0;
    tmat.at<double>(1,2) = imgSize.height/2.0;

    tmati.at<double>(0,2) = -imgSize.width/2.0;
    tmati.at<double>(1,2) = -imgSize.height/2.0;

    cv::Mat res;
    res = tmat*rmat*tmati;

    return res;

}





int main(int argc, char *argv[])
{
    ///Read Images
    ///

    std::string workingFolder = "/localhome/jordan/Data/Mica/0019SET";

    int imageType = 3;
    int number_of_iterations = 50;
    double termination_eps = 0.05;
    int numPyrLevels = 4;
    cv::TermCriteria term_criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                   number_of_iterations, termination_eps);


    int cvMatType = CV_32FC1;

    cv::Size imgSize(1280,960);// = ref.size();
    //cv::Size imgSize(960,1280);// = ref.size();

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


    std::vector<cv::Mat> params;
    std::vector<cv::Mat> mat;
    cv::Mat curWarp = cv::Mat::eye(3,3,CV_64F);


    for (int i  = 50; i < 167;++i)
    {

        std::cout << std::endl << " ### Images: " << i << " to " << i+1 << std::endl;

        //IMG_0165_5.tif
        cv::Mat refImg;
        cv::Mat refMask;
        double refYaw;

        ReadImage(workingFolder,i,2,refImg,refMask,refYaw);




        cv::Mat tmpImg;
        cv::Mat tmpMask;
        double tmpYaw;

        ReadImage(workingFolder,i+1,2,tmpImg,tmpMask,tmpYaw);

        double diffYaw = tmpYaw-refYaw;
        //diffYaw = 0;

        int blurSize = 0;
        cv::Mat brefImg;
        //cv::blur(refImg,brefImg,cv::Size(blurSize,blurSize));
        cv::GaussianBlur(refImg,brefImg,cv::Size(blurSize,blurSize),3.0);
        cv::Mat btmpImg;
        //cv::blur(tmpImg,btmpImg,cv::Size(blurSize,blurSize));
        cv::GaussianBlur(tmpImg,btmpImg,cv::Size(blurSize,blurSize),3.0);



        float initTransScale = 0.25;
        cv::Mat refIs,refMs,tmpIs,tmpMs;
        cv::resize(refImg,refIs,cv::Size(),initTransScale,initTransScale);
        cv::resize(tmpImg,tmpIs,cv::Size(),initTransScale,initTransScale);

        cv::Mat tmpIsRot;
        cv::Mat trotMat = GetYawInitialParam(diffYaw,tmpIs.size()).rowRange(0,2);
        cv::warpAffine(tmpIs,tmpIsRot,trotMat,tmpIs.size());

        cv::threshold(refIs,refMs,1,1,cv::THRESH_BINARY);
        cv::threshold(tmpIs,tmpMs,1,1,cv::THRESH_BINARY);


        AlignedMat::ptr arI = AlignedMat::Create(refIs);
        AlignedMat::ptr arM = AlignedMat::Create(refMs);
        AlignedMat::ptr atI = AlignedMat::Create(tmpIsRot);
        AlignedMat::ptr atM = AlignedMat::Create(tmpMs);

        timeval tStart;
        gettimeofday(&tStart, NULL);

        cv::Point2i initTrans = GetInitialTrans(arI->mat_,arM->mat_,atI->mat_,atM->mat_,8,16);

        trotMat.at<double>(0,2) += (double)initTrans.x;
        trotMat.at<double>(1,2) += (double)initTrans.y;

        cv::warpAffine(tmpIs,tmpIsRot,trotMat,tmpIs.size());


        timeval tZend2;
        gettimeofday(&tZend2, NULL);
        double tE2= (double)(tZend2.tv_sec - tStart.tv_sec)*1000.0+ (double)(tZend2.tv_usec - tStart.tv_usec)/1000.0;
        std::cout << "###############"<< std::endl;
        std::cout << " Initial RotMat: " << trotMat <<  std::endl;

        std::cout << " Initial trans time: " << tE2 << "  InitTrans: " << initTrans <<  std::endl;
        std::cout << "###############"<< std::endl;


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


    }










    return 1;
}










