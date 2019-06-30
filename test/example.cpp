
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



int main4(int argc, char *argv[])
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

    ReadImage("img1.png",1,refImg,refMask);



    cv::Mat tmpImg;
    cv::Mat tmpMask;

    ReadImage("img2.png",1,tmpImg,tmpMask);



    ///Setup ImageReg
    int number_of_iterations = 50;
    double termination_eps = 0.05;
    int numPyrLevels = 4;
    cv::TermCriteria term_criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, number_of_iterations, termination_eps);

    IImageReg::ptr regESM_PE = ImageRegPyr<ESM_HOM>::Create();
    regESM_PE->Setup(refImg.size(),refImg.size(),refImg.type(),numPyrLevels);
    regESM_PE->SetTermCrit(term_criteria);
    ImageRegResults res = regESM_PE->CreateResult();

    timeval tStart;
    gettimeofday(&tStart, NULL);


    regESM_PE->AlignImage(refImg,refMask,tmpImg,tmpMask,res);


    timeval tZend;
    gettimeofday(&tZend, NULL);
    double tE= (double)(tZend.tv_sec - tStart.tv_sec)*1000.0+ (double)(tZend.tv_usec - tStart.tv_usec)/1000.0;



    cv::Mat aligned,alignedMask;
    cv::Mat warp_matrix = res.warpMat;


    cv::warpPerspective(tmpImg, aligned, warp_matrix, tmpImg.size(), cv::INTER_LINEAR);
    cv::warpPerspective(tmpMask, alignedMask, warp_matrix, tmpImg.size(), cv::INTER_NEAREST);
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

    ReadImage("img1.png",1,refImg,refMask);



    cv::Mat tmpImg;
    cv::Mat tmpMask;

    ReadImage("img2.png",1,tmpImg,tmpMask);



    ///Setup ImageReg
    int number_of_iterations = 50;
    double termination_eps = 0.05;
    int numPyrLevels = 4;
    cv::TermCriteria term_criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, number_of_iterations, termination_eps);

    IImageReg::ptr regESM_PE = BlockReg<ESM_Euclidean,BS_EuclidDist>::Create();
    regESM_PE->Setup(refImg.size(),refImg.size(),refImg.type(),numPyrLevels);
    regESM_PE->SetTermCrit(term_criteria);
    ImageRegResults resEuc = regESM_PE->CreateResult();

    timeval tStart;
    gettimeofday(&tStart, NULL);


    regESM_PE->AlignImage(refImg,refMask,tmpImg,tmpMask,resEuc);


    timeval tZend;
    gettimeofday(&tZend, NULL);
    double tE= (double)(tZend.tv_sec - tStart.tv_sec)*1000.0+ (double)(tZend.tv_usec - tStart.tv_usec)/1000.0;



    cv::Mat aligned,alignedMask;
    cv::Mat warp_matrix = resEuc.warpMat;


    cv::warpAffine(tmpImg, aligned, warp_matrix, tmpImg.size(), cv::INTER_LINEAR);
    cv::warpAffine(tmpMask, alignedMask, warp_matrix, tmpImg.size(), cv::INTER_NEAREST);
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









