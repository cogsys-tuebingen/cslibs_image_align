
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/video/video.hpp>

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




constexpr bool doCrop = true;

void ReadImage(std::string imgName, int erosion_size, cv::Mat &img, cv::Mat &mask)
{

    cv::Mat refImg = cv::imread(imgName);



    if (doCrop)
    {
    //// for image crop
    cv::Rect r( 160, 150, 288, 288 );

    cv::Mat C;
    refImg(r).copyTo(C);
    refImg = C;

    /// end image crop
    }
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


void TestAlign(IImageReg::ptr proc, std::string name, cv::Mat &refImg, cv::Mat &refMask, cv::Mat &tmpImg, cv::Mat &tmpMask, ImageRegResults &results)
{
    timeval tStart;
     gettimeofday(&tStart, NULL);


    proc->AlignImage(refImg,refMask,tmpImg,tmpMask,results);


    timeval tZend;
    gettimeofday(&tZend, NULL);
    double tE= (double)(tZend.tv_sec - tStart.tv_sec)*1000.0+ (double)(tZend.tv_usec - tStart.tv_usec)/1000.0;



    cv::Mat aligned,alignedMask;
    cv::Mat warp_matrix = results.warpMat;

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

    Display32FImage("Residual " + name, diffImg,0.0f,100.0f);

    Display32FImage("Warped " + name, aligned,0.0f,250.0f);

    cv::Mat sqrdDiff;
    cv::pow(diffImg,2.0,sqrdDiff);

    double ferror = cv::sum(sqrdDiff)[0]/cv::sum(alignedMask)[0];
    std::cout << " Results for " << name << std::endl;
    std::cout << " Error: " << ferror << std::endl;
    std::cout << " Time: " << tE << "ms" << std::endl;
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

    /// Test block wise registration
    auto regBlock = BlockReg<ESM_Euclidean,BS_EuclidDist>::Create();
    IImageReg::ptr regESM_PE = regBlock;
    regBlock->SetBlockParams(refImg.size(),cv::Size(64,64),cv::Point2i(32,32));
    //regBlock->SetBlockParams(refImg.size(),cv::Size(32,32),cv::Point2i(16,16));
    regBlock->SetDistThreshold(0.3);

    regESM_PE->Setup(refImg.size(),refImg.size(),refImg.type(),numPyrLevels);
    regESM_PE->SetTermCrit(term_criteria);
    ImageRegResults resEuc = regESM_PE->CreateResult();

    resEuc.params.at<double>(0,0) = initial.x;
    resEuc.params.at<double>(1,0) = initial.y;

    TestAlign(regESM_PE," with blocks",refImg,refMask,tmpImg,tmpMask,resEuc);



    IImageReg::ptr regESM_EU = ImageRegPyr<ESM_Euclidean>::Create();
    regESM_EU->Setup(refImg.size(),refImg.size(),refImg.type(),numPyrLevels);
    regESM_EU->SetTermCrit(term_criteria);

    ImageRegResults resPyr = regESM_EU->CreateResult();


    TestAlign(regESM_EU," pyramide",refImg,refMask,tmpImg,tmpMask,resPyr);




    IImageReg::ptr  regESM_RE = ImageRegRes<ESM_Euclidean>::Create();
    regESM_RE->Setup(refImg.size(),refImg.size(),refImg.type(),numPyrLevels);
    regESM_RE->SetTermCrit(term_criteria);

    ImageRegResults resRes = regESM_RE->CreateResult();

    resRes.params.at<double>(0,0) = initial.x;
    resRes.params.at<double>(1,0) = initial.y;

    TestAlign(regESM_RE," residual",refImg,refMask,tmpImg,tmpMask,resRes);


    char key = 0;

    while (key != 'q') key = cv::waitKey(0);



    return 1;
}









