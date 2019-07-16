#include <cslibs_image_align/imagereg.h>
#include "gtest/gtest.h"

namespace {



cv::Mat ToHom(cv::Mat in)
{
    cv::Mat res = in.clone();
    if (in.type() != CV_32F) in.convertTo(res,CV_32F);
    cv::Mat row = cv::Mat::zeros(1,3,CV_32F);
    row.at<float>(0,2) = 1.0f;

    res.push_back(row);

    return res;
}

class ImageTest : public ::testing::Test {
 protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  ImageTest() {
     // You can do set-up work for each test here.
  }

  ~ImageTest() override {
     // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
     // Code here will be called immediately after the constructor (right
     // before each test).
      img = cv::imread("test_transform.png");
      img.convertTo(img,CV_32F);
      if (img.channels() > 1) cv::cvtColor(img,img, cv::COLOR_RGB2GRAY);
  }

  void TearDown() override {
     // Code here will be called immediately after each test (right
     // before the destructor).
  }

  // Objects declared here can be used by all tests in the test suite for Foo.
  cv::Mat img;
};


TEST_F(ImageTest, LoadImageTest) {


    // Test if image was loaded
    EXPECT_EQ(img.empty(),false);
}

TEST_F(ImageTest, ImageAlignTestConvergeF) {

    cv::Mat mask = cv::Mat::ones(img.size(),img.type() );


    float angleDeg = 15.0;
    float angleRad = angleDeg * (CV_PI/180.0);
    float x = 10;
    float y = 20;
    cv::Mat warpMat = cv::getRotationMatrix2D(cv::Point2f(img.cols/2,img.rows/2),angleDeg,1.0);
    warpMat.at<double>(0,2) = x;
    warpMat.at<double>(1,2) = y;



    std::cout << img.size() << std::endl;

    cv::Mat warpedImg;
    cv::warpAffine(img,warpedImg,warpMat,img.size());

    cv::Mat warpedMask;
    cv::threshold(warpedImg,warpedMask,1.0,1.0,CV_THRESH_BINARY);

    int erosion_size = 3;
    int erosion_type = cv::MORPH_RECT;
    cv::Mat element = cv::getStructuringElement(erosion_type, cv::Size(2*erosion_size + 1, 2*erosion_size+1));

    cv::Mat etmpMask;
    cv::erode(warpedMask,etmpMask, element,cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0));
    warpedMask = etmpMask;





    int number_of_iterations = 50;
    double termination_eps = 0.05;
    cv::TermCriteria term_criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, number_of_iterations, termination_eps);

    auto regESM_Euc = ImageRegPyr<ESM_Euclidean>::Create();

    regESM_Euc->Setup(img.size(),img.size(),img.type(),4);
    regESM_Euc->SetTermCrit(term_criteria);
    auto result = regESM_Euc->CreateResult();


    regESM_Euc->AlignImage(img,mask,warpedImg,warpedMask,result);

    cv::Mat alignedImg,alignedMask;


    cv::warpAffine(warpedImg, alignedImg, result.warpMat, img.size(), cv::INTER_LINEAR);
    cv::warpAffine(warpedMask, alignedMask, result.warpMat, img.size(), cv::INTER_NEAREST);



    cv::Mat diffImg;
    cv::absdiff(alignedImg,img,diffImg);
    cv::multiply(diffImg , alignedMask,diffImg);
    cv::multiply(diffImg , mask,diffImg);


    std::cout << "######## GT MAT:" << std::endl;
    std::cout << warpMat << std::endl;

    std::cout << "######## RES MAT:" << std::endl;
    std::cout << result.warpMat << std::endl;


    cv::Mat warpMatH = ToHom(cv::getRotationMatrix2D(cv::Point2f(img.cols/2,img.rows/2),15.0,1.0));
    cv::Mat warpMatHI = ToHom(cv::getRotationMatrix2D(cv::Point2f(img.cols/2,img.rows/2),-15.0,1.0));
    cv::Mat resMatH = ToHom(result.warpMat);
    cv::Mat resMatT = cv::Mat::eye(3,3,CV_32F);
    resMatT.at<float>(0,2) = resMatH.at<float>(0,2);
    resMatT.at<float>(1,2) = resMatH.at<float>(1,2);
    cv::Mat resCor = warpMatH*resMatT*warpMatHI;

    std::cout << "######## RES MAT:" << std::endl;
    std::cout << resCor << std::endl;


    std::cout << "####### Parameters found : ground truth " << std::endl;
    std::cout << resCor.at<float>(0,2) << " : " << -x << std::endl;
    std::cout << resCor.at<float>(1,2) << " : " << -y << std::endl;
    std::cout << std::acos(resMatH.at<float>(0,0)) << " : " << angleRad << std::endl;


    EXPECT_NEAR(resCor.at<float>(0,2),-x,0.1);
    EXPECT_NEAR(resCor.at<float>(1,2),-y,0.1);
    EXPECT_NEAR(std::acos(resMatH.at<float>(0,0)),angleRad,0.1);


    /*
    cv::namedWindow("img",0);
    cv::imshow("img",img*(1.0/255.0));

    cv::namedWindow("warpedImg",0);
    cv::imshow("warpedImg",warpedImg*(1.0/255.0));

    cv::namedWindow("warpedMask",0);
    cv::imshow("warpedMask",warpedMask);

    cv::namedWindow("alignedImg",0);
    cv::imshow("alignedImg",alignedImg*(1.0/255.0));


    cv::namedWindow("diffImg",0);
    cv::imshow("diffImg",diffImg*(1.0/255.0));

    cv::waitKey();
    */

}



TEST(ImageAlignTest, ImageAlignTestConverge) {

    cv::Mat img = cv::imread("test_transform.png");
    img.convertTo(img,CV_32F);
    if (img.channels() > 1) cv::cvtColor(img,img, cv::COLOR_RGB2GRAY);

    cv::Mat mask = cv::Mat::ones(img.size(),img.type() );


    // Test if image was loaded
    EXPECT_EQ(img.empty(),false);

    float angleDeg = 15.0;
    float angleRad = angleDeg * (CV_PI/180.0);
    float x = 10;
    float y = 20;
    cv::Mat warpMat = cv::getRotationMatrix2D(cv::Point2f(img.cols/2,img.rows/2),angleDeg,1.0);
    warpMat.at<double>(0,2) = x;
    warpMat.at<double>(1,2) = y;



    std::cout << img.size() << std::endl;

    cv::Mat warpedImg;
    cv::warpAffine(img,warpedImg,warpMat,img.size());

    cv::Mat warpedMask;
    cv::threshold(warpedImg,warpedMask,1.0,1.0,CV_THRESH_BINARY);

    int erosion_size = 3;
    int erosion_type = cv::MORPH_RECT;
    cv::Mat element = cv::getStructuringElement(erosion_type, cv::Size(2*erosion_size + 1, 2*erosion_size+1));

    cv::Mat etmpMask;
    cv::erode(warpedMask,etmpMask, element,cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0));
    warpedMask = etmpMask;





    int number_of_iterations = 50;
    double termination_eps = 0.05;
    cv::TermCriteria term_criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, number_of_iterations, termination_eps);

    auto regESM_Euc = ImageRegPyr<ESM_Euclidean>::Create();

    regESM_Euc->Setup(img.size(),img.size(),img.type(),4);
    regESM_Euc->SetTermCrit(term_criteria);
    auto result = regESM_Euc->CreateResult();


    regESM_Euc->AlignImage(img,mask,warpedImg,warpedMask,result);

    cv::Mat alignedImg,alignedMask;


    cv::warpAffine(warpedImg, alignedImg, result.warpMat, img.size(), cv::INTER_LINEAR);
    cv::warpAffine(warpedMask, alignedMask, result.warpMat, img.size(), cv::INTER_NEAREST);



    cv::Mat diffImg;
    cv::absdiff(alignedImg,img,diffImg);
    cv::multiply(diffImg , alignedMask,diffImg);
    cv::multiply(diffImg , mask,diffImg);


    std::cout << "######## GT MAT:" << std::endl;
    std::cout << warpMat << std::endl;

    std::cout << "######## RES MAT:" << std::endl;
    std::cout << result.warpMat << std::endl;


    cv::Mat warpMatH = ToHom(cv::getRotationMatrix2D(cv::Point2f(img.cols/2,img.rows/2),15.0,1.0));
    cv::Mat warpMatHI = ToHom(cv::getRotationMatrix2D(cv::Point2f(img.cols/2,img.rows/2),-15.0,1.0));
    cv::Mat resMatH = ToHom(result.warpMat);
    cv::Mat resMatT = cv::Mat::eye(3,3,CV_32F);
    resMatT.at<float>(0,2) = resMatH.at<float>(0,2);
    resMatT.at<float>(1,2) = resMatH.at<float>(1,2);
    cv::Mat resCor = warpMatH*resMatT*warpMatHI;

    std::cout << "######## RES MAT:" << std::endl;
    std::cout << resCor << std::endl;


    std::cout << "####### Parameters found : ground truth " << std::endl;
    std::cout << resCor.at<float>(0,2) << " : " << -x << std::endl;
    std::cout << resCor.at<float>(1,2) << " : " << -y << std::endl;
    std::cout << std::acos(resMatH.at<float>(0,0)) << " : " << angleRad << std::endl;


    EXPECT_NEAR(resCor.at<float>(0,2),-x,0.1);
    EXPECT_NEAR(resCor.at<float>(1,2),-y,0.1);
    EXPECT_NEAR(std::acos(resMatH.at<float>(0,0)),angleRad,0.1);


    /*
    cv::namedWindow("img",0);
    cv::imshow("img",img*(1.0/255.0));

    cv::namedWindow("warpedImg",0);
    cv::imshow("warpedImg",warpedImg*(1.0/255.0));

    cv::namedWindow("warpedMask",0);
    cv::imshow("warpedMask",warpedMask);

    cv::namedWindow("alignedImg",0);
    cv::imshow("alignedImg",alignedImg*(1.0/255.0));


    cv::namedWindow("diffImg",0);
    cv::imshow("diffImg",diffImg*(1.0/255.0));

    cv::waitKey();
    */

}

}  // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
