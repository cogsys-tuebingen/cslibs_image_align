#ifndef BLOCKREG_H
#define BLOCKREG_H

#include "imagereg.h"
#include <iomanip>



/*!
  Class for storing the data of one image sub block
*/
class BlockInfo
{
public:

    // All mats are float
    static constexpr int CV_MAT_TYPE_ = CV_32F;
    typedef float MAT_TYPE_;

    /// Blocks are constructed at specific positions
    BlockInfo(int params, cv::Point2i pos, cv::Point2i bsize)
    {
        blockPos = pos;
        blockSize = bsize;

        hess = cv::Mat(params,params,CV_MAT_TYPE_);
        hessI = cv::Mat(params,params,CV_MAT_TYPE_);
        jacDiff = cv::Mat(params,1,CV_MAT_TYPE_);
        currentParams = cv::Mat(params,1,CV_MAT_TYPE_);

        hessSum = cv::Mat(params,params,CV_MAT_TYPE_);
        jacDiffSum = cv::Mat(params,1,CV_MAT_TYPE_);



        numPixelSum = 0;
        numPixels = 0;
        isSelected = false;
        clusterValue = 0;
        isValid = false;

    }

    /// Reset Data
    void SetZero()
    {
        hess.setTo(0);
        hessI.setTo(0);
        jacDiff.setTo(0);

        hessSum.setTo(0);
        jacDiffSum.setTo(0);

        currentParams.setTo(0);

        selectedBlock.clear();
        numPixels = 0;
        isSelected = false;
        clusterValue = 0;
        isValid = false;
    }

    /// Add another block to this one, scaled with weight
    void Add(BlockInfo* b, float weight)
    {
        if (weight <= 0) return;
        hessSum += b->hess*weight;
        jacDiffSum += b->jacDiff*weight;
        numPixelSum += b->numPixels;
        clusterValue += weight;
        selectedBlock.push_back(b);

    }

    /*
    void operator += (BlockInfo &b)
    {
        hess += b.hess;
        jacDiff += b.jacDiff;
        numPixels += b.numPixels;

    }
    */

    /// Calculate params for this block
    void CalcParams()
    {
        cv::invert(hess,hessI);
        currentParams = hessI*jacDiff;
        isValid = IsValid();
    }

    /// Check if block contains pixels or has invalid parameters
    bool IsValid()
    {
        if (numPixels == 0) return false;
        //if (!isValid) return false;
        for (int r = 0; r < currentParams.rows; ++r)
        {
            if (std::isnan(currentParams.at<MAT_TYPE_>(r,0))) return false;
            if (std::isinf(currentParams.at<MAT_TYPE_>(r,0))) return false;

        }
        return true;
    }

    /// Calculate parameters for sum over all added blocks
    cv::Mat CalcSumParams()
    {
        cv::Mat hInv;
        cv::Mat hessD;
        cv::Mat jacDiffD;
        hessSum.convertTo(hessD,CV_64F);
        jacDiffSum.convertTo(jacDiffD,CV_64F);
        //std::cout << "Hess: " << hessD;
        //std::cout << "JacD: " << jacDiffD;

        cv::invert(hessD,hInv);
        return hInv*jacDiffD;
    }

    /// Data members
    cv::Mat hess;
    cv::Mat hessI;
    cv::Mat jacDiff;

    cv::Mat hessSum;
    cv::Mat jacDiffSum;

    cv::Mat currentParams;
    cv::Point2i blockPos;
    cv::Point2i blockSize;

    std::vector<BlockInfo*> selectedBlock;

    int numPixels;
    int numPixelSum;
    bool isSelected;
    double clusterValue;
    bool isValid;

};


/*!
    Select blocks by euclidean distance implementation
 */
class BS_EuclidDist
{
public:

    inline float GetWeightTukey(const float &val, const float &param) const
    {
        const float absval = std::abs(val);
        if (absval > param) return 0;
        else
        {
        const float invVal = val/param;
        return (1.0f- (invVal)*(invVal))*(1.0f - (invVal)*(invVal));
        }
    }

    cv::Mat Filter(std::vector<BlockInfo> &infos)
    {
        BlockInfo *selected, *testing;

        cv::Mat sPars;
        cv::Mat tPars;
        cv::Mat diff;
        float distance;
        float tweight;
        float maxWeight = 0;
        BlockInfo *maxBlock = nullptr;

        for (unsigned int t = 0; t < infos.size();++t)
        {
            selected = &infos[t];
            sPars = paramScale_*selected->currentParams;

            for (unsigned int s = 0; s < infos.size();++s)
            {
                testing = &infos[s];
                if (!testing->isValid) continue;
                tPars = paramScale_*testing->currentParams;

                diff = sPars-tPars;

                distance = std::sqrt(diff.dot(diff));
                tweight = GetWeightTukey(distance,distThreshold_);

                selected->Add(testing,tweight);
            }

            if (selected->clusterValue > maxWeight)
            {
                maxWeight = selected->clusterValue;
                maxBlock = selected;
            }


        }

        // for debug
        for (unsigned int t = 0; t < maxBlock->selectedBlock.size();++t) maxBlock->selectedBlock[t]->isSelected = true;

        lastSelectedBlock_ = maxBlock;
        return maxBlock->CalcSumParams();



    }

    BlockInfo *lastSelectedBlock_;
    cv::Mat paramScale_;

    float distThreshold_;

};



/*!
 Block registration implementation. Requires a registration proc and a filter proc.
 */
template <typename IR_PROC, typename BLOCK_PROC>
class BlockReg : public ImageReg<IR_PROC>
{

public:

    typedef std::shared_ptr<BlockReg > ptr;

    static BlockReg::ptr Create(){ return std::make_shared< BlockReg >() ; }


    /// This is specific for block registration
    void SetBlockParams(cv::Size imgSize,  cv::Size blockSize, cv::Point2i blockStep)
    {
        blocks_.clear();

        blockProc_.paramScale_ = cv::Mat::eye(IR_PROC::numParams,IR_PROC::numParams,CV_32F);

        cv::Point2i curPos(0,0);

        for (curPos.y = 0; curPos.y+blockSize.height <= imgSize.height;curPos.y += blockStep.y)
        {
            for (curPos.x = 0; curPos.x+blockSize.width <= imgSize.width;curPos.x += blockStep.x)
            {
                BlockInfo nInfo(IR_PROC::numParams,curPos, cv::Point2i(blockSize.width,blockSize.height));
                blocks_.emplace_back(nInfo);
            }

        }

        blockProc_.distThreshold_ = 1.0;


    }

    /// Set parameter scales
    void SetParamScales(cv::Mat paramScale)
    {
        blockProc_.paramScale_ = paramScale;

    }

    /// Max distance for block filter
    void SetDistThreshold(float thresh)
    {
        blockProc_.distThreshold_ = thresh;
    }


    /// Perform the image align
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

        cv::Point2i offset(0,0);
        cv::Point2i tmpPos(0,0);
        cv::Point2i tmpSize(tmpImg.cols,tmpImg.rows);


        while (result.TestResults(termCrit_))
        {

            result.warpMat = IR_PROC::DoWarp(tmpImg_->mat_,tmpMask_->mat_,tmpGradX_->mat_,tmpGradY_->mat_,result.params,wTmpImg_->mat_,wTmpMask_->mat_,wTmpGradX_->mat_,wTmpGradY_->mat_);

            if (proc_.useESMJac)
            {
                for (unsigned int t = 0; t < blocks_.size();++t)
                {
                    BlockInfo *curInfo = &blocks_[t];
                    curInfo->SetZero();
                    CalcHesJacDifESMAVX(refImg_->mat_,refGradX_->mat_,refGradY_->mat_,refMask_->mat_,
                                        wTmpImg_->mat_,wTmpGradX_->mat_,wTmpGradY_->mat_,wTmpMask_->mat_,
                                        offset,curInfo->blockPos,curInfo->blockSize,curInfo->hess,curInfo->jacDiff,curInfo->numPixels);

                    curInfo->CalcParams();

                }
            }
            else
            {
                for (unsigned int t = 0; t < blocks_.size();++t)
                {
                    BlockInfo *curInfo = &blocks_[t];
                    curInfo->SetZero();

                    CalcHesJacDifICAVX(refImg_->mat_,refGradX_->mat_,refGradY_->mat_,refMask_->mat_,
                                       wTmpImg_->mat_,wTmpMask_->mat_,
                                       offset,curInfo->blockPos,curInfo->blockSize,curInfo->hess,curInfo->jacDiff,curInfo->numPixels);

                    curInfo->CalcParams();

                }

            }

            cv::Mat resPars = blockProc_.Filter(blocks_);


            cv::Mat deltaPars = IR_PROC::toDeltaPars(resPars,stepFactor_);

            result.Update(deltaPars);
            result.pixels = blockProc_.lastSelectedBlock_->numPixelSum;

            Utils_SIMD::CalcErrorSqrAVX(refImg_->mat_,refMask_->mat_,wTmpImg_->mat_,wTmpMask_->mat_,offset,result.error,result.pixels);

#ifndef NDEBUG
            DrawDebug();
#endif
            PrintResult(result);

        }

        PrintResult(result);



    }

    /// Show the debug image
    void DrawDebug()
    {
        cv::Mat visMat = VisualizeBlocks2(refImg_->mat_,proc_);



        cv::namedWindow("blocks",cv::WINDOW_AUTOSIZE);

        cv::imshow("blocks",visMat);

        cv::waitKey();


    }


    /// Draw points to image
    void drawPoints(cv::Mat &img, cv::Point2d tl, cv::Point2d tr, cv::Point2d bl, cv::Point2d br, cv::Scalar color)
    {
        cv::line(img,tl,tr,color);
        cv::line(img,tr,br,color);
        cv::line(img,br,bl,color);
        cv::line(img,bl,tl,color);

    }

    /// Draw the debug image
    cv::Mat VisualizeBlocks2(cv::Mat &img, IR_PROC &esmProc)
    {
        cv::Mat colorImg;

        cv::Mat target;

        img.convertTo(target,CV_8U);

        if (target.channels() == 1)
        {
        cv::cvtColor(target,colorImg,CV_GRAY2BGR);
        }
        else colorImg = target;

        cv::Point2d tl,tr,bl,br;
        cv::Point2d wtl,wtr,wbl,wbr;

        cv::Point2d otl,otr,obl,obr;


        for (unsigned int i = 0; i < blocks_.size();i++)
        {

            BlockInfo *cB = &blocks_[i];
            cv::Size bSize(cB->blockSize.x,cB->blockSize.y);
            if (!cB->isValid) continue;


            tl = cv::Point2d(cB->blockPos.x,cB->blockPos.y);
            tr = cv::Point2d(cB->blockPos.x+bSize.width,cB->blockPos.y);
            bl = cv::Point2d(cB->blockPos.x,cB->blockPos.y+bSize.height);
            br = cv::Point2d(cB->blockPos.x+bSize.width,cB->blockPos.y+bSize.height);

            otl = cv::Point2d(-bSize.width/2,-bSize.height/2);
            otr = cv::Point2d(bSize.width/2,-bSize.height/2);
            obl = cv::Point2d(-bSize.width/2,bSize.height/2);
            obr = cv::Point2d(bSize.width/2,bSize.height/2);


            std::vector<cv::Point2d> inputP;
            std::vector<cv::Point2d> outputP;
            inputP.push_back(otl);
            inputP.push_back(otr);
            inputP.push_back(obl);
            inputP.push_back(obr);
            outputP.push_back(otl);
            outputP.push_back(otr);
            outputP.push_back(obl);
            outputP.push_back(obr);

            bool containsInvalid = false;
            for (int r = 0; r < cB->currentParams.rows;r++)
            {
                if (std::isnan((cB->currentParams.at<float>(r,0))))  containsInvalid = true;
                if (std::isinf((cB->currentParams.at<float>(r,0))))  containsInvalid = true;

            }
            if (containsInvalid) continue;

            cv::Mat tParams;
            cB->currentParams.convertTo(tParams,CV_64F);

            cv::Mat warpMat = esmProc.CreateMat(tParams);
            cv::Mat tMat = warpMat.rowRange(0,2);
            cv::transform(inputP, outputP, tMat);

            wtl = outputP[0]+(tr+tl+br+bl)*0.25;
            wtr = outputP[1]+(tr+tl+br+bl)*0.25;
            wbl = outputP[2]+(tr+tl+br+bl)*0.25;
            wbr = outputP[3]+(tr+tl+br+bl)*0.25;


            drawPoints(colorImg,tl,tr,bl,br,cv::Scalar(255,0,0));
            drawPoints(colorImg,wtl,wtr,wbl,wbr,cv::Scalar(0,0,255));
            cv::line(colorImg,tl,wtl,cv::Scalar(0,255,0));
            cv::line(colorImg,tr,wtr,cv::Scalar(0,255,0));
            cv::line(colorImg,bl,wbl,cv::Scalar(0,255,0));
            cv::line(colorImg,br,wbr,cv::Scalar(0,255,0));

            if (cB->isSelected)
            {
                cv::circle(colorImg,(tr+tl+br+bl)*0.25,3,cv::Scalar(0,255,0));
            }


            std::ostringstream oss;
            oss << std::fixed << std::setprecision( 1 ) << cB->clusterValue;
            cv::putText(colorImg, oss.str(), cv::Point2i(cB->blockPos.x +2,cB->blockPos.y +10),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(50,200,250), 1, CV_AA);



        }
        return colorImg;


    }


private:
    using ImageReg<IR_PROC>::refImg_;
    using ImageReg<IR_PROC>::refMask_;
    using ImageReg<IR_PROC>::refGradX_;
    using ImageReg<IR_PROC>::refGradY_;
    using ImageReg<IR_PROC>::tmpImg_;
    using ImageReg<IR_PROC>::tmpMask_;
    using ImageReg<IR_PROC>::tmpGradX_;
    using ImageReg<IR_PROC>::tmpGradY_;

    using ImageReg<IR_PROC>::wTmpImg_;
    using ImageReg<IR_PROC>::wTmpMask_;
    using ImageReg<IR_PROC>::wTmpGradX_;
    using ImageReg<IR_PROC>::wTmpGradY_;

    using ImageReg<IR_PROC>::proc_;
    using ImageReg<IR_PROC>::termCrit_;

    using ImageReg<IR_PROC>::stepFactor_;
    using ImageReg<IR_PROC>::PrintResult;

    using ImageReg<IR_PROC>::CalcHesJacDifESMAVX;
    using ImageReg<IR_PROC>::CalcHesJacDifICAVX;



    BS_EuclidDist blockProc_;
    //    BLOCK_PROC blockProc_;
    std::vector<BlockInfo> blocks_;
};


#endif // BLOCKREG_H
