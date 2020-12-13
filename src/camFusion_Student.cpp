
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    cv::flip(topviewImg, topviewImg, 0); 
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg.t());

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


template<typename T>
double computeMean(const std::vector<T>& numbers)
{
    if (numbers.empty())
        return std::numeric_limits<double>::quiet_NaN();

    return std::accumulate(numbers.begin(), numbers.end(), 0.0) / numbers.size();
}

template<typename T, typename U>
double computeVariance(const T mean, const std::vector<U>& numbers)
{
    if (numbers.size() <= 1u)
        return std::numeric_limits<T>::quiet_NaN();

    auto const add_square = [mean](T sum, U i) {
        auto d = i - mean;
        return sum + d*d;
    };
    double total = std::accumulate(numbers.begin(), numbers.end(), 0.0, add_square);
    return total / (numbers.size() - 1);
}


void lidarPointInXDirection(vector<LidarPoint>& lidarPoints, vector<double>& lidarPointsInGivenDirection)
{
    auto filterDirectionPoints = [&](const LidarPoint points){ lidarPointsInGivenDirection.push_back(points.x); };
    for_each(lidarPoints.begin(), lidarPoints.end(), filterDirectionPoints);
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double laneWidth = 4.0; // assumed width of the ego lane

    vector<double> lidarPointsInDirectionX;
    lidarPointInXDirection(lidarPointsPrev, lidarPointsInDirectionX);
    auto meanLidarPointsPrev = computeMean<double>(lidarPointsInDirectionX);
    auto varianceLidarPointPrev = computeVariance(meanLidarPointsPrev, lidarPointsInDirectionX);
    auto stdDevPrev = sqrt(varianceLidarPointPrev);
    auto thresholdPrev = stdDevPrev * 2;
    auto lowerLimitPrev = meanLidarPointsPrev - thresholdPrev;
    auto upperLimitPrev = meanLidarPointsPrev + thresholdPrev;
//    cout << "LidarPointsPrev Mean  : " <<meanLidarPointsPrev << " stdDevPrev : " << stdDevPrev<<endl;

    vector<double> lidarPointsInDirectionY;
    lidarPointInXDirection(lidarPointsCurr, lidarPointsInDirectionY);
    auto meanLidarPointsCurr = computeMean<double>(lidarPointsInDirectionY);
    auto varianceLidarPointCurr = computeVariance(meanLidarPointsCurr, lidarPointsInDirectionY);
    auto stdDevCurr = sqrt(varianceLidarPointCurr);
    auto thresholdCurr = stdDevCurr * 2;
    auto lowerLimitCurr = meanLidarPointsCurr - thresholdCurr;
    auto upperLimitCurr = meanLidarPointsCurr + thresholdCurr;
//    cout << "LidarPointsCurr Mean  : " <<meanLidarPointsCurr << " stdDevPrev : " << stdDevCurr<<endl;


    vector<float> distPointsPrev;
    vector<float> distPointsCurr;
    // find closest distance to Lidar points within ego lane
//    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        if (abs(it->y) <= laneWidth / 2 and it->x > lowerLimitPrev and it->x < upperLimitPrev)
        {
            distPointsPrev.push_back(it->x);
            //minXPrev = minXPrev > it->x ? it->x : minXPrev;
            // minXPrev = it->x;

        }
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        if (abs(it->y) <= laneWidth / 2 and it->x > lowerLimitCurr and it->x < upperLimitCurr)
        {
            distPointsCurr.push_back(it->x);
            //minXCurr = minXCurr > it->x ? it->x : minXCurr;
            // minXCurr = it->x;
        }
    }
    auto minXPrev = min_element(distPointsPrev.begin(), distPointsPrev.end());
    auto minXCurr = min_element(distPointsCurr.begin(), distPointsCurr.end());
    cout << "size of lidarPointsPrev : " << lidarPointsPrev.size() << " size of distPointsPrev : " << distPointsPrev.size() << endl;
    cout << "size of lidarPointsCurr : " << lidarPointsCurr.size() << " size of distPointsCurr : " << distPointsCurr.size() << endl;
    // compute TTC from both measurements
    double dT = 1 / frameRate;
    TTC = *minXCurr * dT / (*minXPrev - *minXCurr);

}

/* The idea here is keep track of the boundingboxes from the current and previous frames through matched keypoints
 * that are contained in the ROI of two frames. To track bounding box correctly a scoring mechanism is implemented
 * where when both ROI under match keypoints loop has keypoints then their externally tracked integer id is incremented.
 * The routine where ids are correctly tracked based on the number of keypoints in both ROIs i.e, for the corresponding
 * previous frame id check current frame id that has maximum number meaning maximum number of corresponding keypoints
 * enclosed, which has already been updated in the scoring routine should update bbBestMatches map.
 * TODO {Algorithm for clusterKptMatchesWithROI} Check: 1. If we can cluster the keypoints
 * here then in function clusterKptMatchesWithROI create shrink bounding boxes and add only matching keypoints to the
 * vector of kptmatches. That vector is used to create a loop in which Euclidean Norm is computer for the matched keypoints.
 * After that mean is calculated to create a threshold that is used again to calculate norm on matched keypoints and
 * if threshold is met matched keypoints are added on KptMatches.
*/

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    auto prevFrameBboxSize = prevFrame.boundingBoxes.size();
    auto currFrameBboxSize = currFrame.boundingBoxes.size();
//    int countKeypointsInROIs[prevFrameBboxSize][currFrameBboxSize] = {};
    vector<vector<int>> countKeypointsInROIs(prevFrameBboxSize, vector<int> (currFrameBboxSize, 0));
//    map<int, int> bbMatchExact; //test to scored bboxid is equal to acutal bboxid from yolo routine
    for (auto& matchingKeypoints: matches)
    {
        cv::KeyPoint queryKeyPoint = prevFrame.keypoints[matchingKeypoints.queryIdx];
        cv::Point queryPoint = cv::Point (queryKeyPoint.pt.x, queryKeyPoint.pt.y);
        bool queryPointFound = false;
        vector<int> queryBboxIdx;
        cv::KeyPoint trainKeyPoint = currFrame.keypoints[matchingKeypoints.trainIdx];
        cv::Point trainPoint = cv::Point (trainKeyPoint.pt.x, trainKeyPoint.pt.y);
        bool trainPointFound = false;
        vector<int> trainBboxIdx;

        for (std::size_t bbxidx = 0; bbxidx < prevFrameBboxSize; bbxidx++)
        {
            if (prevFrame.boundingBoxes[bbxidx].roi.contains(queryPoint))
            {
                queryPointFound = true;
                queryBboxIdx.push_back(bbxidx);
            }
        }

        for (std::size_t bbxidx = 0; bbxidx < currFrameBboxSize; bbxidx++)
        {
            if (currFrame.boundingBoxes[bbxidx].roi.contains(trainPoint))
            {
                trainPointFound = true;
                trainBboxIdx.push_back(bbxidx);
            }
        }

        if (queryPointFound && trainPointFound)
        {
            for (auto queryId: queryBboxIdx)
            {
//                prevFrame.boundingBoxes[queryId].keypoints.push_back(queryKeyPoint);
                for (auto trainId: trainBboxIdx)
                {

                    countKeypointsInROIs[queryId][trainId] += 1; // prevFrame bbox enclosed keypoints in relation to current frame bbox enclosed keypoints
//                    currFrame.boundingBoxes[trainId].keypoints.push_back(trainKeyPoint);
                }
            }
        }

    }
    // loop over external indexing for bounding box

    for (int i = 0; i < prevFrameBboxSize; i++)
    {
        int maxKeypointCount = 0;
        int maxCurrFrameId = 0;
//        int exactBoxId {0};
        for (int j = 0; j < currFrameBboxSize; j++)
        {
            if (countKeypointsInROIs[i][j] > maxKeypointCount)
            {
                maxKeypointCount = countKeypointsInROIs[i][j];
                maxCurrFrameId = j;
//                exactBoxId = currFrame.boundingBoxes[j].boxID;
            }
        }
        bbBestMatches[i] = maxCurrFrameId;
//        bbMatchExact[prevFrame.boundingBoxes[i].boxID] = exactBoxId;

    }

}
