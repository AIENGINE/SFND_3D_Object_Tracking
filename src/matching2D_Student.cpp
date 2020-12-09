#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;


    if (matcherType.compare("MATCH_BF") == 0)
    {
        int normType = descriptorType.compare("DESCRIPTOR_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType == "MATCH_FLANN")
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::FlannBasedMatcher::create();
        cout << "FLANN matching"<<endl;
    }

    // perform matching task
    if (selectorType == "SELECT_NN")
    { // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType == "SELECT_KNN")
    { // k nearest neighbors (k=2)

        vector<vector<cv::DMatch>> knnMatches;
        const float ratioThreshold = 0.80;

        matcher->knnMatch(descSource, descRef, knnMatches, 2);

        for (auto& knnMatch: knnMatches)
        {

            if ((knnMatch[0].distance / knnMatch[1].distance) < ratioThreshold)
                matches.push_back(knnMatch[0]);

        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, double& totalTime)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    cv::Ptr<cv::AKAZE> extractorAKAZE;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
        //create descriptors with default parameters
    else if (descriptorType == "SIFT")
    {
        extractor = cv::SIFT::create();
    }
    else if (descriptorType == "BRIEF")
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType == "ORB")
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType == "FREAK")
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType == "AKAZE")
    {
        extractorAKAZE = cv::AKAZE::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    if(descriptorType == "AKAZE")
        extractorAKAZE->compute(img, keypoints, descriptors);
    else
        extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    totalTime = 1000 * t / 1.0;
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double& totalTime, bool bVis)
{
    // compute detector parameters based on image size
    const int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    const double maxOverlap = 0.0; // max. permissible overlap between two features in %
    const double minDistance = (1.0 - maxOverlap) * blockSize;
    const int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    const double qualityLevel = 0.01; // minimal accepted quality of image corners
    const double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto & corner : corners)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f(corner.x, corner.y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    totalTime = 1000 * t / 1.0;
    cout << "Shi-Tomasi detection with n =" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double& totalTime, bool bVis)
{

    // Detector parameters
    const int blockSize = 5;     // for every pixel, a blockSize × blockSize neighborhood is considered for corner
    const int apertureSize = 5;  // aperture parameter for Sobel operator (must be odd)
    const int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    const double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    double t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
//    cv::convertScaleAbs(dst_norm, dst_norm_scaled); //consider only abs values rounding off to integers

    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;// region to checked overlapping and visualization as well
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto & keypoint : keypoints)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, keypoint);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > keypoint.response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            keypoint = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    totalTime = 1000 * t / 1.0;
    cout << "HARRIS Corner detection with n =" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "HARRIS Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, double& totalTime, bool bVis)
{

    cv::Ptr<cv::FeatureDetector> calcFeature;
    double t;
    if (detectorType.compare("BRISK") == 0)
    {
        calcFeature = cv::BRISK::create();
        t = (double)cv::getTickCount();
        calcFeature->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        totalTime = 1000 * t / 1.0;
        cout << "BRISK detector with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    }
    else if (detectorType.compare("SIFT") == 0)
    {
        calcFeature = cv::SIFT::create();
        t = (double)cv::getTickCount();
        calcFeature->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        totalTime = 1000 * t / 1.0;
        cout << "SIFT detector with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (detectorType.compare("FAST") == 0)
    {
        const int threshold = 40; //intensity difference from center point to points on the ring
        const bool bNMS = true; //enable/disable Non-Maximum Suppression
        cv::FastFeatureDetector::DetectorType detType = cv::FastFeatureDetector::TYPE_9_16;
        calcFeature = cv::FastFeatureDetector::create(threshold, bNMS, detType);
        t = (double)cv::getTickCount();
        calcFeature->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        totalTime = 1000 * t / 1.0;
        cout << "FAST detector with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        calcFeature = cv::AKAZE::create();
        t = (double)cv::getTickCount();
        calcFeature->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        totalTime = 1000 * t / 1.0;
        cout << "AKAZE detector with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (detectorType.compare("ORB") == 0)
    {
        calcFeature = cv::ORB::create();
        t = (double)cv::getTickCount();
        calcFeature->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        totalTime = 1000 * t / 1.0;
        cout << "ORB detector with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Feature Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }


}