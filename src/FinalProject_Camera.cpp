
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;


int evaluate3DobjectTracking(string selectedDetectorType, string selectedDescriptorType, MatchingParameters& matchingParameters, ofstream& fileOut, bool bVis=false)
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector

    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;

    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;

    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;

    // misc
    double sensorFrameRate = 10.0 / matchingParameters.imgStepWidth; // frames per second for Lidar and camera
    const int dataBufferSize{2};      // no. of images which are held in memory (ring buffer) at the same time
    array<DataFrame, dataBufferSize> dataBuffer; // list of data frames which are held in memory at the same time
    uint8_t circularIdx{0};
    std::size_t idxTrack{0};

    uint numberOfKeypointsOnVehicle{0};
    vector<uint> numOfKeypointsPerImage;
    uint avgKeypointNeighborhoodSize{0};
    vector<uint> nunOfKeypointNeighborhoodSizePerImage;
    vector<uint> numOfKeypointMatchesPerImage;
    uint numOfKeypointMatches;
    double keypointDetectionTime;
    double keypointDiscriptorTime;
    vector<double> keypointDetectionTimePerImg;
    vector<double> keypointDiscriptorTimePerImg;

    bool kptsDetvisualizationEnable{false};
    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= matchingParameters.imgEndIndex - matchingParameters.imgStartIndex; imgIndex+=matchingParameters.imgStepWidth)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(matchingParameters.imgFillWidth) << matchingParameters.imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file
        cv::Mat img = cv::imread(imgFullFilename);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;

        circularIdx = circularIdx % dataBufferSize;
        if (idxTrack > 1 and circularIdx >= 0)
        {
            swap(dataBuffer[0], dataBuffer[1]);
            circularIdx += 1;
            dataBuffer[circularIdx] = frame;
        }
        else if (circularIdx == 0 and idxTrack == 0)
            dataBuffer[circularIdx+1] = frame;
        else if (circularIdx == 1 and idxTrack == 1)
        {
            swap(dataBuffer[0], dataBuffer[1]);
            dataBuffer[circularIdx] = frame;
        }

        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold = 0.4;
        detectObjects((dataBuffer.end()-1)->cameraImg, (dataBuffer.end()-1)->boundingBoxes, confThreshold, nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

        cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;

        /* CROP LIDAR POINTS */

        // load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);

        (dataBuffer.end()-1)->lidarPoints = lidarPoints;

        cout << "#3 : CROP LIDAR POINTS done" << endl;


        /* CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI
        float shrinkFactor = 0.20; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end()-1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

        // Visualize 3D objects
        if(bVis)
        {
            show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(1000, 2000), true);
        }
        cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        if (selectedDetectorType == "SHITOMASI")
        {
            detKeypointsShiTomasi(keypoints, imgGray, keypointDetectionTime, kptsDetvisualizationEnable);
        }
        else if (selectedDetectorType == "HARRIS")
        {
            detKeypointsHarris(keypoints, imgGray, keypointDetectionTime, kptsDetvisualizationEnable);
        }
        else
        {
            detKeypointsModern(keypoints, imgGray, selectedDetectorType, keypointDetectionTime, kptsDetvisualizationEnable);
        }
        keypointDetectionTimePerImg.push_back(keypointDetectionTime);

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            const int maxKeypoints = 50;
            keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end()-1)->keypoints = keypoints;

        cout << "#5 : DETECT KEYPOINTS done" << endl;


        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
        descKeypoints((dataBuffer.end()-1)->keypoints, (dataBuffer.end()-1)->cameraImg, descriptors, selectedDescriptorType, keypointDiscriptorTime);

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end()-1)->descriptors = descriptors;

        cout << "#6 : EXTRACT DESCRIPTORS done" << endl;
        circularIdx++;
        idxTrack++;


        if (!dataBuffer[0].cameraImg.empty() && !dataBuffer[1].cameraImg.empty())
        {

            /* MATCH KEYPOINT DESCRIPTORS */
            // Default parameters are already loaded in struct MatchingParameters
            vector<cv::DMatch> matches;
            string matcherType = matchingParameters.matcherType;        // MATCH_BF, MATCH_FLANN
            string descriptorType = matchingParameters.descriptorType; // DESCRIPTOR_BINARY, DESCRIPTOR_HOG for distance computation selection
            string selectorType = matchingParameters.selectorType;       // SELECT_NN, SELECT_KNN

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;


            /* TRACK 3D OBJECT BOUNDING BOXES */

            //// STUDENT ASSIGNMENT
            //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)

            map<int, int> bbBestMatches;
            matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end() - 2), *(dataBuffer.end() - 1)); // associate bounding boxes between current and previous frame using keypoint matches
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end()-1)->bbMatches = bbBestMatches;

            cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;

            /* COMPUTE TTC ON OBJECT IN FRONT */

            // loop over all BB match pairs
            for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
            {
                // find bounding boxes associates with current match
                BoundingBox *prevBB, *currBB;
                for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                {
                    if (it1->second == it2->boxID) // check whether current match partner corresponds to this BB
                    {
                        currBB = &(*it2);
                    }
                }

                for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                {
                    if (it1->first == it2->boxID) // check whether current match partner corresponds to this BB
                    {
                        prevBB = &(*it2);
                    }
                }

                // compute TTC for current match
                if( currBB->lidarPoints.size() > 0 && prevBB->lidarPoints.size() > 0 ) // only compute TTC if we have Lidar points
                {
                    //// STUDENT ASSIGNMENT
                    //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                    double ttcLidar;
                    computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
//                    fileOut << "TTC lidar : " << ttcLidar <<endl;
                    //// EOF STUDENT ASSIGNMENT

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                    //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                    double ttcCamera;
                    clusterKptMatchesWithROI(*currBB, *prevBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);
                    computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);
//                    fileOut << "TTC Camera : " << ttcCamera <<endl;
                    fileOut << ttcLidar << "," << ttcCamera <<endl;
                    //// EOF STUDENT ASSIGNMENT

                    if (bVis)
                    {
                        cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);

                        char str[200];
                        sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                        string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
                        cv::imshow(windowName, visImg);
                        cout << "Press key to continue to next frame" << endl;
                        cv::waitKey(0);
                    }

                } // eof TTC computation
            } // eof loop over all BB matches

        }

    } // eof loop over all images
    return 0;
}

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    MatchingParameters descMatchingParameters;
    vector<string> keypointDetectorTypes{"SHITOMASI", "HARRIS", "SIFT", "FAST", "ORB", "BRISK"};
    vector<string> keypointDescriptorTypes{"SIFT", "BRIEF", "BRISK", "FREAK", "ORB"};
    uint retval;
    string outputLogFile{"../src/evaluation_3d_tracking_sf_0_2_kptsthr_2_0.csv"};
    ofstream out(outputLogFile, ios::out);
    bool visualizationFlag{true};
    for (auto& detectorType: keypointDetectorTypes)
    {
        for (auto& descriptorType: keypointDescriptorTypes)
        {
            out<< "DetectorType = "<< detectorType << ", "<< "DescriptorType = "<< descriptorType<<endl;
            out<< "TTC_Lidar" << ", "<< "TTC_Camera"<<endl;
            cout<< "DetectorType = "<< detectorType << ", "<< "DescriptorType = "<< descriptorType<<endl;

            if(detectorType == "SIFT" and descriptorType == "ORB")
                continue; //SIFT det. and ORB Desc.
            if (descriptorType == "BRISK")
            {
                descMatchingParameters.matcherType = "MATCH_BF";
                descMatchingParameters.descriptorType = "DESCRIPTOR_BINARY";
                descMatchingParameters.selectorType = "SELECT_KNN";
                retval = evaluate3DobjectTracking(detectorType, descriptorType, descMatchingParameters, out, visualizationFlag);
                if(retval != 0)
                {
                    cerr<< "Evaluation failed!!!" <<endl;
                    exit(1);
                }
            }
            if(descriptorType != "BRISK")
            {
                descMatchingParameters.matcherType = "MATCH_BF";
                descMatchingParameters.descriptorType = "DESCRIPTOR_HOG";
                descMatchingParameters.selectorType = "SELECT_KNN";
                retval = evaluate3DobjectTracking(detectorType, descriptorType, descMatchingParameters, out, visualizationFlag);
                if(retval != 0)
                {
                    cerr<< "Evaluation failed!!!" <<endl;
                    exit(1);
                }
            }
        }
    }
    out<< "DetectorType = "<< "AKAZE"<< ", "<< "DescriptorType = "<< "AKAZE"<<endl;
    out<< "TTC_Lidar" << ","<< "TTC_Camera"<<endl;
    cout<< "DetectorType = "<< "AKAZE"<< ", "<< "DescriptorType = "<< "AKAZE"<<endl;
    retval = evaluate3DobjectTracking("AKAZE", "AKAZE", descMatchingParameters, out, visualizationFlag);
    if(retval != 0)
    {
        cerr<< "Evaluation failed!!!" <<endl;
        exit(1);
    }
    out.close();
    return 0;

    return 0;
}
