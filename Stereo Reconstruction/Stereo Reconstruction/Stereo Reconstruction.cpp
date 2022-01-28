// Stereo Reconstruction.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
//#include <opencv2/viz.hpp>
#include <fstream>
#include <string>
#include <cstring>  



using namespace cv;
using namespace std;

int main()
{

    // Mat image_array[2];
    // image_array[0] = cv::imread("../../data/pendulum1/im0.png");
    // image_array[1] = cv::imread("../../data/pendulum1/im1.png");
    
    // if (image_array[0].empty() || image_array[1].empty())
    // {
    //     std::cout << "Could not read the image: " << "../../data/pendulum1/im0.png" << std::endl;
    //     return 1;
    // }

    // for (size_t i = 0; i < 2; i++)
    // {
    //     cv::namedWindow("image", WINDOW_NORMAL);
    //     imshow("image", image_array[i]);
    //     int k = waitKey(0); // Wait for a keystroke in the window
    //     if (k == 's')
    //     {
    //         imwrite(i + ".png", image_array[i]);
    //     }
        
    // }

    // return 0;

    std::cout << "Hello World!\n";


    // Reading the intrinsics 

    fstream newfile;
    std::string cam0;
    std::string cam1;
    newfile.open("../../data/artroom1/calib.txt", ios::in); //open a file to perform read operation using file object
    if (newfile.is_open()) { //checking whether the file is open
        for (int i = 0; i < 2; i++) {
            if (i == 0) {
                getline(newfile, cam0);
            }
            else {
                getline(newfile, cam1);
            }

        }
        newfile.close(); //close the file object.
    }

    cout << cam0 << endl;
    cout << cam1 << endl;

    string cam0_values;
    string cam1_values;
    bool enter = false;
    
    // reading the cam0 values
    for(int i=0;i<cam0.length();i++){ //taking values inside the square brackets to a string
    if(enter){
        if(cam0.at(i) ==']'){
            enter=false;
        }
        else{
            cam0_values += cam0.at(i);
        }
    }
    if(cam0.at(i) =='[')	{
            enter = true;
        }
    }
    
    cout << cam0_values << endl;

    std::string segment0;
    std::vector<std::string> cam0list;
    float cam0data[3][3];

    std::istringstream split0(cam0_values);

    while(std::getline(split0, segment0, ';')){ // spliting the string with semi-colon
        // cout << segment0 << endl;
        cam0list.push_back(segment0); //Spit string at ';' character
    }

    cout << cam0list[0] << endl;


    std::istringstream split01(cam0list[1]);


    for (size_t i = 0; i < cam0list.size(); i++) { // filling the matrix values
        std::istringstream split0str(cam0list[i]);
        int j = 0;
        while (std::getline(split0str, segment0, ' ')) { // spliting the string with semi-colon
            if (!segment0.empty()) {
                cout << segment0 << endl;
               
                cam0data[i][j] = std::atof(segment0.c_str());
                cout << cam0data[i][j] << endl;
                j++;
            } //Spit string at ';' character
        }

    
        
    }
    
   
   cv::Mat calibMat = cv::Mat(3, 3, CV_32F, cam0data); // calibration Matrix 
   cout << calibMat << endl;

    cv::Mat img1 = cv::imread("../../data/artroom1/im0.png", IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("../../data/artroom1/im1.png", IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty())
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        //parser.printMessage();
        return -1;
    }

    // 

    




    //-- Step 1: Detect the keypoints using ORB Detector, compute the descriptors
    int minHessian = 400;
    Ptr<FeatureDetector> detector = ORB::create();
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    std::cout << "key points 1 size:  " << keypoints1.size() << std::endl;
    std::cout << "key points 2 size:  " << keypoints2.size() << std::endl;

    cv::Mat output;
    cv::drawKeypoints(img1, keypoints1, output);
    descriptors1.convertTo(descriptors1, CV_32F);
    descriptors2.convertTo(descriptors2, CV_32F);

  /*  FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);*/

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);


    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    std::vector<Point2f> points1;
    std::vector<Point2f> points2;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
            points1.push_back(keypoints1[knn_matches[i][0].queryIdx].pt);
            points2.push_back(keypoints2[knn_matches[i][0].trainIdx].pt);
        }
    }


    ////-- Draw matches
    //Mat img_matches;
    //cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
    //    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    ////-- Show detected matches
    //cv::namedWindow("Good Matches", WINDOW_NORMAL);
    //cv::imshow("Good Matches", img_matches);

    //


    //int k = waitKey(0); // Wait for a keystroke in the window
    //if (k == 's')
    //{
    //    imwrite("crosspond.png", img_matches);
    //}

    // Find the essential between image 1 and image 2
    cv::Mat inliers;
    cv::Mat essential = cv::findEssentialMat(points1, points2, calibMat, cv::RANSAC, 0.9, 1.0, inliers);
    cout << essential << endl;

    // recover relative camera pose from essential matrix
    cv::Mat rotation, translation;
    cv::recoverPose(essential, points1, points2, calibMat, rotation, translation, inliers);
    cout << rotation << endl;
    cout << translation << endl;


    // Compute disparity
    cv::Mat disparity;
    cv::Ptr<cv::StereoMatcher> pStereo = cv::StereoSGBM::create(0, 32, 5);
    pStereo->compute(img1, img2, disparity);
    cv::imwrite("disparity.jpg", disparity);

    cv::Mat R1;
    cv::Mat R2;
    cv::Mat P1;
    cv::Mat P2;
    cv::Mat Q;
    cv::Mat1d dis = (Mat1d(1,5) << 0, 0,0,0,0);

    cv::stereoRectify(calibMat, dis, calibMat, dis, Size(1920, 1080), rotation, translation, R1, R2, P1, P2, Q);
    cv::Mat pointClouds;

    reprojectImageTo3D(disparity, pointClouds, Q);

    cv::Mat color_img;

    cv::cvtColor(img1, color_img, cv::COLOR_BGR2RGB);

    Vec3b color = color_img.at<Vec3b>(Point(500, 500));
    cout << "Abdooooo" << pointClouds.at<float>(0, 0) << endl;
    cout << "Abdooooo" << pointClouds.size().height << endl;
    cout << "color" <<  (int)color_img.at<cv::Vec3b>(500, 500)[0] << endl;
 

    int count = pointClouds.size().height * pointClouds.size().width;

    ofstream outfile("pointcloud.ply");
    outfile << "ply\n" << "format ascii 1.0\n";
    outfile << "element vertex '\'" << count << "\n";
    outfile << "property float x\n" << "property float y\n" << "property float z\n" ;
    outfile << "property uint8 red\n" << "property uint8 green\n" << "property uint8 blue\n"  ;
    outfile << "end_header\n";


    for (size_t i = 0; i < pointClouds.size().height; i++)
    {
        for (size_t j = 0; j < pointClouds.size().width; j++)
        {
            outfile << i << " ";
            outfile << j << " ";
            outfile << pointClouds.at<float>(i, j) << " ";
            outfile << (int)color_img.at<cv::Vec3b>(i, j)[0] << " ";
            outfile << (int)color_img.at<cv::Vec3b>(i, j)[1] << " ";
            outfile << (int)color_img.at<cv::Vec3b>(i, j)[2] << " ";
            outfile << "\n";
        }
    }
    outfile.close();



    
    //cout << inliers << endl;

    // ------------------------------------------- Triangulation -----------------------------------------------------


    //// compose projection matrix from R,T
    //cv::Mat projection2(3, 4, CV_32F); // the 3x4 projection matrix
    //rotation.copyTo(projection2(cv::Rect(0, 0, 3, 3)));
    //translation.copyTo(projection2.colRange(3, 4));
    //// compose generic projection matrix
    //cv::Mat projection1(3, 4, CV_32F, 0.); // the 3x4 projection matrix
    //cv::Mat diag(cv::Mat::eye(3, 3, CV_32F));
    //diag.copyTo(projection1(cv::Rect(0, 0, 3, 3)));
    //// to contain the inliers
    //std::vector<cv::Vec2d> inlierPts1;
    //std::vector<cv::Vec2d> inlierPts2;
    //// create inliers input point vector for triangulation
    //int j(0);
    //for (int i = 0; i < inliers.rows; i++) {
    //    if (inliers.at<uchar>(i)) {
    //        inlierPts1.push_back(cv::Vec2d(points1[i].x, points1[i].y));
    //        inlierPts2.push_back(cv::Vec2d(points2[i].x, points2[i].y));
    //    }
    //}

    //// // we multipied the calib martix with the projection matrices to compansate for the distortion in the image points 

    //projection1 = calibMat * projection1;
    //projection2 = calibMat * projection2;

    //cout << inlierPts1[0](0) << endl;

    //// doing the trangiulation 

    //std::vector<cv::Vec3f> points3D;

    //for (size_t i = 0; i < sizeof(inlierPts1); i++)
    //{
    //    cv::Vec2d u1 = inlierPts1[i];
    //    cv::Vec2d u2 = inlierPts2[i];
    //    

    //    cv::Matx44f A(  
    //        u1(1) * projection1.at<float>(2, 0) - projection1.at<float>(1, 0),
    //        u1(1) * projection1.at<float>(2, 1) - projection1.at<float>(1, 1),
    //        u1(1) * projection1.at<float>(2, 2) - projection1.at<float>(1, 2),
    //        u1(1) * projection1.at<float>(2, 3) - projection1.at<float>(1, 3),

    //        projection1.at<float>(0, 0) - u1(0) * projection1.at<float>(2, 0),
    //        projection1.at<float>(0, 1) - u1(0) * projection1.at<float>(2, 1),
    //        projection1.at<float>(0, 2) - u1(0) * projection1.at<float>(2, 2),
    //        projection1.at<float>(0, 3) - u1(0) * projection1.at<float>(2, 3),

    //        u2(1)* projection2.at<float>(2, 0) - projection2.at<float>(1, 0),
    //        u2(1)* projection2.at<float>(2, 1) - projection2.at<float>(1, 1),
    //        u2(1)* projection2.at<float>(2, 2) - projection2.at<float>(1, 2),
    //        u2(1)* projection2.at<float>(2, 3) - projection2.at<float>(1, 3),

    //        projection2.at<float>(0, 0) - u2(0) * projection2.at<float>(2, 0),
    //        projection2.at<float>(0, 1) - u2(0) * projection2.at<float>(2, 1),
    //        projection2.at<float>(0, 2) - u2(0) * projection2.at<float>(2, 2),
    //        projection2.at<float>(0, 3) - u2(0) * projection2.at<float>(2, 3)

    //                    
    //                   
    //        
    //        );

    //    cv::Matx41f B;
    //    cv::Vec4f X;
    //    /*cout << A << endl;
    //    cout << B << endl;*/


    //    cv::SVD::solveZ(A, X);
    //    
    //    X = X * (1 / X[3]);
    //    cv::Vec3f X_3 (X[0], X[1], X[2]);

    //    points3D.push_back(X_3);
    //    
    //    
    //}

    //
    //cout << "3D points :" << points3D.size() << endl;

    //for (size_t i = 0; i < points3D.size(); i++)
    //{
    //    cout << points3D[i] << endl;

    //}
    ////cv::viz::Viz3d window; //creating a Viz window
    //////Displaying the Coordinate Origin (0,0,0)
    ////window.showWidget("coordinate", viz::WCoordinateSystem());
    ////window.setBackgroundColor(cv::viz::Color::black());
    //////Displaying the 3D points in green
    ////window.showWidget("points", viz::WCloud(points3D, viz::Color::green()));
    ////window.spin();

    return 0;

    //std::string image_path = samples::findFile("starry_night.jpg");
    //Mat img = imread(image_path, IMREAD_COLOR);
    //if (img.empty())
    //{
    //    std::cout << "Could not read the image: " << image_path << std::endl;
    //    return 1;
    //}
    //imshow("Display window", img);
    //int k = waitKey(0); // Wait for a keystroke in the window
    //if (k == 's')
    //{
    //    imwrite("starry_night.png", img);
    //}
    //return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
