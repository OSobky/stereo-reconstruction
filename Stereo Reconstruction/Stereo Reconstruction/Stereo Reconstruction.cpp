// Stereo Reconstruction.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/core.hpp"



using namespace cv;

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
    cv::Mat img1 = cv::imread("../img/data/artroom1/im0C.png", IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("../img/data/artroom1/im1C.png", IMREAD_GRAYSCALE);
  /*  if (img1.empty() || img2.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        parser.printMessage();
        return -1;
    }*/
    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    Ptr<FeatureDetector> detector = ORB::create();
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

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
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    //-- Draw matches
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
        Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //-- Show detected matches
    imshow("Good Matches", img_matches);

    


    int k = waitKey(0); // Wait for a keystroke in the window
   
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
