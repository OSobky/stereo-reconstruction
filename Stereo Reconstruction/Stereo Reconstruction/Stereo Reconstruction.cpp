// Stereo Reconstruction.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;

int main()
{

    Mat image_array[2];
    image_array[0] = cv::imread("../../data/pendulum1/im0.png");
    image_array[1] = cv::imread("../../data/pendulum1/im1.png");
    
    if (image_array[0].empty() || image_array[1].empty())
    {
        std::cout << "Could not read the image: " << "../../data/pendulum1/im0.png" << std::endl;
        return 1;
    }

    for (size_t i = 0; i < 2; i++)
    {
        cv::namedWindow("image", WINDOW_NORMAL);
        imshow("image", image_array[i]);
        int k = waitKey(0); // Wait for a keystroke in the window
        if (k == 's')
        {
            imwrite(i + ".png", image_array[i]);
        }
        
    }

    return 0;

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
