# 3D Stereo Reconstruction

### Team Memebers:
1- Abdelrahman Amr Abdelaziz Mohamed Salem 
2- Omar Elsobky
3- Momen Amgad 
4- Loai Alaa

### Project Description:
3D Reconstruction is one of the main research problems in computer vision which has different applications such as autonomous driving and robot navigation. In this work, we attempt to reconstruct 3D scenes from stereo images captured from different views of the same scene. We Implemented two approaches for Dense and Sparse reconstructions of 3D Scenes using only two regular Images. We utilized Disparity Map for the dense approach and Linear Triangulation for the sparse approach. We experimented with a stereo data-set and reported very good results for the final constructed 3D scenes.

### Libs being used 
- cv2
- numpy
- glob
- json
- tqdm
- PIL.ExifTags
- PIL.Image
- matplotlib
- ast
- open3d

### Table of Contents

    .
    ├── stereo_reconstruction.py     # containing the whole pipline of project including the sparse and dense implementation
    ├── matching_algorithms.py       # containing the the different matching algorithms
    ├── data                         # containing input images and camera calibrations
    │   ├── calib.txt                # camera calibrations
    │   ├── im0.png                  # first view of an image
    │   └── im1.png                  # second view of an image 
    ├── Figures                      # figures
    │   ├── Dense.gif                # Dense output figure
    │   ├── disparity.png            # Disparity map output figure
    │   ├── ORB-FLANN.png            # ORB-FLANN output figure
    │   ├── SIFT-FLANN.png           # SIFT-FLANN output figure
    │   ├── Sparse.gif               # Sparse output figure
    │   └── SURF-FLANN.png           # SURF-FLANN output figure
    └── README.md


### How to Install and Run the Project

First step:
change the paths inside the stereo_reconstruction.py for the input files which are under the data folder. After that run the stereo_reconstruction.py which will create two ply files which are the "reconstructedTriang.ply" and "reconstructedDisparity.ply". During the runing of the files there will be figures appear which will be as follows: firstly the matching between the two correspondances, secondly the disparity output figure after that the dense output figure will appear and finally the sparse output figure will be shown.




### How to Experiment with different matching algorithms  
A specific library required for running the matching_algorithm.py file. please run the following two commands

```python
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
```

inside the matching_algorithm.py you can change the local feature extraction algorithms to choose between different choices which are: SIFT,SURF,& ORB.
In addition you can also change the matching algorithms  to choose between different choices which are: FLANN-based matcher and Brute Force matcher.
