'''
Created by Omar Padierna "Para11ax" on Jan 1 2019
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
'''

import cv2
import numpy as np 
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
from matplotlib import pyplot as plt 

#=====================================
# Function declarations
#=====================================
def Calculate_RT(img1,img2,cam_matrix):
     #Feature detection and Matching as in Task1
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    #getting the matched points in appropriate format
    MatchesList=[]
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.95*n.distance:
            good.append([m])
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            MatchesList.append(m)
    
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3),plt.show()
    # print(len(pts1))
    
    E , mask = cv2.findEssentialMat(pts1,pts2,cam_matrix) #Finding E matrix
    pts1 = pts1[mask.ravel()==1] #eliminate outliers
    pts2 = pts2[mask.ravel()==1]
   
    _, R, t, mask = cv2.recoverPose(E,pts1,pts2,cam_matrix) # Decompose E into R AND t
    pts1 = pts1[mask.ravel()==255] #eliminate outliers
    pts2 = pts2[mask.ravel()==255]
    return R,t
#Function to create point cloud file
def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')

#Function that Downsamples image x number (reduce_factor) of times. 
def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image


#=========================================================
# Stereo 3D reconstruction 
#=========================================================

#Load camera parameters

K = np.array([[1733.74, 0 ,792.27],[ 0, 1733.74, 541.89],[ 0, 0, 1]])
#dist = np.load('./camera_params/dist.npy')

#Specify image paths
img_path1 = 'data/artroom1/im0.png'
img_path2 = 'data/artroom1/im1.png'

#Load pictures
img_1 = cv2.imread(img_path1,0)
img_2 = cv2.imread(img_path2,0)
Colored = cv2.imread(img_path1)

#Get height and width. Note: It assumes that both pictures are the same size. They HAVE to be same size and height. 
h,w = img_2.shape[:2]

# #Get optimal camera matrix for better undistortion 
# new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))

# #Undistort images
# img_1_undistorted = cv2.undistort(img_1, K, dist, None, new_camera_matrix)
# img_2_undistorted = cv2.undistort(img_2, K, dist, None, new_camera_matrix)

# #Downsample each image 3 times (because they're too big)
# img_1 = downsample_image(img_1,3)
# img_2 = downsample_image(img_2,3)

#cv2.imwrite('undistorted_left.jpg', img_1_downsampled)
#cv2.imwrite('undistorted_right.jpg', img_2_downsampled)


#Set disparity parameters
#Note: disparity range is tuned according to specific parameters obtained through trial and error. 
# win_size = 25
# min_disp = 25
# max_disp = 160 #min_disp * 9
# num_disp = 112 # Needs to be divisible by 16

#Create Block matching object. 
# stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
# 	numDisparities = num_disp,
# 	blockSize = 25,
# 	uniquenessRatio = 10,
# 	speckleWindowSize = 5,
# 	speckleRange = 5,
# 	disp12MaxDiff = 1,
# 	P1 = 8*3*win_size**2,#8*3*win_size**2,
# 	P2 =32*3*win_size**2) #32*3*win_size**2)

# #Compute disparity map
# print ("\nComputing the disparity  map...")
# disparity_map = stereo.compute(img_1, img_2)

# #Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
# plt.imshow(disparity_map,'gray')
# plt.show()
# stereo = cv2.StereoBM_create(numDisparities=32, blockSize=5)
# disparity_map = stereo.compute(img_1,img_2)
# plt.imshow(disparity_map,'gray')
# plt.show()
######################################
minDisparity = 15
numDisparities= 128
stereo = cv2.StereoBM_create()
stereo.setNumDisparities(numDisparities)
stereo.setBlockSize(37)
stereo.setPreFilterType(1)
stereo.setPreFilterSize(11)
stereo.setPreFilterCap(19)
stereo.setTextureThreshold(10)
stereo.setUniquenessRatio(7)
stereo.setSpeckleRange(25)
stereo.setSpeckleWindowSize(12)
stereo.setDisp12MaxDiff(5)
stereo.setMinDisparity(minDisparity)
disparity_map = stereo.compute(img_1,img_2)
disparity_map = disparity_map.astype(np.float32)
# disparity_map = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
disparity_map = (disparity_map/16.0 - minDisparity)/numDisparities

groundtruth = cv2.imread('data/artroom1/disp0.pfm', cv2.IMREAD_UNCHANGED)

# Remove infinite value to display
groundtruth[groundtruth==np.inf] = 0

# Normalize and convert to uint8
groundtruth = cv2.normalize(groundtruth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# # Show
# cv2.imshow("groundtruth", groundtruth)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.imshow(disparity_map,'gray')
# plt.show()
# plt.imshow(groundtruth)
# plt.show()
# breakpoint()
# f, axarr = plt.subplots(2)
# axarr[0].imshow(disparity_map,'gray')
# axarr[1].imshow(groundtruth,'gray')
plt.show()

##################################
#Generate  point cloud. 
print ("\nGenerating the 3D map...")

#Get new downsampled width and height 
h,w = img_2.shape[:2]

#Load focal length. 
focal_length = 1733.74


##############################33
# points_3D = np.zeros((h,w,3))
# for i in range(h):
#     for j in range(w):
#         Z = 536.62 * focal_length / (disparity_map[i][j] + 0)
#         points_3D[i][j] = np.array([i,j,Z])



################################3
#Perspective transformation matrix
#This transformation matrix is from the openCV documentation, didn't seem to work for me. 
Q = np.float32([[1,0,0,-w/2.0],
				[0,-1,0,h/2.0],
				[0,0,0,-focal_length],
				[0,0,1,0]])

# #This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision. 
# #Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
Q2 = np.float32([[1,0,0,0],
				[0,-1,0,0],
				[0,0,focal_length*0.05,0], #Focal length multiplication obtained experimentally. 
				[0,0,0,1]])

# #Reproject points into 3D
# Q=0
R,T = Calculate_RT(img_1,img_2,K)
_,_,_,_,QQ,_,_=cv2.stereoRectify(K, 0, K, 0,(h, w), R, T)

points_3D_gt = cv2.reprojectImageTo3D(groundtruth, Q)
points_3D = cv2.reprojectImageTo3D(disparity_map, Q)

#Get color points
colors = cv2.cvtColor(Colored, cv2.COLOR_BGR2RGB)

#Get rid of points with value 0 (i.e no depth)
mask_map_gt = groundtruth > 0
mask_map = disparity_map > 0.25

#Mask colors and points. 
output_points_gt = points_3D[mask_map_gt]
output_colors_gt = colors[mask_map_gt]
output_points = points_3D[mask_map]
output_colors = colors[mask_map]
breakpoint()

#Define name for output file
output_file_gt = 'reconstructed-gt.ply'
output_file = 'reconstructed.ply'

#Generate point cloud 
print ("\n Creating the output file... \n")
create_output(output_points_gt, output_colors_gt, output_file_gt)
create_output(output_points, output_colors, output_file)