
import cv2
import numpy as np 
import glob
import json
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
from matplotlib import pyplot as plt 
from ast import literal_eval
import open3d as o3d

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


def Load_Images(Img1_path,Img2_path ,GreyScale):
    #Load 2 Images
    if(GreyScale):
        image1=cv2.imread(Img1_path,0)
        image2=cv2.imread(Img2_path,0)
    else:
        image1=cv2.imread(Img1_path)
        image2=cv2.imread(Img2_path)

    return image1,image2

def Load_intrinsics(Path):
    with open(Path) as f:
        lines = f.readlines()
    
    cam_matrix = lines[0].split("=")[1][:-1]
    cam_matrix = "[" + cam_matrix +"]"
    cam_matrix  = cam_matrix.replace(";","],[")
    cam_matrix =np.matrix(cam_matrix).reshape(3,3)
    return cam_matrix
    

def calculate_matches(img1,img2,cam_matrix):
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
        if m.distance < 0.8*n.distance:
            good.append([m])
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            MatchesList.append(m)
    
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()
    return pts1, pts2

def Calculate_RT(pts1,pts2,cam_matrix):
    
    E , mask = cv2.findEssentialMat(pts1,pts2,cam_matrix) #Finding E matrix
    pts1 = pts1[mask.ravel()==1] #eliminate outliers
    pts2 = pts2[mask.ravel()==1]
   
    _, R, t, mask = cv2.recoverPose(E,pts1,pts2,cam_matrix) # Decompose E into R AND t
    pts1 = pts1[mask.ravel()==255] #eliminate outliers
    pts2 = pts2[mask.ravel()==255]

    return R,t ,pts1,pts2

def disparityMap(img1,img2):
    stereo = cv2.StereoBM_create() 
    ####### Disparity Hyperparameters #######################
    minDisparity = 15
    numDisparities= 128
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
    ###################################################################################
    ############################## Compute and visualize Disparity between the two Images #####################
    disparity_map = stereo.compute(img1,img2)
    disparity_map = disparity_map.astype(np.float32)
    disparity_map = (disparity_map/16.0 - minDisparity)/numDisparities #### Normalize disparity values #############
    plt.imshow(disparity_map,'gray')
    plt.show()
    return disparity_map
    


def triangulation(pts1,pts2,cam_matrix,R,t , coloredImage):
    T = np.concatenate((R,t),axis =1)
    #T_h = np.concatenate((T,np.array([[0,0,0,1]])),axis =0)
    PM2 = np.matmul(cam_matrix,T)
    PM1 = np.matmul(cam_matrix,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
    
    
    #T_inv = np.linalg.inv(T_h) 
    #T=T_inv
    points_3d=[]
    colors = []
    A = np.zeros((4,4))
    for i in range(len(pts1)):
        # point1 = kp1[m.queryIdx].pt
        # point2 = kp2[m.trainIdx].pt
        point1 = pts1[i]
        
        point2 = pts2[i]
        A[0]=(point1[1]*PM1[2])- PM1[1]
        A[1]= PM1[0] -(point1[0]*PM1[2])
        A[2]=(point2[1]*PM2[2])- PM2[1]
        A[3] = PM2[0] -(point2[0]*PM2[2])
       
        
        W,V =np.linalg.eig(np.matmul(A.T,A))
        point_world=V[:,np.argmin(W)]
        point_world= (1/point_world[3])*point_world
        points_3d.append(point_world[:3])
        #breakpoint()
        colors.append(coloredImage[point1[1],point1[0]])
    points_3d = np.array(points_3d)
    colors = np.array(colors)
    #breakpoint()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    #pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    
    return points_3d , colors



def stereo_reconstruct(img1,img2,colored_image1,colored_image2,cam_matrix):
    pts1 , pts2 = calculate_matches(img1,img2,cam_matrix)
    R , t ,pts1,pts2= Calculate_RT(pts1,pts2,cam_matrix)
    # h,w = img1.shape[:2]
    # _,_,_,_,QQ,_,_=cv2.stereoRectify(cam_matrix, 0, cam_matrix, 0,(h, w), R, t)
    # disparity_map = disparityMap(img1,img2)
    
    # points_3D = cv2.reprojectImageTo3D(disparity_map, QQ)
    # #Get color points
    colors = cv2.cvtColor(colored_image1, cv2.COLOR_BGR2RGB)
  
    # #Get rid of points with value 0 (i.e no depth)
    # mask_map = disparity_map > 0.25

    # #Mask colors and points. 
    # output_points = points_3D[mask_map]
    # output_colors = colors[mask_map]


    # #Define name for output file
    # output_file = 'reconstructedDisparity.ply'

    # #Generate point cloud 
    # print ("\n Creating the output file... \n")
    # create_output(output_points, output_colors, output_file)

    points_3d ,colors= triangulation(pts1,pts2,cam_matrix,R,t,colors)
    output_file = 'reconstructedTriang.ply'

    create_output(points_3d, colors, output_file)
    return

def stereo_reconstruct_trig(img1,img2,camera_matrix):
    return 



if __name__ == '__main__':
    img1_path = "im0.png"
    img2_path = "im1.png"
    calib = "calib.txt"
    img1,img2  = Load_Images(img1_path,img2_path, True)
    colored_image1,colored_image2  = Load_Images(img1_path,img2_path, False)
    cam_matrix = Load_intrinsics(calib)
    stereo_reconstruct(img1,img2,colored_image1,colored_image2,cam_matrix)
    breakpoint()
    
