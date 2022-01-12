import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import open3d as o3d
import sys
Intrinsic_Path = "camera_intrinsic.json"
Extrinsic_Path = "transforms.json"
def Load_Images(Folder,Img1,Img2):
    #Load 2 Images
    Path1 = Folder+"/image_"+ "{number:05}".format(number=Img1)+".png"
    Path2 = Folder+"/image_"+ "{number:05}".format(number=Img2)+".png"
    image1=cv2.imread(Path1,0)
    image2=cv2.imread(Path2,0)
    return image1,image2

def Load_intrinsics(Path):
    #Load the intrinsic camera matrix
    with open(Path) as f:
        data = json.load(f)
        return np.array(data)

def Load_extrinsics(Path,Img1,Img2,cam):
    #Load the ground truth matrices
    Img1 = "image_"+"{number:05}".format(number=Img1)
    Img2 = "image_"+"{number:05}".format(number=Img2)
    with open(Path) as f:
        data = json.load(f)
        
        Img1 = np.array( data[Img1][cam])
        Img2 = np.array(data[Img2][cam])
        return Img1,Img2

def Find_matching_Points(img1,img2):
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
        if m.distance < 0.7*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            MatchesList.append(m)
    
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    cam_matrix = Load_intrinsics(Intrinsic_Path) #Loading intrinsics
    E , mask = cv2.findEssentialMat(pts1,pts2,cam_matrix) #Finding E matrix
    pts1 = pts1[mask.ravel()==1] #eliminate outliers
    pts2 = pts2[mask.ravel()==1]
   
    _, R, t, mask = cv2.recoverPose(E,pts1,pts2,cam_matrix) # Decompose E into R AND t
    pts1 = pts1[mask.ravel()==255] #eliminate outliers
    pts2 = pts2[mask.ravel()==255]
    ################ Linear Triangulation Method  ##############
    #ExtrinsicsGT1,ExtrinsicsGT2 = Load_extrinsics("transforms.json",0,1,0)
    T = np.concatenate((R,t),axis =1)
    #T_h = np.concatenate((T,np.array([[0,0,0,1]])),axis =0)
    PM2 = np.matmul(cam_matrix,T)
    PM1 = np.matmul(cam_matrix,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
    cam_matrix_Inv = np.linalg.inv(cam_matrix)
    
    #T_inv = np.linalg.inv(T_h) 
    #T=T_inv
    points_3d=[]
    A = np.zeros((4,4))
    for i in range(len(pts1)):
        # point1 = kp1[m.queryIdx].pt
        # point2 = kp2[m.trainIdx].pt
        point1 = pts1[i]
        point2 = pts2[i]
        A[0]=(point1[1]*PM1[2])- PM1[1]
        A[1]=(point1[0]*PM1[2])- PM1[0]
        A[2]=(point2[1]*PM2[2])- PM2[1]
        A[3] =(point2[0]*PM2[2])- PM2[0]
       
        
        W,V =np.linalg.eig(np.matmul(A.T,A))
        point_world=V[:,np.argmin(W)]
        point_world= (1/point_world[3])*point_world
        points_3d.append(point_world[:3])
   
    points_3d = np.array(points_3d)
    
    #Claculate_Error(R,t)
    
    
    ##################Uncomment the followinf for visualizations#####################################
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.array(points_3d))
    # o3d.io.write_point_cloud("./data.ply", pcd)
    # o3d.visualization.draw_geometries([pcd])
   
    # final_img = cv2.drawMatches(img1, kp1,img2, kp2, MatchesList,None)
    # final_img = cv2.resize(final_img, (1000,650))
    # cv2.imshow("Matches", final_img)
    # cv2.waitKey(3000)
    return R,t, points_3d , pts1,pts2
def Claculate_Error(R,t,img1,img2,cam):
    #Calculate the transaltional and rotational error of the estimated R,T Wwith respect to the ground truth.
    ExtrinsicsGT1,ExtrinsicsGT2 = Load_extrinsics(Extrinsic_Path,img1,img2,cam)
    Inv =np.linalg.inv(ExtrinsicsGT2)
    T_t = np.matmul(Inv,ExtrinsicsGT1)
    
    R_t = T_t[:3,:3]
    t_t = T_t[:3,3]
    delta_R = np.matmul(R.T,R_t)
    delta_t = np.matmul(R,(t_t-t.T[0]))
    Rotation_error =np.arccos(np.trace((delta_R-1)/2))
    translation_error = np.linalg.norm(delta_t)
   
    return Rotation_error , translation_error

def Task32_init(start,end,Path,cam):
    #This method is used for Task3.2 intilaziation. loads a sequence of frames and 
    # returns estiamted rotaion and translation matrices along with 2d and 3d points per frame.
    cam_matrix = Load_intrinsics(Intrinsic_Path)
    C=start+1
    PtsFF , PtsDF ,Points3D,RotatoinMatrices,Translations = ([] for i in range(5))
    while(C<=end):
        
        img1,img2 = Load_Images(Path,start,C)
        R,T,Points,pts1,pts2 =Find_matching_Points(img1,img2)
        Points3D.append(Points)
        PtsFF.append(pts1)
        PtsDF.append(pts2)
        RotatoinMatrices.append(R)
        Translations.append(T)
        C =C+1
    
    
    
    return PtsDF ,Points3D ,RotatoinMatrices , Translations


if __name__ == '__main__':
    Path = sys.argv[1]
    FirstFrame = int(sys.argv[2])
    SecondFrame= int(sys.argv[3])
    cam = int(sys.argv[4])
    
    Path =Path+"CameraRGB"+str(cam)
    img1,img2 = Load_Images(Path,FirstFrame,SecondFrame)
    
    R,t, _ , _,_=Find_matching_Points(img1,img2)
    Rotation_error , translation_error =Claculate_Error(R,t,FirstFrame,SecondFrame,cam)
    print("Rotational Error is "+str(Rotation_error))
    print("Translational Error is "+str(translation_error))
  