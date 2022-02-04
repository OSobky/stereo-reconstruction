import cv2

import numpy as np 
import glob
from sqlalchemy import null
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
from matplotlib import pyplot as plt 

def detectAndComp(img1, img2, d="SIFT", m="FLANN", lowe_ratio=0.95):
    
    detector = null

    if d == "SIFT":
        detector = cv2.SIFT_create()
    elif d == "SURF":
        detector = cv2.xfeatures2d.SURF_create(400)
    elif d == "ORB":
        detector = cv2.ORB_create()
    elif d == "FAST":
        detector = cv2.FastFeatureDetector_create()
    else: 
        print("No detecor spciefied")
        return


    
     
    if d == "FAST":
        # find the keypoints and descriptors with SIFT
        kp1 = detector.detect(img1,None)
        kp2 = detector.detect(img2,None)
    else:
        # find the keypoints and descriptors with SIFT
        kp1, des1 = detector.detectAndCompute(img1,None)
        kp2, des2 = detector.detectAndCompute(img2,None)
    
    matcher = null
    if m == "FLANN":
        if d != "ORB":
            # FLANN parameters
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params,search_params)
        else:
            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                            table_number = 6, # 12
                            key_size = 12,     # 20
                            multi_probe_level = 1) #2
            matcher = cv2.FlannBasedMatcher(index_params)
    elif m == "BT":
        matcher = cv2.BFMatcher()


    if d == "FAST":
        matches = matcher.knnMatch(kp1,kp2,k=2)
    else:
        matches = matcher.knnMatch(des1,des2,k=2)
    
    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    #getting the matched points in appropriate format
    MatchesList=[]
    for i,(m,n) in enumerate(matches):
        if m.distance < lowe_ratio*n.distance:
            good.append([m])
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            MatchesList.append(m)
    
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    print(np.array(good).shape)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()
    # print(len(pts1))
    return pts1, pts2

img_path1 = 'data/artroom1/im0.png'
img_path2 = 'data/artroom1/im1.png'

#Load pictures
img1 = cv2.imread(img_path1,0)
img2 = cv2.imread(img_path2,0)

detectAndComp(img1, img2, d="SURF", m="BT", lowe_ratio=0.75)

