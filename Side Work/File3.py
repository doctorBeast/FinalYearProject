
'''
import numpy as np
import cv2



img_L = cv2.imread('tape1left.jpg',1)
img_R = cv2.imread('tape1right.jpg',1)

image_part_L = np.array([])
image_part_R = np.array([])
start_num = int(input('Starting Row: '))
end_num = int(input('Ending Row: '))

image_part_L = img_L[start_num:end_num+1]
#Here we have to add zeros at the start and end of the list for left image
a = np.array([0 for _ in range(0,start_num*576*3)])
b = np.array([0 for _ in range(0,(768-end_num-1)*576*3)])
image_part_L = np.append(a,image_part_L)
image_part_L = np.append(image_part_L,b)
image_part_L = image_part_L.reshape(768,576,3)
image_part_L = image_part_L.astype(np.uint8)
# print(np.ma.size(image_part_L,0))
# print(np.ma.size(image_part_L,1))
# print(np.ma.size(image_part_L,2))
# print(len(image_part_L[0]))

image_part_R = img_R[start_num:end_num+1]
#Here we have to add zeros at the start and end of the list for right image
image_part_R = np.append(a,image_part_R)
image_part_R = np.append(image_part_R,b)
image_part_R = image_part_R.reshape(768,576,3)
image_part_R = image_part_R.astype(np.uint8)

cv2.imshow('ImagePartRight',image_part_R)
cv2.imshow('ImagePartLeft',image_part_L)
cv2.waitKey(0)
'''

#This code was copied from stackoverflow
'''

import numpy as np
import cv2

print "Welcome\n"

numBoards = 30  #how many boards would you like to find
board_w = 7
board_h = 6

board_sz = (7,6)
board_n = board_w*board_h

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
object_points = [] # 3d point in real world space
imagePoints1 = [] # 2d points in image plane.
imagePoints2 = [] # 2d points in image plane.

corners1 = []
corners2 = []

#obj = []
#for j in range(0,board_n):
    #obj.append(np.(j/board_w, j%board_w, 0.0))
obj = np.zeros((6*7,3), np.float32)
obj[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)



vidStreamL = cv2.VideoCapture('left.mp4')  # index of your camera
vidStreamR = cv2.VideoCapture('right.mp4')  # index of your camera
success = 0
k = 0
found1 = False
found2 = False

while (success < numBoards):

   retL, img1 = vidStreamL.read()
   height, width, depth  = img1.shape
   retR, img2 = vidStreamR.read()
   #resize(img1, img1, Size(320, 280));
   #resize(img2, img2, Size(320, 280));
   gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
   gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

   found1, corners1 = cv2.findChessboardCorners(img1, board_sz)
   found2, corners2 = cv2.findChessboardCorners(img2, board_sz)

   if (found1):
       cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1),criteria)
       cv2.drawChessboardCorners(gray1, board_sz, corners1, found1)

   if (found2):
       cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
       cv2.drawChessboardCorners(gray2, board_sz, corners2, found2)

   cv2.imshow('image1', gray1)
   cv2.imshow('image2', gray2)

   k = cv2.waitKey(100)
   print k
   if (k == 27):
       break
   if (k == 32 and found1 != 0 and found2 != 0):

       imagePoints1.append(corners1);
       imagePoints2.append(corners2);
       object_points.append(obj);
       print "Corners stored\n"
       success+=1

       if (success >= numBoards):
           break

cv2.destroyAllWindows()
print "Starting Calibration\n"
cameraMatrix1 = cv2.cv.CreateMat(3, 3, cv2.CV_64FC1)
cameraMatrix2 = cv2.cv.CreateMat(3, 3, cv2.CV_64FC1)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(object_points, imagePoints1, imagePoints2, (width, height))
## , cv2.cvTermCriteria(cv2.CV_TERMCRIT_ITER+cv2.CV_TERMCRIT_EPS, 100, 1e-5),   cv2.CV_CALIB_SAME_FOCAL_LENGTH | cv2.CV_CALIB_ZERO_TANGENT_DIST)
#cv2.cv.StereoCalibrate(object_points, imagePoints1, imagePoints2, pointCounts, cv.fromarray(K1), cv.fromarray(distcoeffs1), cv.fromarray(K2), cv.fromarray(distcoeffs2), imageSize, cv.fromarray(R), cv.fromarray(T), cv.fromarray(E), cv.fromarray(F), flags = cv.CV_CALIB_FIX_INTRINSIC)
#FileStorage fs1("mystereocalib.yml", FileStorage::WRITE);
# fs1 << "CM1" << CM1;
#fs1 << "CM2" << CM2;
# #fs1 << "D1" << D1;
#fs1 << "D2" << D2;
#fs1 << "R" << R;
#fs1 << "T" << T;
#fs1 << "E" << E;
#fs1 << "F" << F;
print "Done Calibration\n"
print "Starting Rectification\n"
R1 = np.zeros(shape=(3,3))
R2 = np.zeros(shape=(3,3))
P1 = np.zeros(shape=(3,3))
P2 = np.zeros(shape=(3,3))

#(roi1, roi2) = cv2.cv.StereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,(width, height), R, T, R1, R2, P1, P2, Q=None, flags=cv2.cv.CV_CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0, 0))
cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,(width, height), R, T, R1, R2, P1, P2, Q=None, flags=cv2.cv.CV_CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0,0))
#stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,(width, height), R, T)

#fs1 << "R1" << R1;
#fs1 << "R2" << R2;
#fs1 << "P1" << P1;
#fs1 << "P2" << P2;
#fs1 << "Q" << Q;

print "Done Rectification\n"
print "Applying Undistort\n"



map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (width, height), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (width, height), cv2.CV_32FC1)

print "Undistort complete\n"

while(True):
    retL, img1 = vidStreamL.read()
    retR, img2 = vidStreamR.read()
    imgU1 = np.zeros((height,width,3), np.uint8)
    imgU1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)
    imgU2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    cv2.imshow("imageL", img1);
    cv2.imshow("imageR", img2);
    cv2.imshow("image1L", imgU1);
    cv2.imshow("image2R", imgU2);
    k = cv2.waitKey(5);
    if(k==27):
        break;

'''

'''
import cv2
from matplotlib import pyplot as plt
img1 = cv2.imshow('tape1left.jpg',0)
img2 = cv2.imshow('tape1right.jpg',0)

stereo = cv2.createStereoBM(numDisparities=16, blockSize=15)
disp = stereo.compute(img1,img2)
plt.imshow(disp,'gray')
plt.show()
'''


#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.
Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility


# import numpy as np
# import cv2

# ply_header = '''ply
# format ascii 1.0
# element vertex %(vert_num)d
# property float x
# property float y
# property float z
# property uchar red
# property uchar green
# property uchar blue
# end_header
# '''

# def write_ply(fn, verts, colors):
#     verts = verts.reshape(-1, 3)
#     colors = colors.reshape(-1, 3)
#     verts = np.hstack([verts, colors])
#     with open(fn, 'wb') as f:
#         f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
#         np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


# if __name__ == '__main__':
#     print('loading images...')
#     #imgL = cv2.pyrDown( cv2.imread('duststoolleft.jpg') )  # downscale images for faster processing
#     #imgR = cv2.pyrDown( cv2.imread('duststoolright.jpg') )

#     imgL = cv2.imread('duststoolleft.jpg')
#     imgR = cv2.imread('duststoolright.jpg')
#     # disparity range is tuned for 'aloe' image pair
#     window_size = 9
#     min_disp = 48#16
#     num_disp = 112-min_disp
#     stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
#         numDisparities = num_disp,
#         blockSize = 16,
#         P1 = 8*3*window_size**2,
#         P2 = 32*3*window_size**2,
#         disp12MaxDiff = -1,#1,
#         uniquenessRatio = 10,
#         speckleWindowSize = 100,
#         speckleRange = 32
#     )

#     print('computing disparity...')
#     disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
#     np.savetxt('disp_data.txt',disp,fmt="%.2f  ")
#     np.set_printoptions(threshold=np.nan)

#     print('generating 3d point cloud...',)
#     h, w = imgL.shape[:2]
#     f = 0.8*w                          # guess for focal length
#     Q = np.float32([[1, 0, 0, -0.5*w],
#                     [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
#                     [0, 0, 0,     -f], # so that y-axis looks up
#                     [0, 0, 1,      0]])
#     points = cv2.reprojectImageTo3D(disp, Q)
#     colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
#     mask = disp > disp.min()
#     out_points = points[mask]
#     out_colors = colors[mask]
#     out_fn = 'out.ply'
#     write_ply('out.ply', out_points, out_colors)
#     print('%s saved' % 'out.ply')
#     dispar = (disp-min_disp)/num_disp
#     cv2.imshow('left', imgL)
#     cv2.imshow('right',imgR)
#     cv2.imshow('disparity', dispar)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

'''
#Code to see sections of an image

import cv2
import numpy

imgL = cv2.imread('testL.jpg',1)
imgR = cv2.imread('testR.jpg',1)
i = 0
while i <len(imgL):
  frameL = imgL[i:i+5,0:]
  frameR = imgR[i:i+5,0:]
  cv2.imshow('Left',frameL)
  cv2.imshow('Right',frameR)
  if cv2.waitKey(0) & 0xFF == ord('m'):
    i+=1
  else:
    i-=1


cv2.destroyAllWindows()


'''
#Code to see corresponding pixels

import cv2
import numpy
import pickle

def find_intensity_left(event,x,y,flags,param):
  global imgL
  if event == cv2.EVENT_LBUTTONDBLCLK:
    print('Left Image :',x,y)

def find_intensity_right(event,x,y,flags,param):
  global imgR
  if event == cv2.EVENT_LBUTTONDBLCLK:
    print('Right Image :',x,y)

def find_intensity_disp(event,x,y,flags,param):
  global disp
  if event == cv2.EVENT_LBUTTONDBLCLK:
    print('Disparity Image :',x,y)

def find_same_color(event,x,y,flags,param):
  global imgL,imgR
  # imgL = imgL.astype(numpy.int16)
  # imgR = imgR.astype(numpy.int16)
  if event == cv2.EVENT_LBUTTONDBLCLK:
    ref = imgL[y][x]
    
    for i in range(len(imgR[0])):
      sod = 0
      test_sample = imgR[y][i]
      for j in range(3):
        sod += abs(int(ref[j])-int(test_sample[j]))

      avg_sod = sod/3
      if avg_sod <10:
        
        for j in range(3):
          imgR[y][i][j] = 0

    # imgL = imgL.astype(numpy.uint8)
    # imgR = imgR.astype(numpy.uint8)
    cv2.imshow('Right',imgR)
    cv2.waitKey(0)

imgL = cv2.imread('/home/doctorbeast/Desktop/Github/FinalYearProject/Side Work/dstImages/dstLnew1.jpg',1)
imgR = cv2.imread('/home/doctorbeast/Desktop/Github/FinalYearProject/Side Work/dstImages/dstRnew1.jpg',1)
f = open('PickleFiles/rect15-215winSiz17.pickle','rb')
disp = pickle.load(f)
f.close()
cv2.namedWindow('Left')
cv2.setMouseCallback('Left',find_intensity_left)

cv2.namedWindow('Right')
cv2.setMouseCallback('Right',find_intensity_right)

cv2.namedWindow('Disparity')
cv2.setMouseCallback('Disparity',find_intensity_disp)

cv2.imshow('Left',imgL)
cv2.imshow('Right',imgR)
cv2.imshow('Disparity',disp)
cv2.waitKey(0)

cv2.destroyAllWindows()
# cv2.imwrite('testL.jpg',imgL)
# cv2.imwrite('testR.jpg',imgR)
