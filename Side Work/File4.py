
'''
#code for camera calibration

import numpy as np
import cv2
import glob
import pickle
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('ImagesRight/*.jpg')
i = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(1)
        i+=1

    else:
        print(fname)
print(i)
cv2.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print(mtx)
print(dist)

# img = cv2.imread('programclick.jpg')
# h,w = img.shape[:2]
#
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h),)
#
# print('newcameramtx',newcameramtx)
#
# print('roi', roi)
#
# dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
#
# x,y,w,h = roi
#
# dst = dst[y:y+h,x:x+w]
# cv2.imwrite('calibresult.jpg',dst)

with open('Rightdata.pickle','wb') as f:
    pickle.dump([objpoints,imgpoints,mtx,dist],f)

f.close()
'''

#code for stereocamera calibration

import pickle
import cv2
import numpy as np

# np.set_printoptions(threshold=np.nan)

# img = cv2.imread('Images/img0.jpg',0)
# print(img.shape)
L = open('Leftdata.pickle','rb')
R = open('Rightdata.pickle','rb')

objL , imgL , cameraMatrix1, distCoeffs1 = pickle.load(L)
objR , imgR, cameraMatrix2 , distCoeffs2 = pickle.load(R)

L.close()
R.close()

srcL = cv2.imread('C:/Users/fsg/Github/Both/imgL0.jpg',1)
stereo_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)


retval,cameraMatrix1,distCoeffs1,cameraMatrix2,distCoeffs2,R,T,E,F = cv2.stereoCalibrate(objectPoints = objL,imagePoints1 = imgL,imagePoints2 = imgR,imageSize = (480,640),cameraMatrix1 = cameraMatrix2,distCoeffs1 = distCoeffs1,cameraMatrix2 = cameraMatrix2,distCoeffs2 = distCoeffs2, criteria = stereo_criteria,flags = cv2.CALIB_FIX_INTRINSIC)

R1,R2,P1,P2,Q,ROI1,ROI2 = cv2.stereoRectify(cameraMatrix1 = cameraMatrix1,cameraMatrix2 = cameraMatrix2,distCoeffs1 = distCoeffs1,distCoeffs2 = distCoeffs2,imageSize = (480,640),R = R,T = T,flags = cv2.CALIB_ZERO_DISPARITY,alpha = 1)
#print(R1.shape,R2.shape,P1.shape,P2.shape,Q.shape)

map1 , map2 = cv2.initUndistortRectifyMap(cameraMatrix = cameraMatrix1, distCoeffs = distCoeffs1, R = R1, newCameraMatrix = P1, size = (480,640),m1type = cv2.CV_32FC1)
dstL = cv2.remap(src = srcL, map1 = map1, map2 = map2, interpolation = cv2.INTER_LINEAR)


xl,yl,wl,hl = ROI1


# cv2.imshow('Final',dst)
# cv2.waitKey(0)

srcR = cv2.imread('C:/Users/fsg/Github/Both/imgR0.jpg',1)
map1 , map2 = cv2.initUndistortRectifyMap(cameraMatrix = cameraMatrix2, distCoeffs = distCoeffs2, R = R2, newCameraMatrix = P2, size = (480,640),m1type = cv2.CV_32FC1)
dstR = cv2.remap(src = srcR, map1 = map1, map2 = map2, interpolation = cv2.INTER_LINEAR)


xr,yr,wr,hr = ROI2
x = min([xl,xr])
y = min([yl,yr])
w = max([wl,wr])
h = max([hl,hr])-130

dstR = dstR[yr:yr+h,x:x+w]
dstL = dstL[yl:yl+h,x:x+w]

# i = 0
# while True:
#     cv2.imshow('dstL',dstL[i:i+5])
#     cv2.imshow('dstR',dstR[i:i+5])
#     if cv2.waitKey(1) & 0xFF == ord('n'):
#         i+=1
#     elif cv2.waitKey(1) & 0xFF == ord('p'):
#         i-=1

cv2.imshow('dstL',dstL)
cv2.imshow('dstR',dstR)
cv2.waitKey(0)

print(dstL.shape)
print(dstR.shape)

cv2.imwrite('dstL.jpg',dstL)
cv2.imwrite('dstR.jpg',dstR)

cv2.destroyAllWindows()

#Finding Disparity from here


# print('hello')
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
#
# def write_ply(fn, verts, colors):
#     verts = verts.reshape(-1, 3)
#     colors = colors.reshape(-1, 3)
#     verts = np.hstack([verts, colors])
#     with open(fn, 'wb') as f:
#         f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
#         np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
#
#
# imgL = dstL
# imgR = dstR
#
# window_size = 9
# min_disp = 16
# num_disp = 112-min_disp
# stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
#     numDisparities = num_disp,
#     blockSize = 16,
#     P1 = 8*3*window_size**2,
#     P2 = 32*3*window_size**2,
#     disp12MaxDiff = 1,
#     uniquenessRatio = 10,
#     speckleWindowSize = 100,
#     speckleRange = 32
# )
#
# print('computing disparity...')
# disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
# np.savetxt('disp_data.txt',disp,fmt="%.2f  ")
# np.set_printoptions(threshold=np.nan)
#
# print('generating 3d point cloud...',)
# h, w = imgL.shape[:2]
# f = 0.8*w                          # guess for focal length
# Q = np.float32([[1, 0, 0, -0.5*w],
#                 [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
#                 [0, 0, 0,     -f], # so that y-axis looks up
#                 [0, 0, 1,      0]])
# points = cv2.reprojectImageTo3D(disp, Q)
# colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
# mask = disp > disp.min()
# out_points = points[mask]
# out_colors = colors[mask]
# out_fn = 'out.ply'
# write_ply('out.ply', out_points, out_colors)
# print('%s saved' % 'out.ply')
# dispar = (disp-min_disp)/num_disp
# cv2.imshow('left', imgL)
# cv2.imshow('right',imgR)
# cv2.imshow('disparity', dispar)
# cv2.waitKey()
# cv2.destroyAllWindows()
