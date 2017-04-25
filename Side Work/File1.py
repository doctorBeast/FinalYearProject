'''
#program to click pictures
import cv2
import time

camera_port = 0
camera = cv2.VideoCapture(camera_port)

def get_image():
    global camera
    retval,im = camera.read()
    return im

file = 'Images/img'

for i in range(60):
    while True:
        ret,frame = camera.read()
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            img = frame
            break

    #img = get_image()
    path = file + '%s.jpg' %i
    cv2.imwrite(path,img)
    print(i)


del(camera)
'''
'''
# Program to click picture when i make a sound strong enough in an almost silent environment

import cv2
import audioop
import pyaudio
import math
import time

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

camera = cv2.VideoCapture(0)
path = 'Images/'
j = 0
k = 11
while True:

    # data = stream.read(CHUNK)
    # re = int(math.sqrt(abs(audioop.avg(data,4))))
    # if re>1500 and k>10:
    #     k=0
    #     print(re)
    #     time.sleep(5)
    retval,frame = camera.read()
    for i in range(len(frame)):
        frame[i] = frame[i][::-1]
    cv2.imshow('Video',frame)
    cv2.waitKey(1)
    data = stream.read(CHUNK)
    re = int(math.sqrt(abs(audioop.avg(data,4))))

    if re>1500 and k>10:
        k = 0
        print(re)
        file = path + 'img%s.jpg' %j
        j+=1
        cv2.imwrite(file,frame)
        print('hello')
        if j == 50:
            break
    k+=1

del(camera)
stream.stop_stream()
stream.close()
'''

'''
#capturing images for both cameras
import cv2

camera1 = cv2.VideoCapture(1)
camera2 = cv2.VideoCapture(2)


fileL = '/home/doctorbeast/Desktop/Github/ImagesLeftNew3/img'
fileR = '/home/doctorbeast/Desktop/Github/ImagesRightNew3/img'

for i in range(60):
    while True:
        retL,frameL = camera1.read()
        cv2.imshow('ImageLeft',frameL)
        print(frameL.shape)
        retR,frameR = camera2.read()
        cv2.imshow('ImageRight',frameR)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            imgL = frameL
            imgR = frameR
            break

    #img = get_image()
    pathL = fileL + '%s.jpg' %i
    pathR = fileR + '%s.jpg' %i
    cv2.imwrite(pathL,imgL)
    cv2.imwrite(pathR,imgR)
    print(i)

del(camera1,camera2)
'''

'''
#TOTAL 5 CHANGES
import cv2
import pickle
import cv2
import numpy as np

#camera0 = cv2.VideoCapture(0)
camera1 = cv2.VideoCapture(1)
camera2 = cv2.VideoCapture(2)
i = 6
file = '/home/doctorbeast/Desktop/Github/FinalYearProject/BothNew2/'

while True:
    retL,frameL = camera1.read()
    retR, frameR = camera2.read()
    cv2.imshow('Left',frameL[49:438,177:505])
    cv2.imshow('Right',frameR[49:438,177:505])
    if cv2.waitKey(1) & 0xFF == ord('c'):
        pathL = file + 'imgL%s.jpg' %i
        pathR = file + 'imgR%s.jpg' %i
        cv2.imwrite(pathL,frameL)
        cv2.imwrite(pathR,frameR)
        break

# #code for stereocamera calibration


# np.set_printoptions(threshold=np.nan)0

L = open('Leftdata.pickle','rb')
R = open('Rightdata.pickle','rb')

objL , imgL , cameraMatrix1, distCoeffs1 = pickle.load(L)
objR , imgR, cameraMatrix2 , distCoeffs2 = pickle.load(R)

L.close()
R.close()

srcL = cv2.imread('/home/doctorbeast/Desktop/Github/FinalYearProject/BothNew2/imgL6.jpg',1)
stereo_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

retval,cameraMatrix1,distCoeffs1,cameraMatrix2,distCoeffs2,R,T,E,F = cv2.stereoCalibrate(objectPoints = objL,imagePoints1 = imgL,imagePoints2 = imgR,imageSize = (640,480),cameraMatrix1 = cameraMatrix1,distCoeffs1 = distCoeffs1,cameraMatrix2 = cameraMatrix2,distCoeffs2 = distCoeffs2, criteria = stereo_criteria,flags = cv2.CALIB_FIX_INTRINSIC)
print(cameraMatrix1)
print(cameraMatrix2)
R1,R2,P1,P2,Q,ROI1,ROI2 = cv2.stereoRectify(cameraMatrix1 = cameraMatrix1,cameraMatrix2 = cameraMatrix2,distCoeffs1 = distCoeffs1,distCoeffs2 = distCoeffs2,imageSize = (640,480),R = R,T = T,flags = cv2.CALIB_ZERO_DISPARITY,alpha = -1)
#print(R1.shape,R2.shape,P1.shape,P2.shape,Q.shape)

map1 , map2 = cv2.initUndistortRectifyMap(cameraMatrix = cameraMatrix1, distCoeffs = distCoeffs1, R = R1, newCameraMatrix = P1, size = (640,480),m1type = cv2.CV_32FC1)
dstmap1 , dstmap2 = cv2.convertMaps(map1 = map1.astype(np.float32),map2 = map2.astype(np.float32),dstmap1type = cv2.CV_16SC2,nninterpolation = False)
dstL = cv2.remap(src = srcL, map1 = dstmap1, map2 = dstmap2, interpolation = cv2.INTER_LINEAR)

xl,yl,wl,hl = ROI1

# cv2.imshow('Final',dst)
# cv2.waitKey(0)

srcR = cv2.imread('/home/doctorbeast/Desktop/Github/FinalYearProject/BothNew2/imgR6.jpg',1)

map1 , map2 = cv2.initUndistortRectifyMap(cameraMatrix = cameraMatrix2, distCoeffs = distCoeffs2, R = R2, newCameraMatrix = P2, size = (640,480),m1type = cv2.CV_32FC1)
dstmap1 , dstmap2 = cv2.convertMaps(map1 = map1.astype(np.float32),map2 = map2.astype(np.float32),dstmap1type = cv2.CV_16SC2,nninterpolation = False)
dstR = cv2.remap(src = srcR, map1 = dstmap1, map2 = dstmap2, interpolation = cv2.INTER_LINEAR)


xr,yr,wr,hr = ROI2
x = min([xl,xr])
y = max([yl,yr])
w = min([wl,wr])
h = max([hl,hr])

# dstL = dstL[yl:yl+hl,xl:xl+wl]
# dstR = dstR[yr:yr+hr,xr:xr+wr]

# dstL = dstL[yl:yl+hl,xl:xl+w]
# dstR = dstR[yr:yr+hr,xr:xr+w]

# i = 0
# while True:
#     cv2.imshow('dstL',dstL[i:i+5])
#     cv2.imshow('dstR',dstR[i:i+5])
#     if cv2.waitKey(1) & 0xFF == ord('n'):
#         i+=1
#     elif cv2.waitKey(1) & 0xFF == ord('p'):
#         i-=1

dstL = dstL[49:438,177:505]
dstR = dstR[49:438,177:505]
cv2.imshow('dstL',dstL)
cv2.imshow('dstR',dstR)
cv2.waitKey(0)

print(dstL.shape)
print(dstR.shape)

cv2.imwrite('dstImages/dstLnew6.jpg',dstL)
cv2.imwrite('dstImages/dstRnew6.jpg',dstR)

cv2.destroyAllWindows()
'''


# TOTAL 2 CHANGES
import Disparity
import cv2
import pickle

imgL = cv2.imread('/home/doctorbeast/Desktop/Github/FinalYearProject/Side Work/dstImages/dstLnew6.jpg',1)
imgR = cv2.imread('/home/doctorbeast/Desktop/Github/FinalYearProject/Side Work/dstImages/dstRnew6.jpg',1)
imgL = imgL[100:320,:]
imgR = imgR[100:320,:]
print(imgL.shape)
print(imgR.shape)

disp,val_T,val_AD,val_C = Disparity.compute(imgL,imgR)
with open('disp_data.pickle','wb') as f:
    pickle.dump(disp,f)

f.close()

with open('ADC_value.pickle','wb') as f:
    pickle.dump(val_T,f)

f.close()

with open('val_AD.pickle','wb') as f:
    pickle.dump(val_AD,f)

f.close()

with open('val_Census.pickle','wb') as f:
    pickle.dump(val_C,f)

f.close()
