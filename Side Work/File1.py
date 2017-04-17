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


#capturing images for both cameras
import cv2

camera1 = cv2.VideoCapture(2)
camera2 = cv2.VideoCapture(1)


fileL = 'ImagesLeft/img'
fileR = 'ImagesRight/img'

for i in range(60):
    while True:
        retL,frameL = camera1.read()
        cv2.imshow('ImageLeft',frameL)

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
import cv2

camera1 = cv2.VideoCapture(2)
camera2 = cv2.VideoCapture(1)
i = 3
file = 'Both/'

retL,frameL = camera1.read()
retR, frameR = camera2.read()

pathL = file + 'imgL%s.jpg' %i
pathR = file + 'imgR%s.jpg' %i
cv2.imshow('Left',frameL)
cv2.imshow('Right',frameR)
cv2.waitKey(0)

cv2.imwrite(pathL,frameL)
cv2.imwrite(pathR,frameR)
'''

'''
import Disparity
import cv2
import pickle

imgL = cv2.imread('dstImages/dstL0.jpg',1)
imgR = cv2.imread('dstImages/dstR0.jpg',1)

disp,val = Disparity.compute(imgL,imgR)
with open('disp_data.pickle','wb') as f:
    pickle.dump(disp,f)

f.close()

with open('ADC_value.pickle','wb') as g:
    pickle.dump(val,g)

g.close()

print(type(disp))

print(type(val))
'''