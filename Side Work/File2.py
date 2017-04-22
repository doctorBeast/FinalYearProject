# from array import array
# import pyaudio
#
# CHUNK_SIZE = 7
# FORMAT = pyaudio.paInt16
# THRESHOLD = 500
# RATE = 44100
#
# p = pyaudio.PyAudio()
# stream = p.open(format= FORMAT,channels = 2, rate = RATE,input = True , output = False, frames_per_buffer = CHUNK_SIZE)
# snd_data = stream.read(CHUNK_SIZE)
# print(snd_data)
# stream.stop_stream()
# stream.close()
# p.terminate()

'''
import audioop
import math
import pyaudio
import wave

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "file.wav"

audio = pyaudio.PyAudio()
j = 0
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
print ("recording...")
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    re = int(math.sqrt(abs(audioop.avg(data,4))))
    frames.append(data)
    print(re,end=' ')
print('\n')
print(i)
print ("finished recording")


# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()
'''

import pickle
import numpy
import cv2

def intensity_diff(x,y,i,j):
	global imgL
	avg_diff = 0
	diff_sum = 0
	for k in range(3):
		diff_sum += abs(int(imgL[y][x][k])-int(imgL[y+i][x+j][k]))

	avg_diff = diff_sum/3

	return avg_diff

def cal_distance(event,x,y,flags,param):
	global val
	global disp
	global val_AD
	global val_Census

	total = 0
	winSize = 9
	

	if event == cv2.EVENT_LBUTTONDBLCLK:
		n = 0
		actual_coordinate_x = x+40
		for i in range(int(-winSize/2),int(winSize/2)):
			for j in range(int(-winSize/2),int(winSize/2)):
				avg_diff = intensity_diff(actual_coordinate_x,y,i,j)
				if avg_diff<5:
					n+=1
					total += int(disp[y][x])
		
		total = total/n

		depth = (86 * 743)/total
		print('Depth Value',depth)
		print('Disparity :',disp[y][x])
		print('ADC_Value :',val[y][x])
		print('val_AD :',val_AD[y][x])
		print('val_Census :',val_Census[y][x])

numpy.set_printoptions(threshold = numpy.nan)
imgL = cv2.imread('/home/doctorbeast/Desktop/Github/FinalYearProject/Side Work/dstImages/dstLnew3.jpg',1)

g = open('ADC_value.pickle','rb')
val = pickle.load(g)
g.close()

f = open('disp_data.pickle','rb')
disp = pickle.load(f)
f.close()

f = open('val_AD.pickle','rb')
val_AD = pickle.load(f)
f.close()

f = open('val_Census.pickle','rb')
val_Census = pickle.load(f)
f.close()

cv2.namedWindow('Disparity')
cv2.setMouseCallback('Disparity',cal_distance)

print(disp.shape)
cv2.imshow('Disparity',disp)
cv2.waitKey(0)
cv2.imwrite('DisparityFin.jpg',disp)
cv2.destroyAllWindows()

