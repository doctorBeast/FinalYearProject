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

# numpy.set_printoptions(threshold = numpy.nan)
# f = open('disp_data.pickle','rb')
# disp = pickle.load(f)

# print(type(disp))
# print(disp.shape)
# print(disp.dtype)
# cv2.imshow('Disparity',disp)
# cv2.waitKey(0)
# cv2.destoryAllWindows()


hj = numpy.zeros(shape = (200,200),dtype = numpy.uint8)
cv2.imshow('Check',hj)
cv2.waitKey(0)
cv2.destroyAllWindows()