import cv2
import numpy
cimport numpy
import math
import time
import pickle
from cython.parallel import parallel,prange
cimport openmp
from libc.stdlib cimport malloc,free
import cython

cdef int minDisparity,maxDisparity,windowSize,lambda_Census,lambda_AD

minDisparity = 20          #MAKE CHANGES
maxDisparity = 130           #MAKE CHANGES
windowSize = 9      #Window size must be always taken odd
lambda_Census = 30
lambda_AD = 10


cpdef compute(numpy.ndarray[numpy.uint8_t,ndim = 3] imgL,numpy.ndarray[numpy.uint8_t,ndim = 3] imgR):
    cdef numpy.ndarray[numpy.uint8_t,ndim = 3] col,templ
    cdef numpy.ndarray[numpy.uint8_t,ndim = 2] disp
    cdef numpy.ndarray[numpy.float_t,ndim = 2] ADC_val,AD_val,Census_val
    cdef int B,G,R,start_index,i,j,y,x,num_threads,first,second
    cdef float val_T,val_AD,val_C
    start_time = time.time()
    start_index = 35
    print(start_index)
    
    temp = imgL[0:,start_index:]
    disp = numpy.zeros(shape = (temp.shape[0],temp.shape[1]),dtype = numpy.uint8)
    ADC_val = numpy.zeros(shape = (temp.shape[0],temp.shape[1]))
    AD_val = numpy.zeros(shape = (temp.shape[0],temp.shape[1]))
    Census_val = numpy.zeros(shape = (temp.shape[0],temp.shape[1]))
    first = int((windowSize-1)/2)
    second = len(temp)
    stop = (second-first)+1
    with nogil,parallel(num_threads=4):
        for i in prange(first,stop):
            with gil:
                print(i,'      ',time.time()-start_time)
                for j in range(int((windowSize-1)/2),len(temp[0])-int(((windowSize-1)/2))+1):
                    y , x , val_T,val_AD,val_C = find_closest_match(i,j+start_index,imgL,imgR)           #Define this funtion
                    if val_T>0.3:
                        disp[i][j] = 0
                    else:
                        disp[i][j] = j+start_index-x
           
                    ADC_val[i][j] = val_T
                    AD_val[i][j] = val_AD
                    Census_val[i][j] = val_C


    return disp,ADC_val,AD_val,Census_val

cdef find_closest_match(int row,int column,numpy.ndarray[numpy.uint8_t,ndim = 3] imgL,numpy.ndarray[numpy.uint8_t,ndim = 3] imgR):
    cdef int flag,row_found,column_found,j
    cdef float ADC,ADC_closest
    ADC_closest = 0
    AD_closest = 0
    Census_closest = 0
    flag = 0
    row_found = row
    column_found = column
    for j in range(column-minDisparity,column-(maxDisparity+1),-1):
        if j<int((windowSize+1)/2):
            break
        else:
            ADC,AD,Census = AD_Census(row,column,imgL,row,j,imgR)

        if flag == 0:
            ADC_closest = ADC
            AD_closest = AD
            Census_closest = Census
            flag = 1
            row_found = row
            column_found = j
        else:
            if ADC < ADC_closest:# We are taking the assumption that the pixel which is most similar will give the least value of ADC
                ADC_closest = ADC
                AD_closest = AD
                Census_closest = Census
                row_found = row
                column_found = j

    return row_found,column_found,ADC_closest,AD_closest,Census_closest

cdef AD_Census(int rowL,int colL,numpy.ndarray[numpy.uint8_t,ndim = 3] imgL,int rowR,int colR,numpy.ndarray[numpy.uint8_t,ndim = 3] imgR):
    cdef float cost_AD,cost_Total
    cdef int cost_Census,g
    g = 1
    # First is the AD method
    cost_AD = cal_AD(rowL,colL,imgL,rowR,colR,imgR)
    if cost_AD<10:
        # Second is the Census transform
        cost_Census = cal_Census(rowL,colL,imgL,rowR,colR,imgR)
        cost_Total = (1-math.exp(-(cost_Census/lambda_Census)))+(1-math.exp(-(cost_AD/lambda_AD)))
        return cost_Total,cost_AD,cost_Census
    else:
        return g,g,g

cdef float cal_AD(int rowL,int colL,numpy.ndarray[numpy.uint8_t,ndim = 3] imgL,int rowR,int colR,numpy.ndarray[numpy.uint8_t,ndim = 3] imgR):
    cdef float cost
    cdef int i
    cost = 0
    for i in range(3):
        cost += abs(int(imgL[rowL][colL][i]) - int(imgR[rowR][colR][i]))

    cost /= 3

    return cost

cdef int cal_Census(int rowL,int colL,numpy.ndarray[numpy.uint8_t,ndim = 3] imgL,int rowR,int colR,numpy.ndarray[numpy.uint8_t,ndim = 3] imgR):
    cdef int dist,i,j,k
    cdef numpy.ndarray[numpy.uint8_t,ndim = 1] cpL,cpR,pL,pR 
    
    dist = 0
    cpL = imgL[rowL][colL]
    cpR = imgR[rowR][colR]

    for i in range(int(-(windowSize-1)/2),int((windowSize-1)/2)):
        for j in range(int(-(windowSize-1)/2),int((windowSize-1)/2)):
            pL = imgL[rowL+i][colL+j]
            pR = imgR[rowR+i][colR+j]
            for k in range(3):
                if ((int(cpL[k])-int(pL[k]))*(int(cpR[k])-int(pR[k])))<0:
                    dist+=1

    return dist
