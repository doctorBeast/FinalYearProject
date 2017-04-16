import cv2
import numpy
cimport numpy
import math
import time
import pickle

cdef int minDisparity,maxDisparity,windowSize,lambda_Census,lambda_AD

minDisparity = 20
maxDisparity = 180
windowSize = 17       #Window size must be always taken odd
lambda_Census = 30
lambda_AD = 10


cpdef compute(numpy.ndarray[numpy.uint8_t,ndim = 3] imgL,numpy.ndarray[numpy.uint8_t,ndim = 3] imgR):
    cdef numpy.ndarray[numpy.uint8_t,ndim = 3] col,temp
    cdef numpy.ndarray[numpy.uint8_t,ndim = 2] disp,ADC_val
    cdef int B,G,R,start_index,i,j,y,x
    start_time = time.time()
    col = imgR[0:1,0:]
    B,G,R = find_col_BGR(col)                            #Define this function
    start_index = find_start_index(B,G,R,imgL)           #Define this function
    temp = imgL[0:,start_index:]
    disp = numpy.zeros(shape = (temp.shape[0],temp.shape[1]),dtype = numpy.uint8)
    ADC_val = numpy.zeros(shape = (temp.shape[0],temp.shape[1]),dtype = numpy.uint8)
    for i in range(int((windowSize-1)/2),len(temp)-(int((windowSize-1)/2))+1):
        print(i,'      ',time.time()-start_time)
        for j in range(int((windowSize-1)/2),len(temp[0])-int(((windowSize-1)/2))+1):
            y , x , val = find_closest_match(i,j+start_index,imgL,imgR)           #Define this funtion
            disp[i][j] = j+start_index-x
            ADC_val[i][j] = val

    return disp,ADC_val

cdef find_col_BGR(numpy.ndarray[numpy.uint8_t,ndim = 3] col):
    cdef int B,G,R,i
    B=0
    G=0
    R=0

    for i in range(len(col[0])):
        B += col[0][i][0]

    for i in range(len(col[0])):
        G += col[0][i][1]

    for i in range(len(col[0])):
        R += col[0][i][2]

        return B,G,R

cdef int find_start_index(int B,int G,int R,numpy.ndarray[numpy.uint8_t,ndim = 3] imgL):
    cdef int index,i,flag1,flag2,flag3,diff1,diff2,diff3,b,g,r
    cdef numpy.ndarray[numpy.uint8_t,ndim = 3] col
    index = 0

    for i in range(300):
        col = imgL[i:i+1,0:]
        b,g,r = find_col_BGR(col)

        if i== 0 :
            flag1 = 0
            flag2 = 0
            flag3 = 0
            diff1 = abs(B-b)
            diff2 = abs(G-g)
            diff3 = abs(R-r)
        else:
            if diff1> abs(B-b):
                flag1 = 1
                diff1 = abs(B-b)
            else:
                flag1 = 0

            if diff2> abs(B-b):
                flag2 = 1
                diff2 = abs(G-g)
            else:
                flag2 = 0

            if diff3> abs(B-b):
                flag3 = 1
                diff3 = abs(R-r)
            else:
                flag3 = 0

        if flag1 == 1 and flag2 == 1 and flag3 ==1:
            # print('all matches found:',i)
            # print(diff1)
            # print(diff2)
            # print(diff3)
            break
        elif flag1 == 1 and flag2 == 1 or flag1 == 1 and flag3 == 1 or flag2 == 1 and flag3 == 1:
            # print('two matches found:',i)
            pass

    index = i
    return index





cdef find_closest_match(int row,int column,numpy.ndarray[numpy.uint8_t,ndim = 3] imgL,numpy.ndarray[numpy.uint8_t,ndim = 3] imgR):
    cdef int flag,row_found,column_found,j
    cdef float ADC,ADC_closest
    ADC_closest = 0
    flag = 0
    row_found = row
    column_found = column
    for j in range(column-minDisparity,column-(maxDisparity+1),-1):
        if j<int((windowSize+1)/2):
            break
        else:
            ADC = AD_Census(row,column,imgL,row,j,imgR)

        if flag == 0:
            ADC_closest = ADC
            flag = 1
            row_found = row
            column_found = j
        else:
            if ADC < ADC_closest:# We are taking the assumption that the pixel which is most similar will give the least value of ADC
                ADC_closest = ADC
                row_found = row
                column_found = j

    return row_found,column_found,ADC_closest

cdef float AD_Census(int rowL,int colL,numpy.ndarray[numpy.uint8_t,ndim = 3] imgL,int rowR,int colR,numpy.ndarray[numpy.uint8_t,ndim = 3] imgR):
    cdef float cost_AD,cost_Total
    cdef int cost_Census

    # First is the AD method
    cost_AD = cal_AD(rowL,colL,imgL,rowR,colR,imgR)
    # Second is the Census transform
    cost_Census = cal_Census(rowL,colL,imgL,rowR,colR,imgR)

    cost_Total = (1-math.exp(-(cost_Census/lambda_Census)))+(1-math.exp(-(cost_AD/lambda_AD)))

    return cost_Total


cdef float cal_AD(int rowL,int colL,numpy.ndarray[numpy.uint8_t,ndim = 3] imgL,int rowR,int colR,numpy.ndarray[numpy.uint8_t,ndim = 3] imgR):
    cdef float cost
    cdef int i
    cost = 0
    for i in range(3):
        cost = int(imgL[rowL][colL][i]) - int(imgR[rowR][colR][i])

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
