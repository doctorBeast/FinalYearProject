'''
Here I will make the Diparity class which will take two images and certain parameters as input
and a Disparity map is given as output when we call the function Disparity.compute()

Here i am listing the parameter which will be passed when the Disparity class is initialized
MaxDisparity
MinDisparity
WindowSize



import cv2
import numpy as np
import math

#This function will take a column as input and return the sum of blue, green and red components
def find_col_BGR(col):
    B = 0
    G = 0
    R = 0

    for i in range(len(col[0])):
        B += col[0][i][0]

    for i in range(len(col[0])):
        G += col[0][i][1]

    for i in range(len(col[0])):
        R += col[0][i][2]

    return B,G,R

# Funtion to find the column i.e. left most closest value to BGR input given
def find_start_index(B,G,R,imgL):
    index = 0

    for i in range(300):
        #This loop is for going column wise to check for the most similar left column
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
            print('all matches found:',i)
            print(diff1)
            print(diff2)
            print(diff3)
            break
        elif flag1 == 1 and flag2 == 1 or flag1 == 1 and flag3 == 1 or flag2 == 1 and flag3 == 1:
            print('two matches found:',i)

    index = i
    return index


imgL = cv2.imread('dstL.jpg',1)
imgR = cv2.imread('dstR.jpg',1)

# Testing code for taking out BGR colors separately
# temp = []
#
# for i in range(len(imgL)):
#     for j in range(len(imgL[0])):
#         temp.append(0)
#         temp.append(imgL[i][j][0])
#         temp.append(0)
#
# temp  = np.array(temp)
# temp = temp.reshape(len(imgL),len(imgL[0]),3)
# temp = temp.astype(np.uint8)

col = imgR[0:1,0:]

B, G, R = find_col_BGR(col)

# call the function to find the left most similar column in the left image
start_index = find_start_index(B,G,R,imgL)

# for looking at the cropped left image from where my search will start
# temp = imgL[0:,start_index:]
# cv2.imshow('Cropped left image',temp)
# cv2.imshow('Right image',imgR)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''


class Disparity:

    def __init__(self,minDisparity = None , maxDisparity = None , windowSize = None, lambda_Census = None,lambda_AD = None):
        self.time = __import__('time')
        self.math = __import__('math')
        self.np = __import__('numpy')
        self.minDisparity = minDisparity
        self.maxDisparity = maxDisparity
        self.windowSize = windowSize
        self.lambda_Census = lambda_Census
        self.lambda_AD = lambda_AD

    def compute(self,imgL , imgR):
        start_time = self.time.time()
        col = imgR[0:1,0:]
        B,G,R = self.find_col_BGR(col)
        start_index = self.find_start_index(B,G,R,imgL)
        temp = imgL[0:,start_index:]
        disp = self.np.zeros((temp.shape))
        for i in range(int((self.windowSize-1)/2),len(temp)-(int((self.windowSize-1)/2))+1):
            print(i,'    ',self.time.time()-start_time)
            for j in range(int((self.windowSize-1)/2),len(temp[0])-int(((self.windowSize-1)/2))+1):
                y , x = self.find_closest_match(i,j+start_index,imgL,imgR)# y is the row element and x is the column element of the closest matched pixel
                print(j+start_index,'#####',self.time.time()-start_time)
                disp[i][j] = j+start_index-x

        return disp


    #This function will take a column as input and return the sum of blue, green and red components
    def find_col_BGR(self,col):
        B = 0
        G = 0
        R = 0

        for i in range(len(col[0])):
            B += col[0][i][0]

        for i in range(len(col[0])):
            G += col[0][i][1]

        for i in range(len(col[0])):
            R += col[0][i][2]

        return B,G,R

    # Funtion to find the column i.e. left most closest value to BGR input given
    def find_start_index(self,B,G,R,imgL):
        index = 0

        for i in range(300):
            #This loop is for going column wise to check for the most similar left column
            col = imgL[i:i+1,0:]
            b,g,r = self.find_col_BGR(col)


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

    # Funtion to find the closest match of the input pixel in its corresponding row with min and max disparity as the bounds of search
    def find_closest_match(self,row,column,imgL,imgR):
        flag = 0
        row_found = row
        column_found = column
        for j in range(column-self.minDisparity,column-(self.maxDisparity+1),-1):
            if j<int((self.windowSize+1)/2):
                break
            else:
                ADC = self.AD_Census(row,column,imgL,row,j,imgR)

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

        return row_found,column_found

    def AD_Census(self,rowL,colL,imgL,rowR,colR,imgR):
        # First is the AD method
        cost_AD = self.cal_AD(rowL,colL,imgL,rowR,colR,imgR)
        # Second is the Census transform
        cost_Census = self.cal_Census(rowL,colL,imgL,rowR,colR,imgR)

        cost_Total = (1-self.math.exp(-(cost_Census/self.lambda_Census)))+(1-self.math.exp(-(cost_AD/self.lambda_AD)))

        return cost_Total

    def cal_AD(self,rowL,colL,imgL,rowR,colR,imgR):
        cost = 0
        for i in range(3):
            cost = int(imgL[rowL][colL][i]) - int(imgR[rowR][colR][i])

        cost /= 3

        return cost

    def cal_Census(self,rowL,colL,imgL,rowR,colR,imgR):
        dist = 0
        cpL = imgL[rowL][colL]
        cpR = imgR[rowR][colR]

        for i in range(int(-(self.windowSize-1)/2),int((self.windowSize-1)/2)):
            for j in range(int(-(self.windowSize-1)/2),int((self.windowSize-1)/2)):
                pL = imgL[rowL+i][colL+j]
                pR = imgR[rowR+i][colR+j]
                for k in range(3):
                    if ((int(cpL[k])-int(pL[k]))*(int(cpR[k])-int(pR[k])))<0:
                        dist+=1

        return dist
