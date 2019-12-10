# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:27:51 2019

@author: Dhiraj
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import statistics
from sklearn.datasets import load_linnerud
from gaussianNaiveBayes import gnb
from perceptronLearningRule import perc

def linearLeastSquares(linnerudData):
    np.warnings.filterwarnings('ignore')
    #print(linnerudData)
    x = linnerudData['target']
    y = linnerudData['data']
    #print(x.shape[1])
    
    #temp = 1
    plt.figure(figsize=(12,12))
    for i in range(3):
        #print(i)
        if i==0:
            attributeName='weight'
        elif i==1:
            attributeName='waist'
        elif i==2:
            attributeName='heartrate'
        
        for j in range(3):
            if i==0:
                temp=j+1
            #print(temp)
            if j==0:
                outcomeName='chinups'
            elif j==1:
                outcomeName='situps'
            elif j==2:
                outcomeName='jumps'
            
            xVal = x[:,i]
            yVal = y[:,j]
            #print(xVal, yVal)
            a = np.vstack([xVal, np.ones(len(xVal))]).T
            m, c = np.linalg.lstsq(a, yVal)[0]
            #print(m, c)
            #print(attributeName, outcomeName)
            N = len(xVal)
            x_avg = sum(xVal)/N
            y_avg = sum(yVal)/N
            var_x, cov_xy = 0, 0
            for xTmp,yTmp in zip(xVal, yVal):
                #print(xTmp,yTmp)
                varTemp = xTmp - x_avg
                var_x += varTemp**2
                cov_xy += varTemp * (yTmp - y_avg)
            slope = cov_xy / var_x
            intercept = y_avg - slope*x_avg
            #print(slope, intercept)
            
            plt.subplot(3,3,temp)
            plt.plot(xVal, yVal, 'o')#, label='Original data', markersize=0.2)
            plt.plot(xVal, m*xVal + c)#, 'r', label='Fitted line')
            plt.xlabel(attributeName, fontsize=8)
            plt.ylabel(outcomeName, fontsize=8)
            plt.title('slope='+str(truncate(slope))+',intercept='+str(truncate(intercept)))
            plt.tight_layout()
            #plt.legend()
            temp=temp+1
                        
    plt.show()
            
def truncate(n):
    return int(n * 1000) / 1000

def randArr(size):
    arr = np.random.uniform(low=0.0, high=1.0, size=(1000,size))
    return arr

def coeffPearsonMatrix(a,vecSize):
	corrMatrix= np.zeros((1000,1000))
	for x in range(0,1000):
		#print(x)
		meanX = statistics.mean(a[x])
		for y in range(0,1000):
			diff_sum =0
			sq_x=0
			sq_y=0
			meanY = statistics.mean(a[y])
			for k in range(0,vecSize):
				diff_sum = diff_sum+float((a[x][k]-meanX)*(a[y][k]-meanY))
				sq_x = sq_x+math.pow((a[x][k]-meanX),2)
				sq_y = sq_y+math.pow((a[y][k]-meanY),2)
			diff_sum = diff_sum/(math.pow(sq_x,0.5)*math.pow(sq_y,0.5))
			corrMatrix[x][y]  = diff_sum 
	return corrMatrix

def flatten(arrval):
	listval =[]
	for i in np.nditer(arrval):
		if i != 0.:
			listval.append(i)

	return listval

def lowerTriangle(coeff, x, y): 
  
	  
	for i in range(0, x): 
	  
		for j in range(0, y): 
		  
			if (i <= j): 
			  
				coeff[i][j] = 0 
	return coeff


#1(a)
randArr50 = randArr(50)
print(randArr50)    #array with random numbers between 0 & 1 with size 1000*50

#1(b)
randArr10 = randArr(10)

corrMatrix50 = coeffPearsonMatrix(randArr50,50)
corrMatrix10 = coeffPearsonMatrix(randArr10,10)

triangle50 = lowerTriangle(corrMatrix50,corrMatrix50.shape[0],corrMatrix50.shape[1])
triangle10 = lowerTriangle(corrMatrix10,corrMatrix10.shape[0],corrMatrix10.shape[1])

cList50 = flatten(triangle50)

cList10 = flatten(triangle10)


valAbove50 = len([i for i in cList50 if i < -0.75])
valAbove10 = len([i for i in cList10 if i < -0.75])


valBelow50 = len([i for i in cList50 if i > 0.75])
valBelow10 = len([i for i in cList10 if i > 0.75])


totLen50 = len(cList50)
totLen10 = len(cList10)


probAbove50 =  (valAbove50/totLen50)
probBelow50 = (valBelow50/totLen50)

totalProb50 = probAbove50 + probBelow50


probAbove10 =  (valAbove10/totLen10)
probBelow10 =  (valBelow10/totLen10)

totalProb10 = probAbove10 + probBelow10

plt.hist(cList50,normed=True,bins=100)
plt.title('Vector size = 50, P(r-value < -0.75 or r-value > 0.75  ): '+ str(round(totalProb50 * 100,2))+'%')
plt.show()

plt.hist(cList10,normed=True,bins=100)
plt.title('Vector size = 10, P(r-value < -0.75 or r-value > 0.75 ): '+ str(round(totalProb10 * 100, 2))+'%')
plt.show()

#2(a)
data = load_linnerud()
#print(data)

#2(b)
print("\nComputing the linear-least-squares solution... \n")
#linearLeastSquares(data)

#3(a)
print("\nImplementing Gaussian Naive Bayes... \n")
gnb(data)

#3(b)
print("\nImplementing Perceptron learning rule... \n")
perc(data)