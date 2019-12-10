# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 00:07:27 2019

@author: Dhiraj
"""

import numpy as np
import math
import os

#calculate the posterior probability
def pXgivenY(x,mean_y,variance_y):
    probability = (1/(np.sqrt(2*np.pi*variance_y))) * np.exp(-((x-mean_y)**2)/(2*variance_y))
    return probability[0]

#calculate the variance
def calVariance(weight,mean):
    var_sum  = 0
    for x in weight:
        var_sum = var_sum +math.pow(abs(x-mean),2) 
    return var_sum/(weight.size-1)

def gnb(linnerud):
    weightY = []
    waistY = []
    pulseY = []
    weightN = []
    waistN = []
    pulseN = []

    x = linnerud['target']
    y = linnerud['data']
    X= np.column_stack((x,y[:,0]))
    median = np.median(X[:,3])
    for i in range(0,20):
        #print(X[i:i+1])
        if(X[i:i+1,3]>median):
            X[i:i+1,3]=0
        else:
            X[i:i+1,3]=1
        print(X)
    yes = np.count_nonzero(X[:,3]==1)
    print(yes)
    total = np.size(X[:,3])
    no = total - yes
    yes_probab = yes/total
    no_probab  = no/total
    for i in range(0,20):
        if X[i:i+1,3]==1:
            weightY.append(X[i:i+1,0])
            waistY.append(X[i:i+1,1])
            pulseY.append(X[i:i+1,2])
        else:
            weightN.append(X[i:i+1,0])
            waistN.append(X[i:i+1,1])
            pulseN.append(X[i:i+1,2])
    weightY= np.array((weightY))
    waistY = np.array((waistY))
    pulseY = np.array((pulseY))
    weightN = np.array((weightN))
    waistN = np.array((waistN))
    pulseN = np.array((pulseN))


    weight_no_mean = np.mean(weightN)
    weight_no_var = calVariance (weightN,weight_no_mean)
    waist_no_mean = np.mean(waistN)
    waist_no_var = calVariance(waistN,waist_no_mean)
    pulse_no_mean = np.mean(pulseN)
    pulse_no_var = calVariance(pulseN,pulse_no_mean)
    weight_yes_mean = np.mean(weightY)
    weight_yes_var = calVariance(weightY,weight_yes_mean)
    waist_yes_mean=np.mean(waistY)
    waist_yes_var= calVariance(waistY,waist_yes_mean)
    pulse_yes_mean = np.mean(pulseY)
    pulse_yes_var = calVariance(pulseY,pulse_yes_mean)
    probab = 0 

    pathName = os.path.dirname(os.path.abspath(__file__))
    myfile = open(pathName+'\gnb_result.txt', 'w')
    myfile.write('Probability values output by Gaussian Naive Bayes:\n\n')
    print('Probability values output by Gaussian Naive Bayes:')
    
    for i in range(0,20):
        probab = 0 
        probab_yes =  yes_probab*pXgivenY(X[i:i+1,0],weight_yes_mean,weight_yes_var)*pXgivenY(X[i:i+1,1],waist_yes_mean,waist_yes_var)*pXgivenY(X[i:i+1,2],pulse_yes_mean,pulse_yes_var)
        probab_no = no_probab*pXgivenY(X[i:i+1,0],weight_no_mean,weight_no_var)*pXgivenY(X[i:i+1,1],waist_no_mean,waist_no_var)*pXgivenY(X[i:i+1,2],pulse_no_mean,pulse_no_var)
        probab = probab_yes/(probab_yes+probab_no)
        output = "Probability value for weight %d, waist %d and heartrate %d is {:.7f}".format(probab)%(X[i:i+1,0],X[i:i+1,1],X[i:i+1,2])
        print(output)
        myfile.write(output+"\n")
    print("\nProbability values appended to gnb_result.txt file\n")
    myfile.close()