# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:09:57 2019

@author: Dhiraj
"""

from sklearn.datasets import load_linnerud
import numpy as np
from numpy import arange
import matplotlib.pyplot as plot 
import os


dataset = load_linnerud().data
target = load_linnerud().target
pathName = os.path.dirname(os.path.abspath(__file__))
print(pathName)
myfile = open(pathName+'\perceptron_results.txt', 'w')
myfile.write('Method 1(as per lecture notes example):\n\n')
myfile.write('Probability values output by peceptron:\n')
myfile.write('Instance     Probability Value\n')
print('Method 1(as per lecture notes example):\n')
print('Probability values output by peceptron:')
print('Instance     Probability Value\n')
    
chinup = []

for i in dataset:
	chinup.append(i[0])

median = np.median(chinup)

binary = []

for j in chinup:

	#print(j)
	if j > median:
		binary.append(0)

	else:
		binary.append(1)

weights = np.zeros([3,1])

iteration = 1000

for i in arange(0,iteration):
	counter = 0
	converged = True

	for row_val in target:
		
		pred_val = np.dot(row_val,weights)
		if pred_val < 0:

			predicted = 0

		else:
		
			predicted = 1

		if predicted != binary[counter]:

			converged = False

			if binary[counter] == 0 :

				weights = weights - np.expand_dims(row_val,1)

			else:

				weights = weights + np.expand_dims(row_val,1)

		counter = counter + 1

	if converged == True:
		print("Loop broken")
		break

final_pred = np.dot(target,weights)
valList = final_pred.tolist()

inst=1
for val in valList:
    #print(str(val).strip('[]'))
    val = str(val).strip('[]')
    if inst>9:
        myfile.write('   '+str(inst)+'               '+str(val)+'\n')
        print('   '+str(inst)+'               '+str(val)+'\n')
    else:
       myfile.write('   '+str(inst)+'               '+' '+str(val)+'\n')
       print('   '+str(inst)+'               '+' '+str(val)+'\n')
    inst = inst+1

plot.plot(final_pred,'bo'); 
plot.plot([0,20],[0,0])
plot.show()