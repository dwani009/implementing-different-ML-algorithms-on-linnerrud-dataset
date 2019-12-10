# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:32:35 2019

@author: Dhiraj
"""

import os
import numpy as np
import matplotlib.pyplot as plot
from numpy import median, arange
from sklearn.datasets import load_linnerud


class Perceptron(object):

   def __init__(self, no_of_inputs, threshold=1000, learning_rate=0.01):
       self.threshold = threshold
       self.learning_rate = learning_rate
       self.weights = np.zeros(no_of_inputs + 1)

   def predict(self, inputs):
       summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
       if summation > 0:
         activation = 1
       else:
         activation = 0           
       return activation

   def train(self, training_inputs, labels):
       for _ in range(self.threshold):
           for inputs, label in zip(training_inputs, labels):
               prediction = self.predict(inputs)
               self.weights[1:] += self.learning_rate * (label - prediction) * inputs
               self.weights[0] += self.learning_rate * (label - prediction)


def prepareData(dataset):
   training_inputs = []

   for i in range(len(dataset)):
       training_inputs.append(np.array(dataset[i]))

   return training_inputs

def prepareDatawithMedian(dataset):

   target = dataset.get('target')
   data = dataset.get('data')
   separated = []

   medianArray = []
   for i in range(len(data)):
       value = int(data[i][0])
       medianArray.append(int(value))

   medianvalue = median(medianArray)

   #print("median")
   #print(medianvalue)

   for i in range(len(target)):
       vectorTarget = 0

       vectorData = data[i]

       value = int(vectorData[0])

       if value > medianvalue:
           vectorTarget = 0
       else:
           vectorTarget = 1

       separated.append(vectorTarget)


   return separated

def perc(dataset):
    dataN = dataset.get('data')
    target = dataset.get('target')
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
    
    for i in dataN:
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

    #pathName = os.path.dirname(os.path.abspath(__file__))
    #myfile = open(pathName+'\perceptron_results.txt', 'w')
    myfile.write('\nMethod 2(prediction values):\n\n')
    myfile.write('Probability values output by peceptron:\n')
    myfile.write('Instance     Probability Value\n')
    print('\nMethod 2(prediction values):\n\n')
    print('Probability values output by peceptron:')
    print('Instance     Probability Value\n')
    data = dataset.get('target')
    
    data = prepareData(data)
    
    appeneded = prepareDatawithMedian(dataset)
    
    appeneded = np.array(appeneded)
    
    percep  = Perceptron(3,1000,0.01)
    percep.train(data,appeneded)
    
    
    inst = 1;
    for i in range(len(data)):
       predict = percep.predict(data[i])
       if inst>9:
           myfile.write('   '+str(inst)+'               '+str(predict)+'\n')
           print('   '+str(inst)+'               '+str(predict)+'\n')
       else:
           myfile.write('   '+str(inst)+'               '+' '+str(predict)+'\n')
           print('   '+str(inst)+'               '+' '+str(predict)+'\n')
       inst = inst+1

    print("\nProbability values appended to perceptorn_result.txt file")
