   
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np

import time


def filecreator():
    open("finalprojectdata/digitdata/digitdatatrainer.txt", 'w').close()
    open("finalprojectdata/digitdata/digitdatalabeler.txt", 'w').close() 
    facedatatemp = open("finalprojectdata/digitdata/trainingimages","r")
    facedata = open("finalprojectdata/digitdata/digitdatatrainer.txt","a")
    facedata.write(facedatatemp.read())
    facedatatemp = open("finalprojectdata/digitdata/validationimages","r")
    facedata = open("finalprojectdata/digitdata/digitdatatrainer.txt","a")
    facedata.write(facedatatemp.read())

    facedatatemp = open("finalprojectdata/digitdata/traininglabels","r")
    facedata = open("finalprojectdata/digitdata/digitdatalabeler.txt","a")
    facedata.write(facedatatemp.read())
    facedatatemp = open("finalprojectdata/digitdata/validationlabels","r")
    facedata = open("finalprojectdata/digitdata/digitdatalabeler.txt","a")
    facedata.write(facedatatemp.read())
    open("finalprojectdata/digitdata/digitdatatest.txt", 'w').close()
    open("finalprojectdata/digitdata/digitlabeltestlabeler.txt", 'w').close() 
    
    facedatatemp = open("finalprojectdata/digitdata/testimages","r")
    facedata = open("finalprojectdata/digitdata/digitdatatest.txt","a")
    facedata.write(facedatatemp.read())

    facedatatemp = open("finalprojectdata/digitdata/testlabels","r")
    facedata = open("finalprojectdata/digitdata/digitlabeltestlabeler.txt","a")
    facedata.write(facedatatemp.read())


def bayes_trainer(percent, height,width, percentage):

    train_labels_file = open('finalprojectdata/digitdata/digitdatalabeler.txt', 'r')
    train_labels = train_labels_file.read().split('\n')
    train_labels_file.close()
    train_labels = train_labels[:-1]

    train_images_file = open('finalprojectdata/digitdata/digitdatatrainer.txt', 'r')     
    train_temp_images = train_images_file.read().split('\n')
    train_images_file.close()
    train_images = []

    quantity_of_images = ((len(train_temp_images)//28)*percent)//100
    indeces = list(range(len(train_temp_images)//28))
    np.random.shuffle(indeces)
    indeces = indeces[:quantity_of_images]
    

    for j in indeces:         # For every potential image in your dataset
        image = []
        
        for i in range(28):
            string = ''                           # For every line in that image
            line = train_temp_images[j*28 + i]        # Choose a Line in the image
            for char in line:
                string += char
            image.append(string)



        train_images.append(image) #the right amount of images  for training after running the whole loop 

    train_labels = [train_labels[i] for i in indeces]                                       # the randomized labels for everything

    #generating features

    regions = [0.0000 for _ in range(height*width)] #give it an extremly small constant value

    probabilities = [] #probability x true given digit
    probabilities2 = [] # probability x true given not digit
    probabilitiesbad = [] # probability x false given digit
    probabilitiesbad2 = [] #probability x false given not digit
    for i in range(10):
        probabilities.append([0.0001 for _ in range(height*width)])
        probabilities2.append([0.0001 for _ in range(height*width)])
        probabilitiesbad.append([0.0001 for _ in range(height*width)])
        probabilitiesbad2.append([0.0001 for _ in range(height*width)])


    linecounter =0
    imageindex = -1
    probindex = 0
    arr = [0 for _ in range(10)]
    regions = [0.0 for _ in range(height*width)]
    imagecount = 0
    for image in train_images:  # go through ever image
        regions = [0.0 for _ in range(height*width)] #reset the region features for each image
        currimage = image
        imageindex += 1
        linecounter =0
        probindex = 0
        imagecount = imagecount +1
        for row in currimage:   # go through every row 
            linecounter+=1
            l = ((linecounter)%28)*height//28
            for c in range(len(row)):  # go through every cahracter in teh row
                if row[c]=='#' or row[c] == '+':
                    regions[l*width+c*width//28]+=1         #keep track of how many of these are in each region

        label = train_labels[imageindex]                 
        regions[:] = [(x*height*width) /784 for x in regions]      #percentage pixels being something per  region
        
        probindex = 0
        arr[label]+=1
        for region in regions:
            if region >= percentage: #if >percentage elemeents in the region are + or #
                for i in range(10):
                    probabilities[i][probindex] = probabilities[i][probindex] +1  # state given true and face =  true, increment by 1
                
            elif region <percentage:  
                for i in range(10):
                     probabilities2[i][probindex] = probabilities2[i][probindex] +1  # state given false and digit =  true, increment by 1
            probindex =  probindex+1

    for i in range(10):
        probabilities[i][:] = [(probability) /arr[i] for probability in probabilities[i]]  #probability of zero given regions
        probabilities2[i][:] = [(probability) /arr[i] for probability in probabilities2[i]]            # probability of zero given not regions
        probabilitiesbad[i][:] = [(probability) /(imagecount-arr[i]) for probability in probabilitiesbad[i]]     #probabilty of not zero given regions
        probabilitiesbad2[i][:] = [(probability) /(imagecount-arr[i]) for probability in probabilitiesbad2[i]]    # probability of not zero given not regions

   
    accuracy = bayes_tester(probabilities, probabilities2, probabilitiesbad, probabilitiesbad2, height, width, percentage)

    return accuracy







def bayes_tester(probabilities, probabilities2,probabilitiesbad, probabilitiesbad2, height,width, percentage):
    # prepare the testing images and data to be input to the model

    test_labels_file = open('finalprojectdata/digitdata/digitlabeltestlabeler.txt')
    test_labels = test_labels_file.read().split('\n')
    test_labels_file.close()
    test_labels = test_labels[:-1]

    test_images_file = open('finalprojectdata/digitdata/digitdatatest.txt')
    test_temp_images = test_images_file.read().split('\n')
    test_images_file.close()
    test_images = []

    for j in range(len(test_temp_images)//28):         # For every potential image in your dataset
        image = []
        
        for i in range(28):
            string = ''                           # For every line in that image
            line = test_temp_images[j*28 + i]        # Choose a Line in the image
            for char in line:
                string += char
            image.append(string)

        test_images.append(image) #the right amount of images  for training after running the whole loop 

    regions = [0.0 for _ in range(height*width)]
    linecounter = 0
    imageindex = -1

    digitprobabilities = [0.0 for _ in range(10)]
    probindex = 0
    total =0
    currmax = 0
    bad = 0
    guess = 0
    for image in test_images:  # go through ever image
        regions = [0.0 for _ in range(height*width)] #reset the region features for each image
        currimage = image
        imageindex += 1
        linecounter =0
        probindex = 0
        total = total + 1

        for row in currimage:   # go through every row 
            linecounter+=1
            l = ((linecounter)%28)*height//28
            for c in range(len(row)):  # go through every cahracter in teh row
                if row[c]=='#' or row[c] == '+':
                    regions[l*width+c*width//28]+=1         #keep track of how many of these are in each region
        label = test_labels[imageindex]                 
        regions[:] = [(x*height*width) /784 for x in regions]      #get region values
        
    # start working from here t
        for region in regions:
            if region >= percentage:
                for  i in range(10):
                    if digitprobabilities[i] == 0:
                        digitprobabilities[i] = probabilities[i][probindex]  # probability of face given x
                    else:
                        digitprobabilities[i] = digitprobabilities[i] * probabilities[i][probindex] # state given true and face =  true, adjust probability
            else:
                for  i in range(10):
                    if digitprobabilities[i] == 0:
                        digitprobabilities[i] = probabilities2[i][probindex]  # probability of face given x
                    else:
                        digitprobabilities[i] = digitprobabilities[i] * probabilities2[i][probindex] # state given true and face =  true, adjust probability
            probindex =  probindex+1
        
        for  i in range(len(digitprobabilities)): #select the digit that is most likely 
            if digitprobabilities[i] > currmax:
                currmax = digitprobabilities[i]
                guess = i
        #print(digitprobabilities)

        guess = str(guess)
        if guess != label:
            bad = bad+1
        digitprobabilities = [0.0 for _ in range(10)]
        currmax = 0
    return (1-(bad*1.0/total))*100.0



def main():

    filecreator()

    for i in range(10):
        tic = time.time()
        accuracy = bayes_trainer((10*(1+i)),8,7, .2) #use bayes_trainer(10,7,7, .1) = use 10% data, 7 by 7 regions, if 10% of characters are  #  return true state
        toc = time.time()
        print("Percentage data: %s Accuracy: %s Execution time: %s"  %(10*(i+1), round(accuracy,2), round((toc-tic),2)))
    return


main()
