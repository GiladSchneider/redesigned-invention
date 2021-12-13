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
    onecount = 0
    zerocount = 0
    twocount = 0
    threecount = 0
    fourcount = 0
    fivecount = 0
    sixcount = 0
    sevencount = 0
    eightcount = 0
    ninecount = 0
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
        if(label == '0'):  #if face check the regions and update the probabilities
            zerocount = zerocount +1
            for region in regions:
                if region >= percentage: #if >percentage elemeents in the region are + or #
                    probabilities[0][probindex] = probabilities[0][probindex] +1  # state given true and face =  true, increment by 1
                    probabilitiesbad[1][probindex] = probabilitiesbad[1][probindex] +1  #since this mean the other ones were not true with region being true alter them
                    probabilitiesbad[2][probindex] = probabilitiesbad[2][probindex] +1 
                    probabilitiesbad[3][probindex] = probabilitiesbad[3][probindex] +1 
                    probabilitiesbad[4][probindex] = probabilitiesbad[4][probindex] +1 
                    probabilitiesbad[5][probindex] = probabilitiesbad[5][probindex] +1 
                    probabilitiesbad[6][probindex] = probabilitiesbad[6][probindex] +1 
                    probabilitiesbad[7][probindex] = probabilitiesbad[7][probindex] +1 
                    probabilitiesbad[8][probindex] = probabilitiesbad[8][probindex] +1 
                    probabilitiesbad[9][probindex] = probabilitiesbad[9][probindex] +1 
                elif region <percentage:                                       # state given false and face = true
                    probabilities2[0][probindex] = probabilities2[0][probindex] +1  # state given false and digit =  true, increment by 1
                    probabilitiesbad2[1][probindex] = probabilitiesbad2[1][probindex] +1 
                    probabilitiesbad2[2][probindex] = probabilitiesbad2[2][probindex] +1 #since this mean the other ones were not true with region being false alter them
                    probabilitiesbad2[3][probindex] = probabilitiesbad2[3][probindex] +1 
                    probabilitiesbad2[4][probindex] = probabilitiesbad2[4][probindex] +1 
                    probabilitiesbad2[5][probindex] = probabilitiesbad2[5][probindex] +1 
                    probabilitiesbad2[6][probindex] = probabilitiesbad2[6][probindex] +1 
                    probabilitiesbad2[7][probindex] = probabilitiesbad2[7][probindex] +1 
                    probabilitiesbad2[8][probindex] = probabilitiesbad2[8][probindex] +1 
                    probabilitiesbad2[9][probindex] = probabilitiesbad2[9][probindex] +1    # state given false and face = true
                probindex =  probindex+1
        elif(label == '1'):
            onecount = onecount +1
            for region in regions:
                if region >= percentage: #if >percentage elemeents in the region are + or #
                    probabilitiesbad[0][probindex] = probabilitiesbad[0][probindex] +1  # state given true and face =  true, increment by 1
                    probabilities[1][probindex] =    probabilities[1][probindex] +1  #since this mean the other ones were not true with region being true alter them
                    probabilitiesbad[2][probindex] = probabilitiesbad[2][probindex] +1 
                    probabilitiesbad[3][probindex] = probabilitiesbad[3][probindex] +1 
                    probabilitiesbad[4][probindex] = probabilitiesbad[4][probindex] +1 
                    probabilitiesbad[5][probindex] = probabilitiesbad[5][probindex] +1 
                    probabilitiesbad[6][probindex] = probabilitiesbad[6][probindex] +1 
                    probabilitiesbad[7][probindex] = probabilitiesbad[7][probindex] +1 
                    probabilitiesbad[8][probindex] = probabilitiesbad[8][probindex] +1 
                    probabilitiesbad[9][probindex] = probabilitiesbad[9][probindex] +1 
                elif region <percentage:                                       # state given false and face = true
                    probabilitiesbad2[0][probindex] = probabilitiesbad2[0][probindex] +1  # state given false and digit =  true, increment by 1
                    probabilities2[1][probindex] = probabilities2[1][probindex] +1 
                    probabilitiesbad2[2][probindex] = probabilitiesbad2[2][probindex] +1 #since this mean the other ones were not true with region being false alter them
                    probabilitiesbad2[3][probindex] = probabilitiesbad2[3][probindex] +1 
                    probabilitiesbad2[4][probindex] = probabilitiesbad2[4][probindex] +1 
                    probabilitiesbad2[5][probindex] = probabilitiesbad2[5][probindex] +1 
                    probabilitiesbad2[6][probindex] = probabilitiesbad2[6][probindex] +1 
                    probabilitiesbad2[7][probindex] = probabilitiesbad2[7][probindex] +1 
                    probabilitiesbad2[8][probindex] = probabilitiesbad2[8][probindex] +1 
                    probabilitiesbad2[9][probindex] = probabilitiesbad2[9][probindex] +1    # state given false and face = true
                probindex =  probindex+1
        elif(label == '2'):
            twocount = twocount +1
            for region in regions:
                if region >= percentage: #if >percentage elemeents in the region are + or #
                    probabilitiesbad[0][probindex] = probabilitiesbad[0][probindex] +1  # state given true and face =  true, increment by 1
                    probabilitiesbad[1][probindex] = probabilitiesbad[1][probindex] +1  #since this mean the other ones were not true with region being true alter them
                    probabilities[2][probindex] = probabilities[2][probindex] +1 
                    probabilitiesbad[3][probindex] = probabilitiesbad[3][probindex] +1 
                    probabilitiesbad[4][probindex] = probabilitiesbad[4][probindex] +1 
                    probabilitiesbad[5][probindex] = probabilitiesbad[5][probindex] +1 
                    probabilitiesbad[6][probindex] = probabilitiesbad[6][probindex] +1 
                    probabilitiesbad[7][probindex] = probabilitiesbad[7][probindex] +1 
                    probabilitiesbad[8][probindex] = probabilitiesbad[8][probindex] +1 
                    probabilitiesbad[9][probindex] = probabilitiesbad[9][probindex] +1 
                elif region <percentage:                                       # state given false and face = true
                    probabilitiesbad2[0][probindex] = probabilitiesbad2[0][probindex] +1  # state given false and digit =  true, increment by 1
                    probabilitiesbad2[1][probindex] = probabilitiesbad2[1][probindex] +1 
                    probabilities2[2][probindex] = probabilities2[2][probindex] +1 #since this mean the other ones were not true with region being false alter them
                    probabilitiesbad2[3][probindex] = probabilitiesbad2[3][probindex] +1 
                    probabilitiesbad2[4][probindex] = probabilitiesbad2[4][probindex] +1 
                    probabilitiesbad2[5][probindex] = probabilitiesbad2[5][probindex] +1 
                    probabilitiesbad2[6][probindex] = probabilitiesbad2[6][probindex] +1 
                    probabilitiesbad2[7][probindex] = probabilitiesbad2[7][probindex] +1 
                    probabilitiesbad2[8][probindex] = probabilitiesbad2[8][probindex] +1 
                    probabilitiesbad2[9][probindex] = probabilitiesbad2[9][probindex] +1    # state given false and face = true
                probindex =  probindex+1
        elif(label == '4'):
            fourcount = fourcount +1
            for region in regions:
                if region >= percentage: #if >percentage elemeents in the region are + or #
                    probabilitiesbad[0][probindex] = probabilitiesbad[0][probindex] +1  # state given true and face =  true, increment by 1
                    probabilitiesbad[1][probindex] = probabilitiesbad[1][probindex] +1  #since this mean the other ones were not true with region being true alter them
                    probabilitiesbad[2][probindex] = probabilitiesbad[2][probindex] +1 
                    probabilitiesbad[3][probindex] = probabilitiesbad[3][probindex] +1 
                    probabilities[4][probindex] = probabilities[4][probindex] +1 
                    probabilitiesbad[5][probindex] = probabilitiesbad[5][probindex] +1 
                    probabilitiesbad[6][probindex] = probabilitiesbad[6][probindex] +1 
                    probabilitiesbad[7][probindex] = probabilitiesbad[7][probindex] +1 
                    probabilitiesbad[8][probindex] = probabilitiesbad[8][probindex] +1 
                    probabilitiesbad[9][probindex] = probabilitiesbad[9][probindex] +1 
                elif region <percentage:                                       # state given false and face = true
                    probabilitiesbad2[0][probindex] = probabilitiesbad2[0][probindex] +1  # state given false and digit =  true, increment by 1
                    probabilitiesbad2[1][probindex] = probabilitiesbad2[1][probindex] +1 
                    probabilitiesbad2[2][probindex] = probabilitiesbad2[2][probindex] +1 #since this mean the other ones were not true with region being false alter them
                    probabilitiesbad2[3][probindex] = probabilitiesbad2[3][probindex] +1 
                    probabilities2[4][probindex] = probabilities2[4][probindex] +1 
                    probabilitiesbad2[5][probindex] = probabilitiesbad2[5][probindex] +1 
                    probabilitiesbad2[6][probindex] = probabilitiesbad2[6][probindex] +1 
                    probabilitiesbad2[7][probindex] = probabilitiesbad2[7][probindex] +1 
                    probabilitiesbad2[8][probindex] = probabilitiesbad2[8][probindex] +1 
                    probabilitiesbad2[9][probindex] = probabilitiesbad2[9][probindex] +1    # state given false and face = true
                probindex =  probindex+1

        elif(label == '3'):
            threecount = threecount +1
            for region in regions:
                if region >= percentage: #if >percentage elemeents in the region are + or #
                    probabilitiesbad[0][probindex] = probabilitiesbad[0][probindex] +1  # state given true and face =  true, increment by 1
                    probabilitiesbad[1][probindex] = probabilitiesbad[1][probindex] +1  #since this mean the other ones were not true with region being true alter them
                    probabilitiesbad[2][probindex] = probabilitiesbad[2][probindex] +1 
                    probabilities[3][probindex] = probabilities[3][probindex] +1 
                    probabilitiesbad[4][probindex] = probabilitiesbad[4][probindex] +1 
                    probabilitiesbad[5][probindex] = probabilitiesbad[5][probindex] +1 
                    probabilitiesbad[6][probindex] = probabilitiesbad[6][probindex] +1 
                    probabilitiesbad[7][probindex] = probabilitiesbad[7][probindex] +1 
                    probabilitiesbad[8][probindex] = probabilitiesbad[8][probindex] +1 
                    probabilitiesbad[9][probindex] = probabilitiesbad[9][probindex] +1 
                elif region <percentage:                                       # state given false and face = true
                    probabilitiesbad2[0][probindex] = probabilitiesbad2[0][probindex] +1  # state given false and digit =  true, increment by 1
                    probabilitiesbad2[1][probindex] = probabilitiesbad2[1][probindex] +1 
                    probabilitiesbad2[2][probindex] = probabilitiesbad2[2][probindex] +1 #since this mean the other ones were not true with region being false alter them
                    probabilities2[3][probindex] = probabilities2[3][probindex] +1 
                    probabilitiesbad2[4][probindex] = probabilitiesbad2[4][probindex] +1 
                    probabilitiesbad2[5][probindex] = probabilitiesbad2[5][probindex] +1 
                    probabilitiesbad2[6][probindex] = probabilitiesbad2[6][probindex] +1 
                    probabilitiesbad2[7][probindex] = probabilitiesbad2[7][probindex] +1 
                    probabilitiesbad2[8][probindex] = probabilitiesbad2[8][probindex] +1 
                    probabilitiesbad2[9][probindex] = probabilitiesbad2[9][probindex] +1    # state given false and face = true
                probindex =  probindex+1

        elif(label == '5'):
            fivecount = fivecount +1
            for region in regions:
                if region >= percentage: #if >percentage elemeents in the region are + or #
                    probabilitiesbad[0][probindex] = probabilitiesbad[0][probindex] +1  # state given true and face =  true, increment by 1
                    probabilitiesbad[1][probindex] = probabilitiesbad[1][probindex] +1  #since this mean the other ones were not true with region being true alter them
                    probabilitiesbad[2][probindex] = probabilitiesbad[2][probindex] +1 
                    probabilitiesbad[3][probindex] = probabilitiesbad[3][probindex] +1 
                    probabilitiesbad[4][probindex] = probabilitiesbad[4][probindex] +1 
                    probabilities[5][probindex] = probabilities[5][probindex] +1 
                    probabilitiesbad[6][probindex] = probabilitiesbad[6][probindex] +1 
                    probabilitiesbad[7][probindex] = probabilitiesbad[7][probindex] +1 
                    probabilitiesbad[8][probindex] = probabilitiesbad[8][probindex] +1 
                    probabilitiesbad[9][probindex] = probabilitiesbad[9][probindex] +1 
                elif region <percentage:                                       # state given false and face = true
                    probabilitiesbad2[0][probindex] = probabilitiesbad2[0][probindex] +1  # state given false and digit =  true, increment by 1
                    probabilitiesbad2[1][probindex] = probabilitiesbad2[1][probindex] +1 
                    probabilitiesbad2[2][probindex] = probabilitiesbad2[2][probindex] +1 #since this mean the other ones were not true with region being false alter them
                    probabilitiesbad2[3][probindex] = probabilitiesbad2[3][probindex] +1 
                    probabilitiesbad2[4][probindex] = probabilitiesbad2[4][probindex] +1 
                    probabilities2[5][probindex] = probabilities2[5][probindex] +1 
                    probabilitiesbad2[6][probindex] = probabilitiesbad2[6][probindex] +1 
                    probabilitiesbad2[7][probindex] = probabilitiesbad2[7][probindex] +1 
                    probabilitiesbad2[8][probindex] = probabilitiesbad2[8][probindex] +1 
                    probabilitiesbad2[9][probindex] = probabilitiesbad2[9][probindex] +1    # state given false and face = true
                probindex =  probindex+1

        elif(label == '6'):
            sixcount = sixcount +1
            for region in regions:
                if region >= percentage: #if >percentage elemeents in the region are + or #
                    probabilitiesbad[0][probindex] = probabilitiesbad[0][probindex] +1  # state given true and face =  true, increment by 1
                    probabilitiesbad[1][probindex] = probabilitiesbad[1][probindex] +1  #since this mean the other ones were not true with region being true alter them
                    probabilitiesbad[2][probindex] = probabilitiesbad[2][probindex] +1 
                    probabilitiesbad[3][probindex] = probabilitiesbad[3][probindex] +1 
                    probabilitiesbad[4][probindex] = probabilitiesbad[4][probindex] +1 
                    probabilitiesbad[5][probindex] = probabilitiesbad[5][probindex] +1 
                    probabilities[6][probindex] = probabilities[6][probindex] +1 
                    probabilitiesbad[7][probindex] = probabilitiesbad[7][probindex] +1 
                    probabilitiesbad[8][probindex] = probabilitiesbad[8][probindex] +1 
                    probabilitiesbad[9][probindex] = probabilitiesbad[9][probindex] +1 
                elif region <percentage:                                       # state given false and face = true
                    probabilitiesbad2[0][probindex] = probabilitiesbad2[0][probindex] +1  # state given false and digit =  true, increment by 1
                    probabilitiesbad2[1][probindex] = probabilitiesbad2[1][probindex] +1 
                    probabilitiesbad2[2][probindex] = probabilitiesbad2[2][probindex] +1 #since this mean the other ones were not true with region being false alter them
                    probabilitiesbad2[3][probindex] = probabilitiesbad2[3][probindex] +1 
                    probabilitiesbad2[4][probindex] = probabilitiesbad2[4][probindex] +1 
                    probabilitiesbad2[5][probindex] = probabilitiesbad2[5][probindex] +1 
                    probabilities2[6][probindex] = probabilities2[6][probindex] +1 
                    probabilitiesbad2[7][probindex] = probabilitiesbad2[7][probindex] +1 
                    probabilitiesbad2[8][probindex] = probabilitiesbad2[8][probindex] +1 
                    probabilitiesbad2[9][probindex] = probabilitiesbad2[9][probindex] +1    # state given false and face = true
                probindex =  probindex+1

        elif(label == '7'):
            sevencount = sevencount +1
            for region in regions:
                if region >= percentage: #if >percentage elemeents in the region are + or #
                    probabilitiesbad[0][probindex] = probabilitiesbad[0][probindex] +1  # state given true and face =  true, increment by 1
                    probabilitiesbad[1][probindex] = probabilitiesbad[1][probindex] +1  #since this mean the other ones were not true with region being true alter them
                    probabilitiesbad[2][probindex] = probabilitiesbad[2][probindex] +1 
                    probabilitiesbad[3][probindex] = probabilitiesbad[3][probindex] +1 
                    probabilitiesbad[4][probindex] = probabilitiesbad[4][probindex] +1 
                    probabilitiesbad[5][probindex] = probabilitiesbad[5][probindex] +1 
                    probabilitiesbad[6][probindex] = probabilitiesbad[6][probindex] +1 
                    probabilities[7][probindex] = probabilities[7][probindex] +1 
                    probabilitiesbad[8][probindex] = probabilitiesbad[8][probindex] +1 
                    probabilitiesbad[9][probindex] = probabilitiesbad[9][probindex] +1 
                elif region <percentage:                                       # state given false and face = true
                    probabilitiesbad2[0][probindex] = probabilitiesbad2[0][probindex] +1  # state given false and digit =  true, increment by 1
                    probabilitiesbad2[1][probindex] = probabilitiesbad2[1][probindex] +1 
                    probabilitiesbad2[2][probindex] = probabilitiesbad2[2][probindex] +1 #since this mean the other ones were not true with region being false alter them
                    probabilitiesbad2[3][probindex] = probabilitiesbad2[3][probindex] +1 
                    probabilitiesbad2[4][probindex] = probabilitiesbad2[4][probindex] +1 
                    probabilitiesbad2[5][probindex] = probabilitiesbad2[5][probindex] +1 
                    probabilitiesbad2[6][probindex] = probabilitiesbad2[6][probindex] +1 
                    probabilities2[7][probindex] = probabilities2[7][probindex] +1 
                    probabilitiesbad2[8][probindex] = probabilitiesbad2[8][probindex] +1 
                    probabilitiesbad2[9][probindex] = probabilitiesbad2[9][probindex] +1    # state given false and face = true
                probindex =  probindex+1

        elif(label == '8'):
            eightcount = eightcount +1
            for region in regions:
                if region >= percentage: #if >percentage elemeents in the region are + or #
                    probabilitiesbad[0][probindex] = probabilitiesbad[0][probindex] +1  # state given true and face =  true, increment by 1
                    probabilitiesbad[1][probindex] = probabilitiesbad[1][probindex] +1  #since this mean the other ones were not true with region being true alter them
                    probabilitiesbad[2][probindex] = probabilitiesbad[2][probindex] +1 
                    probabilitiesbad[3][probindex] = probabilitiesbad[3][probindex] +1 
                    probabilitiesbad[4][probindex] = probabilitiesbad[4][probindex] +1 
                    probabilitiesbad[5][probindex] = probabilitiesbad[5][probindex] +1 
                    probabilitiesbad[6][probindex] = probabilitiesbad[6][probindex] +1 
                    probabilitiesbad[7][probindex] = probabilitiesbad[7][probindex] +1 
                    probabilities[8][probindex] = probabilities[8][probindex] +1 
                    probabilitiesbad[9][probindex] = probabilitiesbad[9][probindex] +1 
                elif region <percentage:                                       # state given false and face = true
                    probabilitiesbad2[0][probindex] = probabilitiesbad2[0][probindex] +1  # state given false and digit =  true, increment by 1
                    probabilitiesbad2[1][probindex] = probabilitiesbad2[1][probindex] +1 
                    probabilitiesbad2[2][probindex] = probabilitiesbad2[2][probindex] +1 #since this mean the other ones were not true with region being false alter them
                    probabilitiesbad2[3][probindex] = probabilitiesbad2[3][probindex] +1 
                    probabilitiesbad2[4][probindex] = probabilitiesbad2[4][probindex] +1 
                    probabilitiesbad2[5][probindex] = probabilitiesbad2[5][probindex] +1 
                    probabilitiesbad2[6][probindex] = probabilitiesbad2[6][probindex] +1 
                    probabilitiesbad2[7][probindex] = probabilitiesbad2[7][probindex] +1 
                    probabilities2[8][probindex] = probabilities2[8][probindex] +1 
                    probabilitiesbad2[9][probindex] = probabilitiesbad2[9][probindex] +1    # state given false and face = true
                probindex =  probindex+1

        elif(label == '9'):
            ninecount = ninecount +1
            for region in regions:
                if region >= percentage: #if >percentage elemeents in the region are + or #
                    probabilitiesbad[0][probindex] = probabilitiesbad[0][probindex] +1  # state given true and face =  true, increment by 1
                    probabilitiesbad[1][probindex] = probabilitiesbad[1][probindex] +1  #since this mean the other ones were not true with region being true alter them
                    probabilitiesbad[2][probindex] = probabilitiesbad[2][probindex] +1 
                    probabilitiesbad[3][probindex] = probabilitiesbad[3][probindex] +1 
                    probabilitiesbad[4][probindex] = probabilitiesbad[4][probindex] +1 
                    probabilitiesbad[5][probindex] = probabilitiesbad[5][probindex] +1 
                    probabilitiesbad[6][probindex] = probabilitiesbad[6][probindex] +1 
                    probabilitiesbad[7][probindex] = probabilitiesbad[7][probindex] +1 
                    probabilitiesbad[8][probindex] = probabilitiesbad[8][probindex] +1 
                    probabilities[9][probindex] = probabilities[9][probindex] +1 
                elif region <percentage:                                       # state given false and face = true
                    probabilitiesbad2[0][probindex] = probabilitiesbad2[0][probindex] +1  # state given false and digit =  true, increment by 1
                    probabilitiesbad2[1][probindex] = probabilitiesbad2[1][probindex] +1 
                    probabilitiesbad2[2][probindex] = probabilitiesbad2[2][probindex] +1 #since this mean the other ones were not true with region being false alter them
                    probabilitiesbad2[3][probindex] = probabilitiesbad2[3][probindex] +1 
                    probabilitiesbad2[4][probindex] = probabilitiesbad2[4][probindex] +1 
                    probabilitiesbad2[5][probindex] = probabilitiesbad2[5][probindex] +1 
                    probabilitiesbad2[6][probindex] = probabilitiesbad2[6][probindex] +1 
                    probabilitiesbad2[7][probindex] = probabilitiesbad2[7][probindex] +1 
                    probabilitiesbad2[8][probindex] = probabilitiesbad2[8][probindex] +1 
                    probabilities2[9][probindex] = probabilities2[9][probindex] +1    # state given false and face = true
                probindex =  probindex+1
        
    #getting the probabilities of digit given region
    probabilities[0][:] = [(probability) /zerocount for probability in probabilities[0]]  #probability of zero given regions
    probabilities2[0][:] = [(probability) /zerocount for probability in probabilities2[0]]            # probability of zero given not regions
    probabilitiesbad[0][:] = [(probability) /(imagecount-zerocount) for probability in probabilitiesbad[0]]     #probabilty of not zero given regions
    probabilitiesbad2[0][:] = [(probability) /(imagecount-zerocount) for probability in probabilitiesbad2[0]]    # probability of not zero given not regions

    probabilities[1][:] = [(probability) /onecount for probability in probabilities[1]]  #probability of zero given regions
    probabilities2[1][:] = [(probability) /onecount for probability in probabilities2[1]]            # probability of zero given not regions
    probabilitiesbad[1][:] = [(probability) /(imagecount-onecount) for probability in probabilitiesbad[1]]     #probabilty of not zero given regions
    probabilitiesbad2[1][:] = [(probability) /(imagecount-onecount) for probability in probabilitiesbad2[1]]    # probability of not zero given not regions

    probabilities[2][:] = [(probability) /twocount for probability in probabilities[2]]  #probability of zero given regions
    probabilities2[2][:] = [(probability) /twocount for probability in probabilities2[2]]            # probability of zero given not regions
    probabilitiesbad[2][:] = [(probability) /(imagecount-twocount) for probability in probabilitiesbad[2]]     #probabilty of not zero given regions
    probabilitiesbad2[2][:] = [(probability) /(imagecount-twocount) for probability in probabilitiesbad2[2]]    # probability of not zero given not regions


    probabilities[3][:] = [(probability) /threecount for probability in probabilities[3]]  #probability of zero given regions
    probabilities2[3][:] = [(probability) /threecount for probability in probabilities2[3]]            # probability of zero given not regions
    probabilitiesbad[3][:] = [(probability) /(imagecount-threecount) for probability in probabilitiesbad[3]]     #probabilty of not zero given regions
    probabilitiesbad2[3][:] = [(probability) /(imagecount-threecount) for probability in probabilitiesbad2[3]]    # probability of not zero given not regions


    probabilities[4][:] = [(probability) /fourcount for probability in probabilities[4]]  #probability of zero given regions
    probabilities2[4][:] = [(probability) /fourcount for probability in probabilities2[4]]            # probability of zero given not regions
    probabilitiesbad[4][:] = [(probability) /(imagecount-fourcount) for probability in probabilitiesbad[4]]     #probabilty of not zero given regions
    probabilitiesbad2[4][:] = [(probability) /(imagecount-fourcount) for probability in probabilitiesbad2[4]]    # probability of not zero given not regions


    probabilities[5][:] = [(probability) /fivecount for probability in probabilities[5]]  #probability of zero given regions
    probabilities2[5][:] = [(probability) /fivecount for probability in probabilities2[5]]            # probability of zero given not regions
    probabilitiesbad[5][:] = [(probability) /(imagecount-fivecount) for probability in probabilitiesbad[5]]     #probabilty of not zero given regions
    probabilitiesbad2[5][:] = [(probability) /(imagecount-fivecount) for probability in probabilitiesbad2[5]]    # probability of not zero given not regions


    probabilities[6][:] = [(probability) /sixcount for probability in probabilities[6]]  #probability of zero given regions
    probabilities2[6][:] = [(probability) /sixcount for probability in probabilities2[6]]            # probability of zero given not regions
    probabilitiesbad[6][:] = [(probability) /(imagecount-sixcount) for probability in probabilitiesbad[6]]     #probabilty of not zero given regions
    probabilitiesbad2[6][:] = [(probability) /(imagecount-sixcount) for probability in probabilitiesbad2[6]]    # probability of not zero given not regions


    probabilities[7][:] = [(probability) /sevencount for probability in probabilities[7]]  #probability of zero given regions
    probabilities2[7][:] = [(probability) /sevencount for probability in probabilities2[7]]            # probability of zero given not regions
    probabilitiesbad[7][:] = [(probability) /(imagecount-sevencount) for probability in probabilitiesbad[7]]     #probabilty of not zero given regions
    probabilitiesbad2[7][:] = [(probability) /(imagecount-sevencount) for probability in probabilitiesbad2[7]]    # probability of not zero given not regions


    probabilities[8][:] = [(probability) /eightcount for probability in probabilities[8]]  #probability of zero given regions
    probabilities2[8][:] = [(probability) /eightcount for probability in probabilities2[8]]            # probability of zero given not regions
    probabilitiesbad[8][:] = [(probability) /(imagecount-eightcount) for probability in probabilitiesbad[8]]     #probabilty of not zero given regions
    probabilitiesbad2[8][:] = [(probability) /(imagecount-eightcount) for probability in probabilitiesbad2[8]]    # probability of not zero given not regions


    probabilities[9][:] = [(probability) /ninecount for probability in probabilities[9]]  #probability of zero given regions
    probabilities2[9][:] = [(probability) /ninecount for probability in probabilities2[9]]            # probability of zero given not regions
    probabilitiesbad[9][:] = [(probability) /(imagecount-ninecount) for probability in probabilitiesbad[9]]     #probabilty of not zero given regions
    probabilitiesbad2[9][:] = [(probability) /(imagecount-ninecount) for probability in probabilitiesbad2[9]]    # probability of not zero given not regions

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