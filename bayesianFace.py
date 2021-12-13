from os import error
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

    train_labels_file = open('finalprojectdata/facedata/facedatalabeler.txt', 'r')
    train_labels = train_labels_file.read().split('\n')
    train_labels_file.close()
    train_labels = train_labels[:-1]

    train_images_file = open('finalprojectdata/facedata/facedatatrainer.txt', 'r')     
    train_temp_images = train_images_file.read().split('\n')
    train_images_file.close()
    train_images = []

    quantity_of_images = ((len(train_temp_images)//70)*percent)//100
    indeces = list(range(len(train_temp_images)//70))
    np.random.shuffle(indeces)
    indeces = indeces[:quantity_of_images]
    

    for j in indeces:         # For every potential image in your dataset
        image = []
        
        for i in range(70):
            string = ''                           # For every line in that image
            line = train_temp_images[j*70 + i]        # Choose a Line in the image
            for char in line:
                string += char
            image.append(string)



        train_images.append(image) #the right amount of images  for training after running the whole loop 

    train_labels = [train_labels[i] for i in indeces]                                       # the randomized labels for everything

    # getting the right amount of images for this training run 
    quantity_of_images = (len(train_labels)*percent)//100
    train_images = train_images[:quantity_of_images]
    train_labels = train_labels[:quantity_of_images]

    #generating features

    regions = [0.0 for _ in range(height*width)]
    probabilities = [0.0001 for _ in range(height*width)] #stores the probabilities of face given region true
    probabilities2 = [0.0001 for _ in range(height*width)] #stores the probabilities of face given region false
    probabilitiesbad = [0.0001 for _ in range(height*width)] #stores the probabilities of not face given region true
    probabilitiesbad2 = [0.0001 for _ in range(height*width)] #stores the probabilities of face given region false
    
    # go through each image and calculate symbol  in each 
    linecounter =0
    imageindex = -1
    probindex = 0
    faceCount = 0
    notfaceCount = 0
    for image in train_images:  # go through ever image
        regions = [0.0 for _ in range(height*width)] #reset the region features for each image
        currimage = image
        imageindex += 1
        linecounter =0
        probindex = 0
        for row in currimage:   # go through every row 
            linecounter+=1
            l = ((linecounter)%70)*height//70
            for c in range(len(row)):  # go through every cahracter in teh row
                if row[c]=='#' or row[c] == '+':
                    regions[l*width+c*width//60]+=1         #keep track of how many of these are in each region

        label = train_labels[imageindex]                 
        regions[:] = [(x*height*width) /4200 for x in regions]      #look at this
        
        if(label == '1'):  #if face check the regions and update the probabilities
            faceCount = faceCount +1
            for region in regions:
                if region >= percentage: #if >percentage elemeents in the region are + or #
                    probabilities[probindex] = probabilities[probindex] +1  # state given true and face =  true, increment by 1
                elif region <percentage:                                       # state given false and face = true
                    probabilities2[probindex] = probabilities2[probindex] +1    # state given false and face = true
                probindex =  probindex+1
        probindex = 0

        if(label == '0'):  #if not face check the regions and update the probabilities
            notfaceCount = notfaceCount +1
            for region in regions:
                if region >= percentage: #if >percentage elemeents in the region are + or #
                    probabilitiesbad[probindex] = probabilitiesbad[probindex] +1  # state given true and face =  false, increment by 1
                elif region <percentage:                                       # state given false and face = true
                    probabilitiesbad2[probindex] = probabilitiesbad2[probindex] +1    # state given false and face = falsw
                probindex =  probindex+1
        


    probabilities[:] = [(probability) /faceCount for probability in probabilities]  #divides the number of times face was true given x, by the number of faces trained
    probabilities2[:] = [(probability) /faceCount for probability in probabilities2]             #this gets the conditional probability
    probabilitiesbad[:] = [(probability) /notfaceCount for probability in probabilitiesbad]  #divides the number of times face was false given x, by the number of not faces trained
    probabilitiesbad2[:] = [(probability) /notfaceCount for probability in probabilitiesbad2]             #this gets the conditional probability
    
    #getting rid of probabilities that are 0 because that ruins the algorith, just mean these features are irrelevant



    accuracy = bayes_tester(probabilities, probabilities2,probabilitiesbad, probabilitiesbad2, height,width, percentage)

    return accuracy


def bayes_tester(probabilities, probabilities2,probabilitiesbad, probabilitiesbad2, height,width, percentage):
    # prepare the testing images and data to be input to the model

    test_labels_file = open('finalprojectdata/facedata/facelabeltest.txt')
    test_labels = test_labels_file.read().split('\n')
    test_labels_file.close()
    test_labels = test_labels[:-1]

    test_images_file = open('finalprojectdata/facedata/facedatatest.txt')
    test_temp_images = test_images_file.read().split('\n')
    test_images_file.close()
    test_images = []
    
    for j in range(len(test_temp_images)//70):         # For every potential image in your dataset
        image = []
        
        for i in range(70):
            string = ''                           # For every line in that image
            line = test_temp_images[j*70 + i]        # Choose a Line in the image
            for char in line:
                string += char
            image.append(string)



        test_images.append(image) #the right amount of images  for training after running the whole loop 

    regions = [0.0 for _ in range(height*width)]
    linecounter =0
    imageindex = -1
    probabilityface = 0
    probabilitynotface = 0
    probindex = 0
    total =0
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
            l = ((linecounter)%70)*height//70
            for c in range(len(row)):  # go through every cahracter in teh row
                if row[c]=='#' or row[c] == '+':
                    regions[l*width+c*width//60]+=1         #keep track of how many of these are in each region
        label = test_labels[imageindex]                 
        regions[:] = [(x*height*width) /4200 for x in regions]      #get region values
        
        for region in regions:
            if region >= percentage:
                if probabilityface == 0:
                    probabilityface = probabilities[probindex]  # probability of face given x
                if probabilitynotface == 0:
                    probabilitynotface = probabilitiesbad[probindex]
                probabilityface = probabilityface * probabilities[probindex] # state given true and face =  true, adjust probability
                probabilitynotface = probabilitynotface*probabilitiesbad[probindex]

            else:
                if probabilityface == 0:
                    probabilityface = probabilities2[probindex] #probability of face given not x
                if probabilitynotface == 0:
                    probabilitynotface = probabilitiesbad2[probindex]
                probabilityface = probabilityface * probabilities2[probindex] # state given face and face =  true, adjust probability
                probabilitynotface = probabilitynotface*probabilitiesbad2[probindex]
            probindex =  probindex+1

       # print( str(imageindex) + ": faceprob = " + str(probabilityface) + " , notfacepro = " + str(probabilitynotface))
        if probabilityface >= probabilitynotface: # if more likely to be face or probability of not face is 0
            guess = '1' # guess face
        else:
            guess = '0'
        if (guess != label ): #if wrong keep track of wrong
            bad = bad +1
        probabilityface = 0
        probabilitynotface = 0
   # print(bad)
    return (1-(bad*1.0/total))*100.0



def main():

    filecreator()


    accuracies = torch.zeros(10, 10)
    times = torch.zeros(10, 10)
    for j in range(10):
        for i in range(10):
            tic = time.time()
            accuracy = bayes_trainer((10*(1+i)),7,6, .1)
            toc = time.time()
            accuracies[i][j] = accuracy
            time1 = toc-tic
            times[i][j] = time1
    
    accuracies_std = torch.std(accuracies, dim=1)
    times_std = torch.std(times, dim=1)
    accuracies = torch.mean(accuracies, dim=1)
    times = torch.mean(times, dim=1)

    print(f'Mean Accuracies: {accuracies}, Accuracy STD: {accuracies_std}')
    print(f'Mean Times: {times}, Time STD: {times_std}')

    plt.plot([10*(i+1) for i in range(10)], accuracies, yerr = accuracies_std)
    plt.ylabel('Mean Accuracy')
    plt.xlabel('Percentage of Training Data Used')
    plt.title(f'Face Model Mean Accuracy by Training Data Access Percentage: ')
    plt.savefig(f'Face Model Mean Accuracy by Training Data Access Percentage:')

    plt.show()
    time.sleep(5)
    plt.clf()

    plt.plot([10*(i+1) for i in range(10)], times, yerr = times_std)
    plt.ylabel('Mean Time')
    plt.xlabel('Percentage of Training Data Used')
    plt.title(f'Face Model Mean Time by Training Data Access Percentage:')
    plt.savefig(f'Face Model Mean Time by Training Data Access Percentage: .png')

    plt.show()
    time.sleep(5)
    return

def old():
    filecreator()
    for i in range(10):
        
     accuracy = bayes_trainer((10*(1+i)),7,6, .1) #use bayes_trainer(10,7,7, .1) = use 10% data, 7 by 7 regions, if 10% of characters are  #  return true state
    toc = time.time()
    time = toc-tic
    print("Percentage data: %s Accuracy: %s Execution time: %s"  %(10*(i+1), round(accuracy,2), round((toc-tic),2)))
    




main()
