import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
from time import time

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


def model_driver(percent = 10, epochs = 15):
    # Prepare the training images and labels to be input to the model
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

    for j in indeces:       # For every potential image in your dataset
        count = 0
        image = ''
        for i in range(70):                           # For every line in that image
            line = train_temp_images[j*70 + i]        # Choose a Line in the image
            for char in line:
                if char == ' ':
                    image += '0'
                else:
                    image += '1'
                count += 1

        train_images.append(image)

    train_images = torch.Tensor([[int(char) for char in image] for image in train_images])
    train_labels = [train_labels[i] for i in indeces]
    train_labels = torch.LongTensor([int(char) for char in train_labels])

    # prepare the testing images and data to be input to the model
    test_labels_file = open('finalprojectdata/facedata/facelabeltest.txt')
    test_labels = test_labels_file.read().split('\n')
    test_labels_file.close()
    test_labels = test_labels[:-1]

    test_images_file = open('finalprojectdata/facedata/facedatatest.txt')
    test_temp_images = test_images_file.read().split('\n')
    test_images_file.close()
    test_images = []
    
    for j in range(len(test_temp_images)//70):       # For every potential image in your dataset
        image = ''
        for i in range(70):                           # For every line in that image
            line = test_temp_images[j*70 + i]        # Choose a Line in the image
            for char in line:
                if char == ' ':
                    image += '0'
                else:
                    image += '1'
        test_images.append(image)

    test_images = torch.Tensor([[int(char) for char in image] for image in train_images])
    test_labels = torch.LongTensor([int(char) for char in train_labels])
    
    quantity_of_images = (len(train_labels)*percent)//100
    train_images = train_images[:quantity_of_images]
    train_labels = train_labels[:quantity_of_images]

    model = nn.Sequential(
        nn.Linear(4200, 3600),
        nn.ReLU(),
        nn.Linear(3600, 1800),
        nn.ReLU(),
        nn.Linear(1800, 900),
        nn.ReLU(),
        nn.Linear(900, 450),
        nn.ReLU(),
        nn.Linear(450, 1),
        nn.Sigmoid())

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr = 0.0003)

    start = time()
    
    losses = []
    for epoch in range(epochs):

        out = model.forward(train_images)
        out = torch.Tensor(out)
        train_labels.resize_(out.shape)
        loss = criterion(out, train_labels)
        losses.append(loss.item())
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Saving the loss over time
    # plt.plot(losses, label = 'losses')
    # plt.title(f'Digit_{percent}%_Percent_of_data')
    # plt.savefig(f'Digit_{percent}%_Percent_of_data')
    # plt.clf()

    # Texting the model using the testing data
    with torch.no_grad():
        correct = 0
        out = model.forward(test_images)
        for i in range(len(out)):
            if out[i][0] - 0.5 > 0:
                out[i][0] = 1
            else:
                out[i][0] = 0
    
        out.resize_(test_labels.shape)
        equals = out == test_labels
        accuracy = torch.mean(equals.type(torch.FloatTensor))
    
    end = time()
    e_time = end-start

    print(f'{percent}% model Accuracy: {accuracy.item()*100}, Training Time: {e_time}')
    return accuracy, e_time

def main():
    filecreator()

    accuracies = torch.zeros(10, 10)
    times = torch.zeros(10, 10)
    epochs = 10
    for j in range(10):
        for i in range(10):
            accuracy, time = model_driver(percent = 10*(i+1), epochs = epochs)
            accuracies[i][j] = accuracy
            times[i][j] = time

    accuracies_std = torch.std(accuracies, dim=1)
    times_std = torch.std(times, dim=1)
    accuracies = torch.mean(accuracies, dim=1)
    times = torch.mean(times, dim=1)

    print(f'Mean Accuracies: {accuracies}, Accuracy STD: {accuracies_std}')
    print(f'Mean Times: {times}, Time STD: {times_std}')

    plt.plot([10*(i+1) for i in range(10)], accuracies)
    plt.ylabel('Mean Accuracy')
    plt.xlabel('Percentage of Training Data Used')
    plt.title(f'Face Model Mean Accuracy by Training Data Access Percentage: {epochs} Epochs')
    plt.savefig(f'Face Model Mean Accuracy by Training Data Access Percentage: {epochs} Epochs')

    plt.clf()

    plt.plot([10*(i+1) for i in range(10)], times)
    plt.ylabel('Mean Time')
    plt.xlabel('Percentage of Training Data Used')
    plt.title(f'Face Model Mean Time by Training Data Access Percentage: {epochs} Epochs')
    plt.savefig(f'Face Model Mean Time by Training Data Access Percentage: {epochs} Epochs')

main()