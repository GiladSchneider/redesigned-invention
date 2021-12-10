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

    for j in indeces:       # For every potential image in your dataset
        image = ''
        for i in range(28):                           # For every line in that image
            line = train_temp_images[j*28 + i]        # Choose a Line in the image
            for char in line:
                if char == ' ':
                    image += '0'
                elif char == '+':
                    image += '1'
                else:
                    image += '2'
        train_images.append(image)

    train_images = torch.Tensor([[int(char) for char in image] for image in train_images])
    
    train_labels = [train_labels[i] for i in indeces]
    train_labels = torch.LongTensor([int(char) for char in train_labels])

    # prepare the testing images and data to be input to the model
    test_labels_file = open('finalprojectdata/digitdata/digitlabeltestlabeler.txt')
    test_labels = test_labels_file.read().split('\n')
    test_labels_file.close()
    test_labels = test_labels[:-1]

    test_images_file = open('finalprojectdata/digitdata/digitdatatest.txt')
    test_temp_images = test_images_file.read().split('\n')
    test_images_file.close()
    test_images = []
    
    for j in range(len(test_temp_images)//28):       # For every potential image in your dataset
        image = ''
        for i in range(28):                           # For every line in that image
            line = test_temp_images[j*28 + i]        # Choose a Line in the image
            for char in line:
                if char == ' ':
                    image += '0'
                elif char == '+':
                    image += '1'
                else:
                    image += '2'
        test_images.append(image)

    test_images = torch.Tensor([[int(char) for char in image] for image in train_images])
    test_labels = torch.LongTensor([int(char) for char in train_labels])

    model = nn.Sequential(
        nn.Linear(784, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
        nn.LogSoftmax(dim = 1))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.0003)

    start = time()

    losses = []
    for epoch in range(epochs):

        out = model.forward(train_images)
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
        pred = torch.argmax(torch.exp(out), dim=1)
        equals = pred == test_labels
    
        total = 0
        correct  = 0
        for i in equals:
            if i:
                correct += 1
            total += 1
        accuracy = correct/total

    end = time()
    e_time = end - start
    print(f'{percent}% model Accuracy: {accuracy*100}, Training Time: {e_time}')
    return accuracy, e_time

def main():
    # Create the txt files from the excutable
    filecreator()

    accuracies = torch.zeros(10, 10)
    times = torch.zeros(10, 10)
    epochs = 20
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
    plt.title(f'Digit Model Mean Accuracy by Training Data Access Percentage: {epochs} Epochs')
    plt.savefig(f'Digit Model Mean Accuracy by Training Data Access Percentage: {epochs} Epochs')

    plt.clf()

    plt.plot([10*(i+1) for i in range(10)], times)
    plt.ylabel('Mean Time')
    plt.xlabel('Percentage of Training Data Used')
    plt.title(f'Digit Model Mean Time by Training Data Access Percentage: {epochs} Epochs')
    plt.savefig(f'Digit Model Mean Time by Training Data Access Percentage: {epochs} Epochs')

main()
