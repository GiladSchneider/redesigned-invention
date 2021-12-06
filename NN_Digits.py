import torch
from torch import nn, optim
from torchvision import transforms as T
import matplotlib.pyplot as plt

def digit_print(images, n = 1):
    for j in range(n):
        for i in range(28):
            print(f'{images[j][i*28 : i*28 + 27]}')
        print()

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
    # Create the txt files from the excutable
    filecreator()
    
    # Prepare the training images and labels to be input to the model
    train_labels_file = open('finalprojectdata/digitdata/digitdatalabeler.txt', 'r')
    train_labels = train_labels_file.read().split('\n')
    train_labels_file.close()
    train_labels = train_labels[:-1]

    train_images_file = open('finalprojectdata/digitdata/digitdatatrainer.txt', 'r')     
    train_temp_images = train_images_file.read().split('\n')
    train_images_file.close()
    train_images = []

    for j in range(len(train_temp_images)//28):       # For every potential image in your dataset
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
    
    quantity_of_images = (len(train_labels)*percent)//100
    train_images = train_images[:quantity_of_images]
    train_labels = train_labels[:quantity_of_images]

    train_labels = train_labels
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.LogSoftmax(dim = 1))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.0003)

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
    correct = 0
    out = model.forward(test_images)
    pred = torch.argmax(torch.exp(out), dim=1)
    equals = pred == test_labels
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    
    print(f'{percent}% model Accuracy: {accuracy.item()*100}')
    return accuracy

def main():
    accuracies = []
    epochs = 100
    for i in range(10):
        accuracy = model_driver(percent = 10*(i+1), epochs = epochs)
        accuracies.append(accuracy)
    plt.bar([10*(i+1) for i in range(10)], accuracies)
    plt.ylabel('Accuracy')
    plt.xlabel('Percentage of Training Data Used')
    plt.title(f'Model Accuracy by Training Data Access Percentage: {epochs} Epochs')
    plt.savefig(f'Model Accuracy by Training Data Access Percentage: {epochs} Epochs')

main()