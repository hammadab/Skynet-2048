import cv2
import numpy as np
import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import torchvision


def get_dataset(stage):
    # TODO:
    data = []  # images, class index of image (0-9)
    labels = []  # names of classes
    #  Read dataset files
    folders = [x[1] for x in os.walk('dataset')]  # walk through everything in dataset
    folders = folders[0]  # names of immediate folders in dataset
    for i, folder_name in enumerate(folders):  # for each folder in dataset
        labels.append(folder_name)  # folder name = class name
        j = 0
        for filename in glob.glob("dataset/" + str(folder_name) + "/*"):  # for each file in that folder
            j += 1
            if j <= 0.7 * 200:  # training
                if stage == "training":
                    data.append([np.transpose(np.array(cv2.imread(filename)), (2, 0, 1)) / 255, np.array(
                        [i])])  # load that image and convert it to numpy array, add the class index for this image
            elif j <= 0.8 * 200:  # validation
                if stage == "validation":
                    data.append([np.transpose(np.array(cv2.imread(filename)), (2, 0, 1)) / 255, np.array(
                        [i])])  # load that image and convert it to numpy array, add the class index for this image
            else:  # testing
                if stage == "testing":
                    data.append([np.transpose(np.array(cv2.imread(filename)), (2, 0, 1)) / 255, np.array(
                        [i])])  # load that image and convert it to numpy array, add the class index for this image
    # Normalize & flatten datasets
    # Construct training, validation and test sets
    return data  # dataset is a list containing numpy arrays of data and class index


class AnimalDataset(Dataset):
    #  TODO:
    #  Define constructor for AnimalDataset class
    #  HINT: You can pass processed data samples and their ground truth values as parameters
    def __init__(self, stage):
        self.data = get_dataset(stage)

    '''This function should return sample count in the dataset'''

    def __len__(self):
        return len(self.data)

    '''This function should return a single sample and its ground truth value from the dataset corresponding to index parameter '''

    def __getitem__(self, index):
        return self.data[index]


class ConvNet(nn.Module):
    '''Define your neural network'''

    def __init__(self, **kwargs):  #  you can add any additional parameters you want
        # TODO:
        #  You should create your neural network here
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, 3)
        self.conv2 = nn.Conv3d(1, 64, 3)
        self.conv3 = nn.Conv3d(1, 128, 3, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.out = nn.Linear(35389440, 10)  # 35389440 = 61440 * 24 * 24

    def forward(self, x):  #  you can add any additional parameters you want
        #  TODO:
        #  Forward propagation implementation should be here
        x = x.view(-1, 1, 3, 100, 100)
        x = F.relu(self.conv1(x))
        x = x.view(-1, 1, 32, 98, 98)
        x = F.relu(self.conv2(x))
        x = x.view(-1, 1920, 96, 96)  # 1920 = 64 * 30
        x = self.pool(x)
        x = x.view(-1, 1, 960, 48, 48)  # 960, 48, 48 = (1920, 96, 96) / 2
        x = self.conv3(x)
        x = F.relu(x)
        # print(x.shape)
        x = x.view(-1, 122880, 48, 48)  # 122880 = 128 * 960
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, 35389440)  # 35389440 = 61440 * 24 * 24
        x = F.softmax(self.out(x), dim=1)
        return x


# HINT: note that your training time should not take many days.
torch.backends.cudnn.enabled = False
# TODO:
# Pick your hyper parameters
max_epoch = 4000

learning_rate = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():  # you are free to change parameters
    # Create train dataset loader
    #  Create validation dataset loader
    #  Create test dataset loader
    #  initialize your GENet neural network
    model = ConvNet().to(device)

    #  define your loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
                                weight_decay=5e-04)  #  you can play with momentum and weight_decay parameters as well

    #  start training
    #  for each epoch calculate validation performance
    #  save best model according to validation performance
    best_acc = 0
    for epoch in range(0, max_epoch):
        train(epoch, model, criterion, optimizer, DataLoader(AnimalDataset("training"), batch_size=2, shuffle=True),
              device)
        acc = test(model, DataLoader(AnimalDataset("validation"), batch_size=2, shuffle=False), device)
        if acc > best_acc:
            best_acc = acc
            print("Best validation accuracy", acc)
            torch.save(model,
                       "/content/drive/My Drive/CS 464-1 Introduction to Machine Learning/HW3/CNN_best_model.pth")
    print("Final validation accuracy", acc)


''' Train your network for a one epoch '''


def train(epoch, model, criterion, optimizer, loader, device):  # you are free to change parameters
    model.train()

    accuracies = 0
    losses = 0
    batch_time = time.time()
    for batch_idx, (data, labels) in enumerate(loader):
        #  TODO:
        # Implement training code for a one iteration
        data = data.to(device)
        labels = labels.to(device)
        labels = labels.view(-1)
        # print(1)
        # zero the parameter gradients
        optimizer.zero_grad()
        # print(2)
        # forward + backward + optimize
        # print(data)
        # print(data.shape)
        outputs = model(data.float())
        # print(outputs)
        # print(labels)
        # print(outputs.shape)
        # print(labels.shape)
        # print(3)
        loss = criterion(outputs, labels)
        # print(4)
        loss.backward()
        # print(5)
        optimizer.step()
        # print(6)

        _, predicted = torch.max(outputs, 1)
        # accuracies += np.sum(np.array(predicted.numpy() == labels.numpy()))
        accuracies += np.sum(np.array(predicted.cpu().numpy() == labels.cpu().numpy()))
        # losses += loss.detach().numpy()
        losses += loss.cpu().detach().numpy()
        # print("blah train")
        # break

    if epoch % 10 == 9:
        batch_time = time.time() - batch_time
        print("Epoch: " + str(epoch + 1) + " accuracies: " + str(accuracies / 14) + " losses: " + str(
            losses))  # + " batch_time: " + str(batch_time))
        # print('Epoch: [{0}][{1}/{2}]\t'
        #         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #         'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
        #         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #         'Accu {acc.val:.4f} ({acc.avg:.4f})\t'.format(
        #         epoch + 1, batch_idx + 1, len(trainloader),
        #         batch_time=batch_time,
        #         data_time=data_time, # what is data time?
        #         loss=losses,
        #         acc=accuracies))


''' Test&Validate your network '''


def test(model, loader, device):  # you are free to change parameters
    model.eval()

    accuracies = 0
    with torch.no_grad():
        batch_time = time.time()
        for batch_idx, (data, labels) in enumerate(loader):
            #  TODO:
            # Implement test code
            data = data.to(device)
            labels = labels.to(device)
            labels = labels.view(-1)

            # forward
            outputs = model(data.float())
            _, predicted = torch.max(outputs, 1)
            # accuracies += np.sum(np.array(predicted.numpy() == labels.numpy()))
            accuracies += np.sum(np.array(predicted.cpu().numpy() == labels.cpu().numpy()))
            # print("blah test")
            # break
        batch_time = time.time() - batch_time

        # print('Time {batch_time.avg:.3f}\t'
        #       'Accu {acc.avg:.4f}\t'.format(
        #        batch_time=batch_time,
        #        acc=accuracies))
    # print("Test/Validate accuracies: " + str(accuracies) + " batch_time: " + str(batch_time))
    return accuracies / 2


since = time.time()
main()
print("Time elapsed:", (time.time() - since))