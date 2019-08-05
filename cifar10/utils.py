import torch
import torchvision
import torchvision.transforms as transforms
from datetime import datetime as dt

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def getTrainsetAndLoader():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    return trainset, trainloader


def getTestSetAndLoader():
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    return testset, testloader


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def evaluateCifar10(net, testloader):
    start_time = dt.now()
    correct = 0
    total = 0
    confMatrix =  [[0 for j in range(10)] for i in range(10)]
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            batchSize = labels.size(0)
            total += batchSize
            for i in range(batchSize):
                trueLabel = labels[i]
                predictedLabel = predicted[i]
                confMatrix[trueLabel][predictedLabel] = confMatrix[trueLabel][predictedLabel] + 1
            correct += (predicted == labels).sum().item()
    eval_time = dt.now() - start_time
    return correct, total, confMatrix, eval_time