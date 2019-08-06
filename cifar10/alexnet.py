'''

AlexNet variant for CIFAR-10 in PyTorch.
AlexNet Paper (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
The model is adopted to use mu ch smaller number of parameters to adopt to 10 way classification and the size of th
dataset. Using AlexNet as it is will cause heavy over-fitting.
'''
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # Input(224)-> Conv(48 channel) -> MaxPool -> Conv(128 Channel) -> MaxPool -> Conv(192 Channel) ->
        # Conv(192 Channel) -> Conv(128 Channel) -> FC(2048) -> FC(2048) -> FC(1000)
        # CIFAR-10 Input is 3 channel 32 X 32 network and output is 10 way classifier. Here is the scaled down version
        # Input(32)-> Conv(16 channel) -> MaxPool -> Conv(16 Channel) -> MaxPool -> Conv(24 Channel) ->
        # Conv(24 Channel) -> Conv(32 Channel) -> FC(64) -> FC(32) -> FC(10)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=2)
        self.conv3 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(24, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4608, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, kernel_size=3, stride=2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, kernel_size=3, stride=1)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.adaptive_avg_pool2d(out, (12, 12))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
