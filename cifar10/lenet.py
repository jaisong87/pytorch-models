'''

LeNet in PyTorch.
LeNet Paper (http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
LeNet Article (http://yann.lecun.com/exdb/lenet/index.html)
'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Input is 3 channels image and output has 6 channels
        # and convolution kernal size is 5*5 square kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Second convolution is also square kernel(5*5) and has 16 layers
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out