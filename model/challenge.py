"""
EECS 445 - Introduction to Machine Learning
Fall 2023 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        # self.init_weights()
        
    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = nn.ReLU(inplace=True)(out)
        return out
    
    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""
        torch.manual_seed(42)

        for conv in [self.conv1, self.conv2]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
        

class Challenge(nn.Module):
    # a slightly modified ResNet18
    def __init__(self, block, num_blocks, num_classes=2):
        """
        Define the architecture, i.e. what layers our network contains.
        At the end of __init__() we call init_weights() to initialize all model parameters (weights and biases)
        in all layers to desired distributions.
        """
        super(Challenge, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def init_weights(self):
        torch.manual_seed(42)
        C_in = self.conv1.weight.size(1)
        nn.init.normal(self.conv1.weight, 0.0, 1 / sqrt(5*5*C_in))
        
        C_in2 = self.fc.weight.size(1)
        linearLayer = self.fc
        nn.init.normal_(linearLayer.weight, 0.0, 1 / sqrt(C_in2))
        nn.init.constant_(linearLayer.bias, 0.0)       

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        This function defines the forward propagation for a batch of input examples, by
        successively passing output of the previous layer as the input into the next layer (after applying
        activation functions), and returning the final output as a torch.Tensor object.

        You may optionally use the x.shape variables below to resize/view the size of
        the input matrix at different points of the forward pass.
        """

        ## TODO: implement forward pass for your network
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def getResNet18():
    return Challenge(ResNetBlock, [2,2,2,2])

class Challenge2(nn.Module):
    def __init__(self):
        """
        Define the architecture, i.e. what layers our network contains.
        At the end of __init__() we call init_weights() to initialize all model parameters (weights and biases)
        in all layers to desired distributions.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=(5,5), stride=(2,2), padding=2)
        self.bn1 = nn.BatchNorm2d(16) 
        self.conv2 = nn.Conv2d(16, 64, kernel_size=(5,5), stride=(2,2), padding=2)
        self.bn2 = nn.BatchNorm2d(64)  
        self.conv3 = nn.Conv2d(64, 8, kernel_size=(5,5), stride=(2,2), padding=2)
        self.bn3 = nn.BatchNorm2d(8)   
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.fc1 = nn.Linear(32, 2)
        self.init_weights()

    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""

        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        C_in = self.fc1.weight.size(1)
        # print('C_in: ', C_in)
        linearLayer = self.fc1
        nn.init.normal_(linearLayer.weight, 0.0, 1 / sqrt(C_in))
        nn.init.constant_(linearLayer.bias, 0.0)

    def forward(self, x):
        """
        This function defines the forward propagation for a batch of input examples, by
        successively passing output of the previous layer as the input into the next layer (after applying
        activation functions), and returning the final output as a torch.Tensor object.

        You may optionally use the x.shape variables below to resize/view the size of
        the input matrix at different points of the forward pass.
        """
        N, C, H, W = x.shape # C: channel; H: height; W: width; N: number?

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.reshape(N, 32)
        x = self.fc1(x)
        
        return x