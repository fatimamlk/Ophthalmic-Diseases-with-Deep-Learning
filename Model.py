import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten layer
        self.flatten = nn.Flatten()

        # First fully connected layer
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        # Dropout layer
        self.dropout1 = nn.Dropout(p=0.5)
        # Second fully connected layer
        self.fc2 = nn.Linear(256, 128)
        # Dropout layer
        self.dropout2 = nn.Dropout(p=0.5)
        # Output layer for binary classification
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = self.flatten(x)  # Flatten the layer

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after first FC layer
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout after second FC layer
        x = torch.sigmoid(self.fc3(x))  # then Apply sigmoid to output layer for binary classification
        return x
