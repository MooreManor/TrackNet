import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.batchnorm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.batchnorm(self.conv(x))))
        return x
class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.batchnorm(self.conv(x))))
        return x

class ConvBlock_without_Pooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock_without_Pooling, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3), stride=(1))
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.batchnorm(self.conv(x)))
        return x

class EventsSpotting(nn.Module):
    def __init__(self, dropout_p):
        super(EventsSpotting, self).__init__()
        # self.conv1 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv3d(512, 512, (7, 3, 3), stride=(3,1,1), padding=(0, 1, 1))
        self.batchnorm = nn.BatchNorm3d(512)
        self.relu = nn.ReLU()
        self.dropout3d = nn.Dropout3d(p=dropout_p)
        self.convblock = ConvBlock(in_channels=512, out_channels=512)
        self.convblock2d = ConvBlock2d(in_channels=512, out_channels=512)
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=3)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, seq_feat):
        input_eventspotting = seq_feat
        x = self.relu(self.batchnorm(self.conv1(input_eventspotting)))
        x = self.dropout3d(x)
        x = self.convblock(x)
        x = torch.squeeze(x)
        x = self.convblock2d(x)
        x = self.convblock2d(x)
        # x = self.dropout2d(x)

        x = x.contiguous().view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        out = self.softmax(self.fc2(x))

        return out


if __name__ == '__main__':
    net = EventsSpotting(0.5)
    inp = torch.randn(2, 512, 20, 12, 20)
    res = net(inp)