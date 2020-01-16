import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        return self.relu(self.bn(self.conv(x)))



class FCN(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # stride = 8
        self.stage1 = nn.Sequential(
            BasicConv(3, 64, 3, padding=100),
            BasicConv(64, 64, 3),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            BasicConv(64, 128, 3),
            BasicConv(128, 128, 3),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            BasicConv(128, 256, 3),
            BasicConv(256, 256, 3),
            BasicConv(256, 256, 3),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        # stride = 16
        self.stage2 = nn.Sequential(
            BasicConv(256, 512, 3),
            BasicConv(512, 512, 3),
            BasicConv(512, 512, 3),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        # stride = 32
        self.stage3 = nn.Sequential(
            BasicConv(512, 512, 3),
            BasicConv(512, 512, 3),
            BasicConv(512, 512, 3),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifer = nn.Sequential(
            BasicConv(512, 4096, 7, padding=0),
            nn.Dropout2d(),
            BasicConv(4096, 4096, 1, padding=0),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, 1, bias=False),
        )

        self.conv3 = BasicConv(256, num_classes, 1, padding=0)
        self.conv4 = BasicConv(512, num_classes, 1, padding=0)

        self.upsample3 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, bias=False)
        self.upsample4 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)
        self.upsample5 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)


    def forward(self, x):
        input = x
        x =self.stage1(x)
        h1 = x
        x = self.stage2(x)
        h2 = x
        x = self.stage3(x)
        x = self.classifer(x)

        # skip-connect: stride = 16
        x = self.upsample5(x)
        h3 = x
        x = self.conv4(h2)
        x = x[:, :, 5:5 + h3.size(2), 5:5 + h3.size(3)]
        x = x + h3

        # skip-connect: stride = 8
        x = self.upsample4(x)
        h4 = x
        x = self.conv3(h1)
        x = x[:, :, 9:9 + h4.size(2), 9:9 + h4.size(3)]
        x = x + h4

        # skip-connect: stride = 1
        x = self.upsample3(x)
        x = x[:, :, 31:31 + input.size(2), 31:31 + input.size(3)]

        return x


x = torch.randn(1, 3, 224, 224)
net = FCN()
output = net(x)
