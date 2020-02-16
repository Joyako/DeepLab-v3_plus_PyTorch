import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        return self.relu(self.bn(self.conv(x)))


class SegNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Encoder
        self.encoder1 = nn.Sequential(
            BasicConv(3, 64,),
            BasicConv(64, 64))
        self.maxpool1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.encoder2 = nn.Sequential(
            BasicConv(64, 128),
            BasicConv(128, 128))
        self.maxpool2 = nn.MaxPool2d(2, 2, return_indices=True)

        self.encoder3 = nn.Sequential(
            BasicConv(128, 256),
            BasicConv(256, 256),
            BasicConv(256, 256)
        )
        self.maxpool3 = nn.MaxPool2d(2, 2, return_indices=True)

        self.encoder4 = nn.Sequential(
            BasicConv(256, 512),
            BasicConv(512, 512),
            BasicConv(512, 512)
        )
        self.maxpool4 = nn.MaxPool2d(2, 2, return_indices=True)

        self.encoder5 = nn.Sequential(
            BasicConv(512, 512),
            BasicConv(512, 512),
            BasicConv(512, 512)
        )
        self.maxpool5 = nn.MaxPool2d(2, 2, return_indices=True)

        # Decoder
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder1 = BasicConv(64, 64)
        self.last_conv = nn.Conv2d(64, num_classes, kernel_size=3)

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            BasicConv(128, 128),
            BasicConv(128, 64)
        )

        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            BasicConv(256, 256),
            BasicConv(256, 256),
            BasicConv(256, 128)
        )

        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            BasicConv(512, 512),
            BasicConv(512, 512),
            BasicConv(512, 256)
        )

        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder5 = nn.Sequential(
            BasicConv(512, 512),
            BasicConv(512, 512),
            BasicConv(512, 512)
        )

    def forward(self, x):
        x = self.encoder1(x)
        size1 = x.size()
        x, indices1 = self.maxpool1(x)

        x = self.encoder2(x)
        size2 = x.size()
        x, indices2 = self.maxpool2(x)

        x = self.encoder3(x)
        size3 = x.size()
        x, indices3 = self.maxpool3(x)

        x = self.encoder4(x)
        size4 = x.size()
        x, indices4 = self.maxpool4(x)

        x = self.encoder5(x)
        size5 = x.size()
        x, indices5 = self.maxpool5(x)

        # Decoder
        x = self.unpool5(x, indices5, output_size=size5)
        x = self.decoder5(x)

        x = self.unpool4(x, indices4, output_size=size4)
        x = self.decoder4(x)

        x = self.unpool3(x, indices3, output_size=size3)
        x = self.decoder3(x)

        x = self.unpool2(x, indices2, output_size=size2)
        x = self.decoder2(x)

        x = self.unpool1(x, indices1, output_size=size1)
        x = self.last_conv(x)

        return x

