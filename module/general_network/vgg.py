import torch.nn as nn
import torch

cfg = {'A': [1, 1, 2, 2, 2],
       'B': [2, 2, 2, 2, 2],
       'C': [2, 2, 5, 5, 5],
       'D': [2, 2, 3, 3, 3],
       'E': [2, 2, 4, 4, 4]}


class BasicConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1, bn=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(planes) if bn else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)

        return x


class VGG(nn.Module):
    def __init__(self, num_layers, num_classes, bn=True):
        super().__init__()
        layers = []

        inplanes = 3
        planes = 64
        for num in num_layers:
            if num == 5:
                layers.append(BasicConv(inplanes, planes, 3, 1, bn=bn))
                layers.append(BasicConv(planes, planes, 3, 1, bn=bn))
                layers.append(BasicConv(planes, planes, 1, 1, bn=bn))
                inplanes = planes
            else:
                for n in range(num):
                    layers.append(BasicConv(inplanes, planes, 3, 1, bn=bn))
                    inplanes = planes
                    
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if planes == 512:
                continue
            planes = planes * 2

        self.features = nn.Sequential(*layers)

        self.gap = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(inplanes * 7 * 7, 4096),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def vgg11(num_classes=1000, bn=True):

    return VGG(cfg['A'], num_classes, bn)


def vgg13(num_classes=1000, bn=True):

    return VGG(cfg['B'], num_classes, bn)


def vgg16_1(num_classes=1000, bn=True):

    return VGG(cfg['C'], num_classes, bn)


def vgg16(num_classes=1000, bn=True):

    return VGG(cfg['D'], num_classes, bn)


def vgg19(num_classes=1000, bn=True):

    return VGG(cfg['E'], num_classes, bn)


net = vgg19()
print(net)
input = torch.randn(1, 3, 224, 224)
output = net(input)
print(output.size())


