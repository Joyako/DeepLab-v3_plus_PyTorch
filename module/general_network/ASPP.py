import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(inplanes, planes, kernel_size, stride=1,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Ref:
        Rethinking Atrous Convolution for Semantic Image Segmentation
    """
    def __init__(self, inplanes=2048, planes=256, stride=16):
        super(ASPP, self).__init__()
        if stride == 8:
            dilation = [12, 24, 36]
        elif stride == 16:
            dilation = [6, 12, 18]
        else:
            raise NotImplementedError

        self.block1 = ConvBNReLU(inplanes, planes, 1, 0, 1)
        self.block2 = ConvBNReLU(inplanes, planes, 3, dilation[0], dilation[0])
        self.block3 = ConvBNReLU(inplanes, planes, 3, dilation[1], dilation[1])
        self.block4 = ConvBNReLU(inplanes, planes, 3, dilation[2], dilation[2])

        self.block5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            ConvBNReLU(inplanes, planes, 1, 0, 1),
        )

        self.conv = ConvBNReLU(planes * 5, planes, 1, 0, 1)
        self.dropout = nn.Dropout(0.5)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h1 = self.block1(x)
        h2 = self.block2(x)
        h3 = self.block3(x)
        h4 = self.block4(x)
        h5 = self.block5(x)
        h5 = F.interpolate(h5, size=x.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((h1, h2, h3, h4, h5), dim=1)
        x = self.conv(x)
        x = self.dropout(x)
        return x
