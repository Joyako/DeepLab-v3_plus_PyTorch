import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    Ref:
        Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.
    """
    def __init__(self, planes=128, num_classes=3):
        super(Decoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(planes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(planes + 256, 256, 3, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2):
        out1 = self.conv1(x2)
        h, w = x1.size()[2:]
        out0 = F.interpolate(x1, size=(h * 4, w * 4), mode='bilinear', align_corners=True)
        out = torch.cat((out0, out1), dim=1)
        out = self.conv2(out)
        out = F.interpolate(out, size=(h * 16, w * 16), mode='bilinear', align_corners=True)

        return out


