import torch
import torch.nn as nn

from general_network.decoder import Decoder
from backbone_network.AlignedXception import AlignedXception
from general_network.ASPP import ASPP
from backbone_network.resnet import resnet101


def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    "3x3 convolution with padding"

    kernel_size = np.asarray((3, 3))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                     padding=full_padding, dilation=dilation, bias=False)


class DeepLab(nn.Module):
    """
    Ref:
        Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.

    """
    def __init__(self, backbone='resnet', stride=16, num_classes=21):
        """

        :param backbone:
        :param stride:
        :param num_classes:
        """
        super(DeepLab, self).__init__()
        if backbone == 'aligned_inception':
            self.backbone = AlignedXception(stride)
            planes = 128
        elif backbone == 'resnet':
            self.backbone = resnet101(pretrained=True)
            planes = 256
        else:
            raise NotImplementedError
        self.aspp = ASPP(2048, planes=256, stride=16)
        self.decoder = Decoder(planes=planes, num_classes=num_classes)

    def forward(self, x):
        x1, x2 = self.backbone(x)
        x1 = self.aspp(x1)
        x = self.decoder(x1, x2)

        return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    plt.figure()

    rgb = cv2.imread('../data/Emma2.jpg', cv2.IMREAD_COLOR)[:, :, ::-1]
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    ax = plt.subplot(1, 2, 1)
    ax.set_title('original')
    ax.imshow(rgb)
    image = rgb.astype(np.float)
    image = image.transpose(2, 0, 1)
    image /= 255.
    image = torch.from_numpy(image)
    image = image.unsqueeze(0).float()

    name = 'aligned_inception'
    net = DeepLab(name, stride=16, num_classes=3)
    output = net(image)

    output = output.detach().numpy().squeeze(0)
    output = output.transpose(1, 2, 0)
    output *= 255.
    output = output.astype(np.uint8)

    ax = plt.subplot(1, 2, 2)
    ax.set_title(name)
    ax.imshow(output)
    plt.show()

