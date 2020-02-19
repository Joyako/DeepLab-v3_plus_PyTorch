import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    """
    def __init__(self, in_channels, mid_channels, out_channels, upsample_mode='deconv'):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv = nn.Conv2d(in_channels, mid_channels, 1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)

        if self.upsample_mode == 'deconv':
            self.upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels,
                                               kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        elif self.upsample_mode == 'pixelshuffle':
            self.upsample = nn.PixelShuffle(upscale_factor=2)
        else:
            raise NotImplementedError

        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        identity = self.conv(identity)
        identity = self.bn(identity)
        x = x + identity
        x = self.relu2(x)

        x = self.upsample(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x


class ResNet_UNet(nn.Module):
    def __init__(self, backbone, num_classes, pretrained=False):
        super().__init__()
        if backbone == 'resnet-18':
            resnet = models.resnet18(pretrained=pretrained)
            filters = [64, 64, 128, 256, 512]
        elif backbone == 'resnet-34':
            resnet = models.resnet34(pretrained=pretrained)
            filters = [64, 64, 128, 256, 512]
        elif backbone == 'resnet-50':
            resnet = models.resnet50(pretrained=pretrained)
            filters = [64, 256, 512, 1024, 2048]
        elif backbone == 'resnet-101':
            resnet = models.resnet101(pretrained=pretrained)
            filters = [64, 256, 512, 1024, 2048]
        elif backbone == 'resnet-152':
            resnet = models.resnet152(pretrained=pretrained)
            filters = [64, 256, 512, 1024, 2048]
        elif backbone == 'resnext50_32x4d':
            resnet = models.resnext50_32x4d(pretrained=pretrained)
            filters = [64, 256, 512, 1024, 2048]
        elif backbone == 'resnext101_32x8d':
            resnet = models.resnext101_32x8d(pretrained=pretrained)
            filters = [64, 256, 512, 1024, 2048]
        elif backbone == 'wide_resnet50_2':
            resnet = models.wide_resnet50_2(pretrained=pretrained)
            filters = [64, 256, 512, 1024, 2048]
        elif backbone == 'wide_resnet101_2':
            resnet = models.wide_resnet101_2(pretrained=pretrained)
            filters = [64, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError

        # Encoder
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                              stride=1, padding=3, bias=False)
        self.bn = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder1 = Decoder(in_channels=filters[1] * 2, mid_channels=128,
                                out_channels=filters[0])
        self.decoder2 = Decoder(in_channels=filters[2] * 2, mid_channels=256,
                                out_channels=filters[1])
        self.decoder3 = Decoder(in_channels=filters[3] * 2, mid_channels=512,
                                out_channels=filters[2])
        self.decoder4 = Decoder(in_channels=filters[4], mid_channels=1024,
                                out_channels=filters[3])

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=filters[0] * 2, out_channels=filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=filters[0], out_channels=filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=filters[0], out_channels=num_classes, kernel_size=1),
            nn.Sigmoid(),
        )

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 52 * 52, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def stn(self, x):
        # Spatial transformer network forward function
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 52 * 52)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        # x = self.stn(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        h = self.maxpool(x)

        e1 = self.encoder1(h)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4)
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        x = self.final(torch.cat((d1, x), dim=1))

        return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    plt.figure(figsize=(14, 12))

    rgb = cv2.imread('../../data/Emma.jpg', cv2.IMREAD_COLOR)[:, :, ::-1]
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    ax = plt.subplot(5, 2, 1)
    ax.set_title('original')
    ax.imshow(rgb)
    image = rgb.astype(np.float)
    image = image.transpose(2, 0, 1)
    image /= 255.
    image = torch.from_numpy(image)
    image = image.unsqueeze(0).float()

    modes = ['resnet-18', 'resnet-34', 'resnet-50', 'resnet-101', 'resnet-152',
             'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']
    for i, name in enumerate(modes):
        net = ResNet_UNet(name, 3)
        output = net(image)

        output = output.detach().numpy().squeeze(0)
        output = output.transpose(1, 2, 0)
        output *= 255.
        output = output.astype(np.uint8)

        ax = plt.subplot(5, 2, i + 2)
        ax.set_title(name)
        ax.imshow(output)
    plt.show()

