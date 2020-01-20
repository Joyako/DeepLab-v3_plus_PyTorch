import torch.nn as nn
import torch


class BasicConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        return self.relu(self.bn(self.conv(x)))


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN32s(nn.Module):
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

        self.upsample = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        _, _, H, W = x.size()
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.classifer(x)

        # skip-connect: stride = 1
        x = self.upsample(x)
        x = x[:, :, 19:19 + H, 19:19 + W].contiguous()

        return x


class FCN16s(nn.Module):
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

        self.conv4 = BasicConv(512, num_classes, 1, padding=0)

        self.upsample4 = nn.ConvTranspose2d(num_classes, num_classes, 32, stride=16, bias=False)
        self.upsample5 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        _, _, H, W = x.size()
        x = self.stage1(x)
        x = self.stage2(x)
        h1 = x
        x = self.stage3(x)
        x = self.classifer(x)

        # skip-connect: stride = 16
        x = self.upsample5(x)
        h2 = x
        x = self.conv4(h1)
        x = x[:, :, 5:5 + h2.size(2), 5:5 + h2.size(3)]
        x = x + h2

        # skip-connect: stride = 1
        x = self.upsample4(x)
        x = x[:, :, 27:27 + H, 27:27 + W]

        return x


class FCN8s(nn.Module):
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

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        _, _, H, W = x.size()
        x = self.stage1(x)
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
        x = x[:, :, 31:31 + H, 31:31 + W]

        return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    plt.figure()

    rgb = cv2.imread('../../data/Emma.jpg', cv2.IMREAD_COLOR)[:, :, ::-1]
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    ax = plt.subplot(2, 2, 1)
    ax.set_title('original')
    ax.imshow(rgb)
    image = rgb.astype(np.float)
    image = image.transpose(2, 0, 1)
    image /= 255.
    image = torch.from_numpy(image)
    image = image.unsqueeze(0).float()

    modes = ['fcn-32s', 'fcn-16s', 'fcn-8s']
    for i, name in enumerate(modes):
        if name == 'fcn-32s':
            net = FCN32s(3)
        elif name == 'fcn-16s':
            net = FCN16s(3)
        elif name == 'fcn-8s':
            net = FCN8s(3)
        else:
            raise NotImplementedError
        output = net(image)

        output = output.detach().numpy().squeeze(0)
        output = output.transpose(1, 2, 0)
        output *= 255.
        output = output.astype(np.uint8)

        ax = plt.subplot(2, 2, i + 2)
        ax.set_title(name)
        ax.imshow(output)
    plt.show()
