import torch
import torch.nn as nn


class DeformableConv2d(nn.Module):
    """
    Ref:
        Deformable ConvNets v2: More Deformable, Better Results(https://arxiv.org/pdf/1811.11168.pdf)
        Deformable Convolutional Networks(https://arxiv.org/pdf/1703.06211.pdf)

    """
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(inplanes, planes * 2, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 2)

        self.relu = nn.ReLU(inplace=True)

    def _bilinear_interpolation(self, pts1, pts3,
                                pts1_v, pts2_v, pts3_v, pts4_v, pts5):
        """
        Bi-linear Interpolation.
        :param pts1: [B, C, H, W, 2]
        :param pts3: [B, C, H, W, 2]
        :param pts1_v: [B, C, H, W]
        :param pts2_v: [B, C, H, W]
        :param pts3_v: [B, C, H, W]
        :param pts4_v: [B, C, H, W]
        :param pts5: [B, C, H, W, 2]
        :return: [B, C, H, W]
        """
        d1 = pts5[..., 0] - pts1[..., 0].float()
        d2 = pts3[..., 0].float() - pts5[..., 0]
        d3 = pts5[..., 1] - pts1[..., 1].float()
        d4 = pts3[..., 1].float() - pts5[..., 1]
        p1 = d1 * pts1_v + d2 * pts2_v
        p2 = d1 * pts4_v + d2 * pts3_v
        p = d3 * p1 + d4 * p2

        return p

    def _get_pixel_val(self, feature_map, pts_map):
        """

        :param feature_map: [B, C, H, W]
        :param pts_map: [B, C, H, W, 2]
        :return: [B, C, H, W, 1]
        """
        B, C, H, W = feature_map.size()
        feature_map = feature_map.view(B, C, H * W)
        pts_map = pts_map.view(B, C, H * W, -1)
        new_pts_map = pts_map[..., 0] * W + pts_map[..., 1]
        for i in range(B):
            for j in range(C):
                feature_map[i, j] = feature_map[i, j].take(new_pts_map[i, j])

        return feature_map.view(B, C, H, W)

    def _generate_map(self, feature_map, offset_map):
        """

        :param feature_map: [B, C, H, W]
        :param offset_map: [B, C * 2, H, W]
        :return: [B, C, H, W]
        """
        B, C, H, W = feature_map.size()
        offset_map = offset_map.view(B, C, H, W, 2)
        grid_map = torch.zeros((B, C, H, W, 2), dtype=torch.float32)
        for i, j in zip(range(H), range(W)):
            grid_map[:, :, i, :, 1] = grid_map[:, :, i, :, 1] + i
            grid_map[:, :, :, j, 0] = grid_map[:, :, :, j, 0] + j
        grid_map = grid_map + offset_map
        grid_map[..., 0] = torch.clamp(grid_map[..., 0], min=0, max=W - 1)
        grid_map[..., 1] = torch.clamp(grid_map[..., 1], min=0, max=H - 1)
        pts1_map = grid_map.floor().long()
        pts3_map = grid_map.ceil().long()
        pts2_map = torch.stack((pts1_map[..., 0], pts3_map[..., 1]), dim=4)
        pts4_map = torch.stack((pts3_map[..., 0], pts1_map[..., 1]), dim=4)

        pts1_val = self._get_pixel_val(feature_map, pts1_map)
        pts2_val = self._get_pixel_val(feature_map, pts2_map)
        pts3_val = self._get_pixel_val(feature_map, pts3_map)
        pts4_val = self._get_pixel_val(feature_map, pts4_map)

        return self._bilinear_interpolation(pts1_map, pts3_map,
                                            pts1_val, pts2_val, pts3_val, pts4_val, grid_map)

    def forward(self, x):
        out1 = self.bn1(self.conv1(x))
        out2 = self.bn2(self.conv2(x))

        x = self._generate_map(out1, offset_map=out2)
        x = self.relu(x)

        return x


input = torch.randn(1, 2, 4, 4)
dn = DeformableConv2d(2, 2, kernel_size=1)
output = dn(input)

