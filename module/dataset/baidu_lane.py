import os, cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from .preprocess import Normalize, ToTensor, \
    RandomHorizontalFlip, RandomGaussianBlur, RandomScaleCrop


class BaiDuLaneDataset(Dataset):

    labels = {'void': {'id': 0, 'trainId': 0, 'category': 'void', 'catId': 0, 'ignoreInEval': False,
                       'color': [0, 0, 0]},
              's_w_d': {'id': 200, 'trainId': 1, 'category': 'dividing', 'catId': 1, 'ignoreInEval': False,
                        'color': [70, 130, 180]},
              's_y_d': {'id': 204, 'trainId': 1, 'category': 'dividing', 'catId': 1, 'ignoreInEval': False,
                        'color': [220, 20, 60]},
              'ds_w_dn': {'id': 213, 'trainId': 1, 'category': 'dividing', 'catId': 1, 'ignoreInEval': True,
                          'color': [128, 20, 128]},
              'ds_y_dn': {'id': 209, 'trainId': 1, 'category': 'dividing', 'catId': 1, 'ignoreInEval': False,
                          'color': [255, 0, 0]},
              'sb_y_do': {'id': 206, 'trainId': 1, 'category': 'dividing', 'catId': 1, 'ignoreInEval': True,
                          'color': [0, 0, 60]},
              'sb_w_do': {'id': 207, 'trainId': 1, 'category': 'dividing', 'catId': 1, 'ignoreInEval': True,
                          'color': [0, 60, 100]},
              'b_w_g': {'id': 201, 'trainId': 2, 'category': 'guiding', 'catId': 2, 'ignoreInEval': False,
                        'color': [0, 0, 142]},
              'b_y_g': {'id': 203, 'trainId': 2, 'category': 'guiding', 'catId': 2, 'ignoreInEval': False,
                        'color': [119, 11, 32]},
              'db_w_g': {'id': 211, 'trainId': 2, 'category': 'guiding', 'catId': 2, 'ignoreInEval': True,
                         'color': [244, 35, 232]},
              'db_y_g': {'id': 208, 'trainId': 2, 'category': 'guiding', 'catId': 2, 'ignoreInEval': True,
                         'color': [0, 0, 160]},
              'db_w_s': {'id': 216, 'trainId': 3, 'category': 'stopping', 'catId': 3, 'ignoreInEval': True,
                         'color': [153, 153, 153]},
              's_w_s': {'id': 217, 'trainId': 3, 'category': 'stopping', 'catId': 3, 'ignoreInEval': False,
                        'color': [220, 220, 0]},
              'ds_w_s': {'id': 215, 'trainId': 3, 'category': 'stopping', 'catId': 3, 'ignoreInEval': True,
                         'color': [250, 170, 30]},
              's_w_c': {'id': 218, 'trainId': 4, 'category': 'chevron', 'catId': 4, 'ignoreInEval': True,
                        'color': [102, 102, 156]},
              's_y_c': {'id': 219, 'trainId': 4, 'category': 'chevron', 'catId': 4, 'ignoreInEval': True,
                        'color': [128, 0, 0]},
              's_w_p': {'id': 210, 'trainId': 5, 'category': 'parking', 'catId': 5, 'ignoreInEval': False,
                        'color': [128, 64, 128]},
              's_n_p': {'id': 232, 'trainId': 5, 'category': 'parking', 'catId': 5, 'ignoreInEval': True,
                        'color': [238, 232, 170]},
              'c_wy_z': {'id': 214, 'trainId': 6, 'category': 'zebra', 'catId': 6, 'ignoreInEval': False,
                         'color': [190, 153, 153]},
              'a_w_u': {'id': 202, 'trainId': 7, 'category': 'thru/turn', 'catId': 7, 'ignoreInEval': True,
                        'color': [0, 0, 230]},
              'a_w_t': {'id': 220, 'trainId': 7, 'category': 'thru/turn', 'catId': 7, 'ignoreInEval': False,
                        'color': [128, 128, 0]},
              'a_w_tl': {'id': 221, 'trainId': 7, 'category': 'thru/turn', 'catId': 7, 'ignoreInEval': False,
                         'color': [128, 78, 160]},
              'a_w_tr': {'id': 222, 'trainId': 7, 'category': 'thru/turn', 'catId': 7, 'ignoreInEval': False,
                         'color': [150, 100, 100]},
              'a_w_tlr': {'id': 231, 'trainId': 7, 'category': 'thru/turn', 'catId': 7, 'ignoreInEval': True,
                          'color': [255, 165, 0]},
              'a_w_l': {'id': 224, 'trainId': 7, 'category': 'thru/turn', 'catId': 7, 'ignoreInEval': False,
                        'color': [180, 165, 180]},
              'a_w_r': {'id': 225, 'trainId': 7, 'category': 'thru/turn', 'catId': 7, 'ignoreInEval': False,
                        'color': [107, 142, 35]},
              'a_w_lr': {'id': 226, 'trainId': 7, 'category': 'thru/turn', 'catId': 7, 'ignoreInEval': False,
                         'color': [201, 255, 229]},
              'a_n_lu': {'id': 230, 'trainId': 7, 'category': 'thru/turn', 'catId': 7, 'ignoreInEval': True,
                         'color': [0, 191, 255]},
              'a_w_tu': {'id': 228, 'trainId': 7, 'category': 'thru/turn', 'catId': 7, 'ignoreInEval': True,
                         'color': [51, 255, 51]},
              'a_w_m': {'id': 229, 'trainId': 7, 'category': 'thru/turn', 'catId': 7, 'ignoreInEval': True,
                        'color': [250, 128, 224]},
              'a_y_t': {'id': 233, 'trainId': 7, 'category': 'thru/turn', 'catId': 7, 'ignoreInEval': True,
                        'color': [127, 255, 0]},
              'b_n_sr': {'id': 205, 'trainId': 8, 'category': 'reduction', 'catId': 8, 'ignoreInEval': False,
                         'color': [255, 128, 0]},
              'd_wy_za': {'id': 212, 'trainId': 8, 'category': 'attention', 'catId': 8, 'ignoreInEval': True,
                          'color': [0, 255, 255]},
              'r_wp_np': {'id': 227, 'trainId': 8, 'category': 'no parking', 'catId': 8, 'ignoreInEval': False,
                          'color': [178, 132, 190]},
              'vom_wy_n': {'id': 223, 'trainId': 8, 'category': 'others', 'catId': 8, 'ignoreInEval': True,
                           'color': [128, 128, 64]},
              'cm_n_n': {'id': 250, 'trainId': 8, 'category': 'others', 'catId': 8, 'ignoreInEval': False,
                         'color': [102, 0, 204]},
              'noise': {'id': 249, 'trainId': 0, 'category': 'ignored', 'catId': 0, 'ignoreInEval': True,
                        'color': [0, 153, 153]},
              'ignored': {'id': 255, 'trainId': 0, 'category': 'ignored', 'catId': 0, 'ignoreInEval': True,
                          'color': [255, 255, 255]},
              }

    @staticmethod
    def get_file_list(file_path, ext):
        file_list = []
        dir_path = os.listdir(file_path + '/' + ext)
        dir_path = sorted(dir_path)
        for dir in dir_path:
            dir = os.path.join(ext, dir)
            camera_file = os.listdir(file_path + '/' + dir)
            camera_file = sorted(camera_file)
            for file in camera_file:
                path = os.path.join(dir, file)
                for x in sorted(os.listdir(file_path + '/' + path)):
                    file_list.append(path + '/' + x)

        return file_list

    def __init__(self, root_file, phase='train', output_size=224, num_classes=9):
        super().__init__()
        self.root_file = os.path.join(root_file, phase)
        img_ext = 'ColorImage_road04/ColorImage'
        label_ext = 'Labels_road04/Label'
        self.img_list = self.get_file_list(self.root_file, img_ext)
        self.label_list = self.get_file_list(self.root_file, label_ext)
        self.output_size = output_size
        self.transform = self.preprocess(phase)
        self.num_classes = num_classes

    def __getitem__(self, item):
        img = cv2.imread(self.root_file + '/' + self.img_list[item], cv2.IMREAD_UNCHANGED)
        target = cv2.imread(self.root_file + '/' + self.label_list[item], cv2.IMREAD_UNCHANGED)
        target = self.encode_label_map(target)
        # target = self.get_color_map(target)
        img = Image.fromarray(img)
        target = Image.fromarray(target)
        sample = {'image': img, 'label': target}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_list)

    def encode_label_map(self, mask):
        h, w = mask.shape
        for value in self.labels.values():
            pixel = value['id']
            mask[mask == pixel] = value['catId']

        return mask

    def get_color_map(self, mask):
        h, w = mask.shape
        new_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for value in self.labels.values():
            pixel = value['id']
            new_mask[mask == pixel] = value['color']

        return new_mask

    def preprocess(self, phase):
        if phase == 'train':
            preprocess = transforms.Compose([
                RandomHorizontalFlip(),
                RandomScaleCrop(base_size=256, crop_size=self.output_size, fill=0),
                RandomGaussianBlur(),
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225)),
                ToTensor(),
            ])
        elif phase == 'test':
            preprocess = transforms.Compose([
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225)),
                ToTensor(),
            ])
        else:
            raise NotImplementedError

        return preprocess

"""
dataset = BaiDuLaneDataset('/Users/joy/Downloads/dataset')
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

for i, sample in enumerate(data_loader):

    img, label = sample['image'], sample['label']
    if i == 0:
        img = np.transpose(img[0].numpy(), axes=[1, 2, 0])
        img *= (0.229, 0.224, 0.225)
        img += (0.485, 0.456, 0.406)
        img *= 255.0
        img = img.astype(np.uint8)

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(label[0].numpy())
        plt.show()
        break
"""
