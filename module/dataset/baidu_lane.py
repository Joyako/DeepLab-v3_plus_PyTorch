import os, cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from .preprocess import Normalize, ToTensor, \
    RandomHorizontalFlip, RandomGaussianBlur, RandomScaleCrop, FixedResize, AdjustColor


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
        if ext == '':
            dirs = ['Image_Data/Road02', 'Image_Data/Road03', 'Image_Data/Road04']
        elif ext == 'Label':
            dirs = ['Gray_Label/Label_road02', 'Gray_Label/Label_road03',
                    'Gray_Label/Label_road04']
        else:
            raise NotImplementedError

        for d in dirs:
            f_path = os.path.join(file_path, d, ext)
            dir_path = os.listdir(f_path)

            dir_path = sorted(dir_path)
            for dir in dir_path:
                dir = os.path.join(d, ext, dir)
                camera_file = os.listdir(file_path + '/' + dir)
                camera_file = sorted(camera_file)
                for file in camera_file:
                    path = os.path.join(dir, file)
                    for x in sorted(os.listdir(file_path + '/' + path)):
                        file_list.append(path + '/' + x)

        return file_list

    def __init__(self, root_file, phase='train', output_size=(846, 255), num_classes=8, adjust_factor=(0.3, 2.)):
        super().__init__()
        assert phase in ['train', 'val', 'test']
        self.root_file = root_file
        img_ext = ''
        label_ext = 'Label'
        self.img_list = self.get_file_list(self.root_file, img_ext)
        self.label_list = self.get_file_list(self.root_file, label_ext)
        self.output_size = output_size
        self.factor = adjust_factor
        self.transform = self.preprocess(phase)
        self.num_classes = num_classes
        self.phase = phase

        num_data = len(self.img_list)
        assert num_data == len(self.label_list)

        # Shuffle data.
        np.random.seed(0)
        data_list = np.random.permutation(num_data)
        self.img_list = np.array(self.img_list)[data_list].tolist()
        self.label_list = np.array(self.label_list)[data_list].tolist()

        # Allocate the data set as 7:1:2
        if phase == 'train':
            self.img_list = self.img_list[0:int(0.7 * num_data)]
            self.label_list = self.label_list[0:int(0.7 * num_data)]
        elif phase == 'val':
            self.img_list = self.img_list[int(0.7 * num_data):int(0.8 * num_data)]
            self.label_list = self.label_list[int(0.7 * num_data):int(0.8 * num_data)]
        elif phase == 'test':
            self.img_list = self.img_list[int(0.8 * num_data):]
            self.label_list = self.label_list[int(0.8 * num_data):]
        else:
            raise NotImplementedError

    def __getitem__(self, item):
        img = cv2.imread(self.root_file + '/' + self.img_list[item], cv2.IMREAD_UNCHANGED)
        target = cv2.imread(self.root_file + '/' + self.label_list[item], cv2.IMREAD_UNCHANGED)
        offset = 690
        img = img[offset:, :]
        if self.phase != 'test':
            target = target[offset:, :]
        # print(self.img_list[item])
        # print(self.label_list[item])
        target = self.encode_label_map(target)
        img = Image.fromarray(img)
        target = Image.fromarray(target)
        sample = {'image': img, 'label': target}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        # print(len(self.img_list))
        return len(self.img_list)

    def data_generator(self, batch_size):
        index = np.arange(0, len(self.img_list))
        while len(index):
            select = np.random.choice(index, batch_size)
            images = []
            targets = []
            for item in select:
                img = cv2.imread(self.root_file + '/' + self.img_list[item], cv2.IMREAD_UNCHANGED)
                target = cv2.imread(self.root_file + '/' + self.label_list[item], cv2.IMREAD_UNCHANGED)

                index = np.delete(index, select)
                sample = {'image': img, 'label': target}
                if self.transforms is not None:
                    sample = self.transform(sample)
                images.append(sample['image'])
                targets.append(sample['label'])

            yield {'image': images, 'label': targets}

    def encode_label_map(self, mask):
        for value in self.labels.values():
            pixel = value['id']
            if value['ignoreInEval']:
                # 0: category as background
                mask[mask == pixel] = 0
            else:
                trainId = value['trainId']
                if trainId > 4:
                    trainId -= 1
                mask[mask == pixel] = trainId

        return mask

    def decode_label_map(self, mask):
        mask[mask == 1] = 200
        mask[mask == 2] = 201
        mask[mask == 3] = 216
        mask[mask == 4] = 210
        mask[mask == 5] = 214
        mask[mask == 6] = 202
        mask[mask == 7] = 205

        return mask

    def preprocess(self, phase):
        if phase == 'train':
            preprocess = transforms.Compose([
                RandomHorizontalFlip(),
                # RandomScaleCrop(base_size=1024, crop_size=self.output_size, fill=255),
                FixedResize(self.output_size),
                AdjustColor(self.factor),
                RandomGaussianBlur(),
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225)),
                ToTensor(),
            ])

        elif phase == 'val':
            preprocess = transforms.Compose([
                FixedResize(self.output_size),
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225)),
                ToTensor(),
            ])

        elif phase == 'test':
            preprocess = transforms.Compose([
                FixedResize(self.output_size, is_resize=False),
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225)),
                ToTensor(),
            ])

        else:
            raise NotImplementedError

        return preprocess


if __name__ == '__main__':
    dataset = BaiDuLaneDataset('/Users/joy/Downloads/dataset/train', 'test', output_size=(846, 255),
                               adjust_factor=(5, 18))
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for i, sample in enumerate(data_loader):

        img, label = sample['image'], sample['label']
        if i == 0:
            # print(label.size())
            img = np.transpose(img[0].numpy(), axes=[1, 2, 0])
            img *= (0.229, 0.224, 0.225)
            img += (0.485, 0.456, 0.406)
            img *= 255.0
            img = img.astype(np.uint8)

            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            target = label[0].numpy().astype(np.uint8)
            print(np.bincount(target.reshape(-1)))
            plt.imshow(target, cmap='gray')
            plt.show()
            break

