import torch
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from module.backbone_network.FCN import FCN8s, FCN16s, FCN32s
from module.dataset.baidu_lane import BaiDuLaneDataset
from module.utils.metric import Metric


parser = argparse.ArgumentParser(description="Params")
parser.add_argument('--model-path', type=str, default='./weights/FCN8s_pretrained.pth',
                    help='The path of model.')
parser.add_argument('--data-path', type=str, default='/Users/joy/Downloads/dataset/train',
                    help='The path of train data(/root/data/LaneSeg)')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--show-image', type=bool, default=True,
                    help='show predict result')
args = parser.parse_args()


def test(mode, num_classes):

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    if mode == 'fcn-8s':
        net = FCN8s(num_classes)
    elif mode == 'fcn-16s':
        net = FCN16s(num_classes)
    elif mode == 'fcn-32s':
        net = FCN32s(num_classes)
    else:
        raise NotImplementedError

    net.eval()
    net.to(device)

    # Load Model
    net.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage))

    dataset = BaiDuLaneDataset(args.data_path, 'test', num_classes=num_classes, output_size=(846, 255))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

    confusion_matrix = 0.
    metric = Metric(num_classes=num_classes)
    with torch.no_grad():
        for i_batch, sample in enumerate(data_loader):
            img, target = sample['image'], sample['label']
            img = img.to(device)
            target = target.to(device)

            outputs = net(img)
            # outputs = F.softmax(outputs)
            # bilinear
            outputs = F.interpolate(outputs, size=(1020, 3384), mode='bilinear', align_corners=True)
            preds = outputs.data.max(1)[1].cpu().numpy()
            target = target.cpu().numpy()
            mask = np.zeros((img.size(0), 690, 3384))
            # target = np.hstack((mask.astype(target.dtype), target))
            preds = np.hstack((mask.astype(preds.dtype), preds))
            confusion_matrix = metric.add(preds=preds, target=target, m=confusion_matrix)
            print("mIoU : {}".format(metric.mIoU(m=confusion_matrix)))
            if args.show_image:
                img = np.transpose(img[0].cpu().numpy(), axes=[1, 2, 0])
                img *= (0.229, 0.224, 0.225)
                img += (0.485, 0.456, 0.406)
                img *= 255.0
                img = img.astype(np.uint8)

                ax = plt.subplot(2, 2, 1)
                ax.set_title("Color Image")
                ax.imshow(img)

                ax = plt.subplot(2, 2, 3)
                ax.set_title("GroundTruth")
                target = target[0].astype(np.uint8)
                target = dataset.decode_color_map(target)
                ax.imshow(target)

                ax = plt.subplot(2, 2, 4)
                ax.set_title("Predict")
                pred = preds[0].astype(np.uint8)
                pred = dataset.decode_color_map(pred)
                ax.imshow(pred)
                plt.show()

        print('Confusion Matrix')
        print(confusion_matrix)


