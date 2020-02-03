import torch
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F

from module.backbone_network.FCN import FCN8s, FCN16s, FCN32s
from module.dataset.baidu_lane import BaiDuLaneDataset
from module.utils.metric import Metric


parser = argparse.ArgumentParser(description="Params")
parser.add_argument('--model-path', type=str, default='/root/private/',
                    help='The path of model.')
parser.add_argument('--data-path', type=str, default='/root/private/datasets/lane_segmentation',
                    help='The path of train data')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num-classes', type=int, default=8, metavar='N',
                    help='number of classify.')
args = parser.parse_args()


def test(mode, num_classes):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    dataset = BaiDuLaneDataset(args.data_path, 'test', num_classes=args.num_classes)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=12)

    confusion_matrix = 0.
    metric = Metric(num_classes=args.num_classes)
    with torch.no_grad():
        for i_batch, sample in enumerate(data_loader):
            img, target = sample['image'], sample['label']
            img = img.to(device)
            target = target.to(device)

            outputs = net(img)
            outputs = F.softmax(outputs)
            preds = outputs.data.max(1)[1].cpu().numpy()
            confusion_matrix = metric.add(preds=preds, target=target.cpu().numpy(), m=confusion_matrix)

        print('Confusion Matrix')
        print(confusion_matrix)
        print("mIoU : {}".format(metric.mIoU(m=confusion_matrix)))


if __name__ == "__main__":
    test()

