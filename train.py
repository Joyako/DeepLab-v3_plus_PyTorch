import argparse
import torch
from torch.utils.data import DataLoader

from train.fcn_lane_segmentation import build_train
from module.loss.semantic_segmentation import SegmentationLosses
from module.dataset.baidu_lane import BaiDuLaneDataset


parser = argparse.ArgumentParser(description='SemanticSegmentation')
parser.add_argument('--data-path', type=str, default='/root/private/datasets/lane_segmentation',
                    help='The path of train data')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--print-freq', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight-decay', type=float, default=1e-5, metavar='W',
                    help='SGD weight decay (default: 1e-5)')
parser.add_argument('--log-dir', default='runs/exp-0',
                    help='path of data for save log.')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--num-classes', type=int, default=9, metavar='N',
                    help='number of classify.')
parser.add_argument('--pretrain', type=bool, default=False,
                    help='Loading pretrain model.')
parser.add_argument('--model-path', type=str, default='/root/private/SemanticSegmentation/weights/',
                    help='Model path.')
parser.add_argument('--save-path', type=str, default='/root/private/SemanticSegmentation/weights/',
                    help='Model path.')
args = parser.parse_args()

criterion = SegmentationLosses()
dataset = BaiDuLaneDataset(args.data_path)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=24)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num = len(dataset)

if __name__ == '__main__':
    cfg = {
        'mode': 'fcn-8s',
        'num_classes': args.num_classes,
        'optim': 'Adam',
        'milestones': [100, 150],
        'weight_decay': args.weight_decay,
        'print_freq': args.print_freq,
        'lr': args.lr,
        'momentum': args.momentum,
        'epoch': args.epochs,
        'pretrain': args.pretrain,
        'model_path': args.model_path,
        'save_path': args.save_path,
    }
    criterion.to(device)

    build_train(cfg, data_loader, criterion, device)




