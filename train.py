import argparse
import torch
from torch.utils.data import DataLoader

from train.fcn_lane_segmentation import build_train
from module.loss.semantic_segmentation import SegmentationLosses
from module.dataset.baidu_lane import BaiDuLaneDataset


args = argparse.ArgumentParser(description='SemanticSegmentation')
args.add_argument('--data-path', type=str, default='/root/private/datasets/lane_segmentation',
                  help='The path of train data')
args.add_argument('--batch-size', type=int, default=32, metavar='N',
                  help='input batch size for training (default: 64)')
args.add_argument('--print-freq', type=int, default=10, metavar='N',
                  help='how many batches to wait before logging training status')
args.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                  help='learning rate (default: 1e-3)')
args.add_argument('--momentum', type=float, default=0.9, metavar='M',
                  help='SGD momentum (default: 0.5)')
args.add_argument('--weight-decay', type=float, default=1e-5, metavar='W',
                  help='SGD weight decay (default: 1e-5)')
args.add_argument('--log-dir', default='runs/exp-0',
                  help='path of data for save log.')
args.add_argument('--epochs', type=int, default=200, metavar='N',
                  help='number of epochs to train (default: 20)')
args.add_argument('--num-classes', type=int, default=9, metavar='N',
                  help='number of classify.')
args.add_argument('--pretrain', type=bool, default=False,
                  help='Loading pretrain model.')
args.add_argument('--model-path', type=str, default='/root/private/SemanticSegmentation/weights/',
                  help='Model path.')
args.add_argument('--save-path', type=str, default='/root/private/SemanticSegmentation/weights/',
                  help='Model path.')
opt = args.parse_args()

criterion = SegmentationLosses()
dataset = BaiDuLaneDataset(opt.data_path)
data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=24)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num = len(dataset)

if __name__ == '__main__':
    cfg = {
        'mode': 'fcn-8s',
        'num_classes': opt.num_classes,
        'optim': 'Adam',
        'milestones': [100, 150],
        'weight_decay': opt.weight_decay,
        'print_freq': opt.print_freq,
        'lr': opt.lr,
        'momentum': opt.momentum,
        'epoch': opt.epochs,
        'pretrain': opt.pretrain,
        'model_path': opt.model_path,
        'save_path': opt.save_path,
    }
    criterion.to(device)

    build_train(cfg, data_loader, criterion, device, epoch)




