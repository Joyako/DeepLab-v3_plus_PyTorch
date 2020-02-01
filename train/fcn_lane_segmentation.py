import time
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

from module.backbone_network.FCN import FCN8s, FCN16s, FCN32s
from module.utils.metric import Metric


def train(net, data_loader, optimizer, criterion, device, epoch, print_freq=40):
    net.train()
    num_data = len(data_loader)
    for i_batch, sample in enumerate(data_loader):
        start_time = time.time()
        img, target = sample['image'], sample['label']
        img = img.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = net(img)

        loss = criterion(output, target.long())

        loss.backward()
        optimizer.step()

        end_time = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        if i_batch % print_freq == 0:
            print("Train/Epoch: {} Iter: {}/{} Loss: {:.4f} LR: {:.8f} BatchTime: {:.4f}".format(epoch,
                                                                                                 i_batch, num_data,
                                                                                                 loss.item(),
                                                                                                 lr,
                                                                                                 end_time))


def validation(net, data_loader, criterion, device, epoch, metric, print_freq):
    net.eval()
    num_data = len(data_loader)
    for i_batch, sample in enumerate(data_loader):
        with torch.no_grad():
            img, target = sample['image'], sample['label']
            img = img.to(device)
            target = target.to(device)

            outputs = net(img)
            outputs = F.softmax(outputs)
            loss = criterion(outputs, target.long())
            preds = outputs.data.max(1)[1].cpu().numpy()
            mIoU = metric.mIoU(preds=preds, target=target.cpu().numpy())

        if i_batch % print_freq == 0:
            print("Validate/Epoch: {} Iter: {}/{} Loss: {:.4f} mIoU: {:.4f}".format(epoch,
                                                                                i_batch, num_data,
                                                                                loss.item(),
                                                                                mIoU))


def build_train(cfg, data_loader, val_loader,
                criterion, device):
    if cfg['mode'] == 'fcn-8s':
        net = FCN8s(cfg['num_classes'])
    elif cfg['mode'] == 'fcn-16s':
        net = FCN16s(cfg['num_classes'])
    elif cfg['mode'] == 'fcn-32s':
        net = FCN32s(cfg['num_classes'])
    else:
        raise NotImplementedError

    net.to(device)
    if cfg['pretrain']:
        net.load_state_dict(torch.load(cfg['model_path']))

    if cfg['optim'] == 'Adam':
        optimizer = Adam(net.parameters(),
                         lr=cfg['lr'],
                         weight_decay=cfg['weight_decay'])
    elif cfg['optim'] == 'SGD':
        optimizer = SGD(net.parameters(),
                        lr=cfg['lr'],
                        momentum=cfg['momentum'],
                        weight_decay=cfg['weight_decay'])
    else:
        raise NotImplementedError

    lr_scheduler = MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=0.1)
    metric = Metric(cfg['num_classes'])

    for epoch in range(1, cfg['epoch'] + 1):
        train(net, data_loader, optimizer, criterion, device, epoch, cfg['print_freq'])
        validation(net, val_loader, criterion, device, epoch, metric, cfg['print_freq'])

        lr_scheduler.step()
        if epoch > 50 and epoch % 10 == 0:
            torch.save(net.state_dict(), cfg['save_path'] + '/Lane_%s.pth' % epoch)
