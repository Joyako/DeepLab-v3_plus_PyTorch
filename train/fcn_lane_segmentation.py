import time
import torch
from torch.optim import Adam, SGD
from module.backbone_network.FCN import FCN8s, FCN16s, FCN32s
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F


def train(net, data_loader, optimizer, criterion, device, epoch, print_freq=40):
    net.train()

    for i_batch, sample in enumerate(data_loader):
        start_time = time.time()
        img, target = sample['image'], sample['label']
        img = img.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = net(img)
        output = F.softmax(output, dim=1)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        end_time = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        if i_batch % print_freq == 0:
            print("Iter: {}/{} Loss: {:.4f} LR: {:.8f} BatchTime: {:.4f}".format(i_batch, epoch,
                                                                                 loss.item(),
                                                                                 lr,
                                                                                 end_time))


def build_train(cfg, data_loader,
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

    for epoch in range(1, cfg['epoch'] + 1):
        train(net, data_loader, optimizer, criterion, device, epoch, cfg['print_freq'])
        lr_scheduler.step()
        torch.save(net.state_dict(), cfg['save_path'] + '/Lane_%s.pth' % epoch)
