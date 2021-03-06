from __future__ import division,print_function

import sys
from tqdm import tqdm_notebook as tqdm

import random
import matplotlib.pyplot as plt
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable, grad
from torchvision import datasets, transforms as trn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader


import argparse
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import sys
sys.path.append('./')
from PIL import Image
import os
import metrics

import matplotlib.pyplot as plt
import csv


global global_cfg
global_cfg = dict()


# Implement new policy here!
def lr_func_cosine(cfg, cur_epoch):
    return (
            cfg['lr']
            * (math.cos(math.pi * cur_epoch / cfg['max_epoch']) + 1.0)
            * 0.5
        )

_LR_POLICY = {
    'cosine' : lr_func_cosine,
}
        
def get_lr_at_epoch(cfg, cur_epoch):
    lr = get_lr_func(cfg['policy'])(cfg, cur_epoch)
    # Perform warm up.
    if cur_epoch < cfg['warm_epoch']:
        lr_start = cfg['warm_lr']
        lr_end = get_lr_func(cfg['policy'])(
            cfg, cfg['warm_epoch']
        )
        alpha = (lr_end - lr_start) / cfg['warm_epoch']
        lr = cur_epoch * alpha + lr_start
    return lr
        
    
def get_lr_func(policy):
    if policy in _LR_POLICY.keys():
        return _LR_POLICY[policy]
    else:
        raise NotImplementedError(
            "Does not support '{}' lr policy".format(policy)
        )

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

class SaigeDataset3(torch.utils.data.Dataset):
    def __init__(self, data_root, split_root, dataset, split, transform, targets):
        """
            data_root(str) : Root directory of datasets (e.g. "/home/sr2/HDD2/Openset/")
            split_root(str) : Root directroy of split file (e.g. "/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/")
            dataset(str) : dataset name
            split(str) : ['train', 'valid', 'test']
            transform(torchvision transform) : image transform
            targets(list of str) : using targets
        """
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.transform = transform
        self.targets = targets
        self.data_list = []
        f = open(os.path.join(split_root, split + ".txt"), "r")
        while True:
            line = f.readline()
            if not line: break
            [target, _] = line[:-1].split("/")
            # Transform target
            if target in targets:
                target = targets.index(target)
                self.data_list.append((target, line[:-1]))
                
    def __getitem__(self, idx):
        (target, fpath) = self.data_list[idx]
        img = Image.open(os.path.join(self.data_root, fpath))
        rgb= img.convert("RGB")
        img = self.transform(rgb)
        return img, target
           
    def __len__(self):
        return len(self.data_list)

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, split, targets):
        """
            data_root(str) : Root directory of datasets (e.g. "/home/sr2/HDD2/Openset/")
            split_root(str) : Root directroy of split file (e.g. "/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/")
            dataset(str) : dataset name
            split(str) : ['train', 'valid', 'test']
            transform(torchvision transform) : image transform
            targets(list of str) : using targets
        """
        if split=='train':
            self.data_root = './mnist_png/training'
        else:
            self.data_root = './mnist_png/testing'
        # self.dataset = dataset
        self.transform = trn.Compose([trn.Pad(2),trn.ToTensor()])
        self.targets = targets
        self.data_list = []
        # f = open(os.path.join(split_root, split + ".txt"), "r")
        for direc in os.listdir(self.data_root):
            for image in os.listdir(os.path.join(self.data_root,direc)):
                target=int(direc) # Transform target
                if target in targets:
                    target = targets.index(target)
                    self.data_list.append((target, os.path.join(direc,image)))
                    # print(target)
                    # print(os.path.join(direc,image))

                
    def __getitem__(self, idx):
        (target, fpath) = self.data_list[idx]
        img = Image.open(os.path.join(self.data_root, fpath))
        rgb= img.convert("RGB")
        img = self.transform(rgb)
        return img, target
           
    def __len__(self):
        return len(self.data_list)

class SVHNDataset(torch.utils.data.Dataset):
    def __init__(self, split, targets):
        """
            data_root(str) : Root directory of datasets (e.g. "/home/sr2/HDD2/Openset/")
            split_root(str) : Root directroy of split file (e.g. "/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/")
            dataset(str) : dataset name
            split(str) : ['train', 'valid', 'test']
            transform(torchvision transform) : image transform
            targets(list of str) : using targets
        """
        if split=='train':
            self.data_root = './svhn_png/training'
        else:
            self.data_root = './svhn_png/testing'
        # self.dataset = dataset
        self.transform = trn.Compose([trn.Pad(2),trn.ToTensor()])
        self.targets = targets
        self.data_list = []
        # f = open(os.path.join(split_root, split + ".txt"), "r")
        for direc in os.listdir(self.data_root):
            for image in os.listdir(os.path.join(self.data_root,direc)):
                target=int(direc) # Transform target
                if target in targets:
                    target = targets.index(target)
                    self.data_list.append((target, os.path.join(direc,image)))
                    # print(target)
                    # print(os.path.join(direc,image))

                
    def __getitem__(self, idx):
        (target, fpath) = self.data_list[idx]
        img = Image.open(os.path.join(self.data_root, fpath))
        rgb= img.convert('L')
        rgb= rgb.convert('rgb')
        img = self.transform(rgb)
        return img, target
           
    def __len__(self):
        return len(self.data_list)

class TrafficDataset(torch.utils.data.Dataset):
    def __init__(self, split, targets):
        """
            data_root(str) : Root directory of datasets (e.g. "/home/sr2/HDD2/Openset/")
            split_root(str) : Root directroy of split file (e.g. "/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/")
            dataset(str) : dataset name
            split(str) : ['train', 'valid', 'test']
            transform(torchvision transform) : image transform
            targets(list of str) : using targets
        """
        self.split=split
        if self.split=='train':
            self.data_root = './GTSRB/Final_Training/Images'
        else:
            self.data_root = './GTSRB/Final_Test/Images'
        self.targets = targets
        self.transform=trn.Compose([trn.Resize([64,64]),trn.ToTensor()])
        self.data_list = []

        self.images=[]
        self.labels=[]

        if self.split=='train':
            # loop over all 42 classes
            for c in self.targets:
                prefix = self.data_root + '/' + format(c, '05d') + '/' # subdirectory for class
                gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
                gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
                next(gtReader) # skip header
                # loop over all images in current annotations file
                for row in gtReader:
                    if int(row[7]) in self.targets:
                        target=self.targets.index(int(row[7]))
                        img=plt.imread(prefix + row[0])
                        img=Image.fromarray(img)
                        img=self.transform(img)
                        self.images.append(img) # the 1th column is the filename
                        self.labels.append(torch.tensor(target)) # the 8th column is the label
                gtFile.close()
        else:
            prefix = self.data_root + '/' # subdirectory for class
            gtFile = open(prefix + 'GT-final_test.csv') # annotations file
            gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
            next(gtReader) # skip header
            # loop over all images in current annotations file
            for row in gtReader:
                if int(row[7]) in self.targets:
                    target=self.targets.index(int(row[7]))
                    img=plt.imread(prefix + row[0])
                    img=Image.fromarray(img)
                    img=self.transform(img)
                    self.images.append(img) # the 1th column is the filename
                    self.labels.append(torch.tensor(target)) # the 8th column is the label
                # labels.append(row[7]) # the 8th column is the label
            gtFile.close()
        # f = open(os.path.join(split_root, split + ".txt"), "r")

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
       
    def __len__(self):
        return len(self.labels)

class LargeCIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split_root, dataset, split, transform, targets):
        """
            data_root(str) : Root directory of datasets (e.g. "/home/sr2/HDD2/Openset/")
            split_root(str) : Root directroy of split file (e.g. "/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/")
            dataset(str) : dataset name
            split(str) : ['train', 'valid', 'test']
            transform(torchvision transform) : image transform
            targets(list of str) : using targets
        """
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.transform = transform
        self.targets = targets
        
        self.data_list = []
        f = open(os.path.join(split_root, split + ".txt"), "r")
        
        while True:
            line = f.readline()
            if not line: break
            [target, _] = line[:-1].split("/")
            # Transform target
            if target in targets:
                target = targets.index(target)
                self.data_list.append((target, line[:-1]))
                
            
    def __getitem__(self, idx):
        (target, fpath) = self.data_list[idx]
        img = Image.open(os.path.join(self.data_root, fpath))
        img = self.transform(img)
        return img, target
       
    
    def __len__(self):
        return len(self.data_list)
    
    def __len__(self):
        return len(self.data_list)


class SaigeDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split_root, dataset, split, transform, targets):
        """
            data_root(str) : Root directory of datasets (e.g. "/home/sr2/HDD2/Openset/")
            split_root(str) : Root directroy of split file (e.g. "/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/")
            dataset(str) : dataset name
            split(str) : ['train', 'valid', 'test']
            transform(torchvision transform) : image transform
            targets(list of str) : using targets
        """
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.transform = transform
        self.targets = targets
        
        self.data_list = []
        f = open(os.path.join(split_root, split + ".txt"), "r")
        
        while True:
            line = f.readline()
            if not line: break
            [target, _] = line[:-1].split("/")
            # Transform target
            if target in targets:
                target = targets.index(target)
                self.data_list.append((target, line[:-1]))
                
            
    def __getitem__(self, idx):
        (target, fpath) = self.data_list[idx]
        img = Image.open(os.path.join(self.data_root, fpath))
        img = self.transform(img)
        return img, target
       
    
    def __len__(self):
        return len(self.data_list)
    
    def __len__(self):
        return len(self.data_list)


# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 


def getDataLoader(ds_cfg, dl_cfg, split, num_samples=10000):
    if split == 'train':
        train = True
        transform = ds_cfg['train_transform']
    else:
        train = False
        transform = ds_cfg['valid_transform']
        
    if 'split' in ds_cfg.keys() and ds_cfg['split'] == 'train':
           split = 'train'
    elif 'split' in ds_cfg.keys() and ds_cfg['split'] == 'valid':
            split = 'valid'
    elif 'split' in ds_cfg.keys() and ds_cfg['split'] == 'test':
            split = 'test'
    else:
            pass
    if ds_cfg['dataset'] in ['SDI/34Ah','SDI/37Ah','SDI/60Ah']:
        dataset = SaigeDataset(data_root=ds_cfg['data_root'],
                                            split_root=ds_cfg['split_root'],
                                            dataset=ds_cfg['dataset'],
                                            split=split,
                                            transform=transform,
                                            targets=ds_cfg['targets'])
        number= dataset.__len__()
        loader = DataLoader(dataset,
                            batch_size=ds_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset {} ready.'.format(ds_cfg['dataset']))
    elif 'mvtec' in ds_cfg['dataset']:
        dataset = SaigeDataset(data_root=ds_cfg['data_root'],
                                            split_root=ds_cfg['split_root'],
                                            dataset=ds_cfg['dataset'],
                                            split=split,
                                            transform=transform,
                                            targets=ds_cfg['targets'])
        number= dataset.__len__()
        loader = DataLoader(dataset,
                            batch_size=ds_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset {} ready.'.format(ds_cfg['dataset']))

    elif ds_cfg['dataset'] in ['Traffic']:
        dataset = TrafficDataset(split=split,targets=ds_cfg['targets'])
        number= dataset.__len__()
        loader = DataLoader(dataset,
                            batch_size=ds_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset {} ready.'.format(ds_cfg['dataset']))
    elif ds_cfg['dataset'] in ['MNIST']:
        dataset = MNISTDataset(split=split,targets=ds_cfg['targets'])
        number= dataset.__len__()
        loader = DataLoader(dataset,
                            batch_size=ds_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset {} ready.'.format(ds_cfg['dataset']))
    elif ds_cfg['dataset'] in ['SVHN']:
        dataset = SVHNDataset(split=split,targets=ds_cfg['targets'])
        number= dataset.__len__()
        loader = DataLoader(dataset,
                            batch_size=ds_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset {} ready.'.format(ds_cfg['dataset']))

    else :
        dataset = SaigeDataset3(data_root=ds_cfg['data_root'],
                                            split_root=ds_cfg['split_root'],
                                            dataset=ds_cfg['dataset'],
                                            split=split,
                                            transform=transform,
                                            targets=ds_cfg['targets'])
        number= dataset.__len__()
        loader = DataLoader(dataset,
                            batch_size=ds_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset {} ready.'.format(ds_cfg['dataset']))
    
    return loader

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv1 = conv3x3(3,self.inplanes)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    
    def record(self, t):
        if self.collecting:
            self.gram_feats.append(t)
    
    def gram_feature_list(self,x):
        self.collecting = True
        self.gram_feats = []
        self.forward(x)
        self.collecting = False
        temp = self.gram_feats
        self.gram_feats = []
        return temp
    
    def get_min_max(self, data, power):
        mins = []
        maxs = []
        which_layer=6
        for i in range(0,len(data),128):
            batch = data[i:i+128].cuda()
            feat_list = self.gram_feature_list(batch)
            for L1,feat_L in enumerate(feat_list):
                if L1%which_layer==0:
                    L=L1//which_layer
                    if L==len(mins):
                        mins.append([None]*len(power))
                        maxs.append([None]*len(power))

                    for p,P in enumerate(power):
                        g_p = G_p(feat_L,P)

                        current_min = g_p.min(dim=0,keepdim=True)[0]
                        current_max = g_p.max(dim=0,keepdim=True)[0]

                        if mins[L][p] is None:
                            mins[L][p] = current_min
                            maxs[L][p] = current_max
                        else:
                            mins[L][p] = torch.min(current_min,mins[L][p])
                            maxs[L][p] = torch.max(current_max,maxs[L][p])
        
        return mins,maxs


    def get_min_max_real(self, data, power):
        mins = []
        maxs = []
        which_layer=6
        for i in range(0,len(data),128):
            batch = data[i:i+128].cuda()
            feat_list = self.gram_feature_list(batch)
            for L1,feat_L in enumerate(feat_list):
                if L1%which_layer==0:
                    L=L1//which_layer
                    if L==len(mins):
                        mins.append([None]*len(power))
                        maxs.append([None]*len(power))

                    for p,P in enumerate(power):
                        g_p = G_p_entire(feat_L,P)

                        current_max = g_p.max(dim=0,keepdim=True)[0]
                        current_min = g_p.min(dim=0,keepdim=True)[0]

                        if mins[L][p] is None:
                            mins[L][p] = current_min
                            maxs[L][p] = current_max
                        else:
                            mins[L][p] = torch.min(current_min,mins[L][p])
                            maxs[L][p] = torch.max(current_max,maxs[L][p])
        
        return mins,maxs

    def get_deviations(self,data,power,mins,maxs):
        deviations = []
        which_layer=6
        for i in range(0,len(data),128):            
            batch = data[i:i+128].cuda()
            feat_list = self.gram_feature_list(batch)
            batch_deviations = []
            for L1,feat_L in enumerate(feat_list):
                if L1%which_layer==0:
                    L=L1//which_layer
                    dev = 0
                    for p,P in enumerate(power):
                        g_p = G_p(feat_L,P)

                        dev +=  (F.relu(mins[L][p]-g_p)/torch.abs(mins[L][p]+10**-6)).sum(dim=1,keepdim=True)
                        dev +=  (F.relu(g_p-maxs[L][p])/torch.abs(maxs[L][p]+10**-6)).sum(dim=1,keepdim=True)
                    batch_deviations.append(dev.cpu().detach().numpy())
            batch_deviations = np.concatenate(batch_deviations,axis=1)
            deviations.append(batch_deviations)
        deviations = np.concatenate(deviations,axis=0)
        
        return deviations

    def get_entire_Gram(self, data, power, mins_real, maxs_real, PRED):  
        # gram_l_p=dict()
        gram_inside=dict()

        which_layer=6
        count=0
        print("Start Generating Gram Matrices")
        for i in tqdm(range(0,len(data),128)):
            batch = data[i:i+16].cuda()
            feat_list = self.gram_feature_list(batch)
            for L1,feat_L in enumerate(feat_list):
                if L1%which_layer==0:
                    L=L1//which_layer
                    # gram_l_p[L] = dict()
                    # gram_new[L] = dict()
                    gram_inside[L]=dict()

                    for p,P in enumerate(power):
                        mins=mins_real[PRED][L][p]
                        # mins=mins.view(1,-1)
                        maxs=maxs_real[PRED][L][p]
                        # maxs=maxs.view(1,-1)
                        g_p=G_p_entire(feat_L,P)
                        # g_p=g_p.view(128,-1)

                        x=torch.zeros(128,g_p.size(1),g_p.size(2))
                        for k in tqdm(range(g_p.size(0)),leave=False):
                            for i in tqdm(range(g_p.size(1)),leave=False):
                                for j in tqdm(range(g_p.size(2)),leave=False):
                                    time.sleep(0.1)
                                    if g_p[k][j][i]>mins_real[PRED][L][p][0][j][i] and g_p[k][j][i]<maxs_real[PRED][L][p][0][j][i]:
                                        x[k][j][i]=1
                        # x=x.view(128,mins_real[PRED][L][p].size(2),-1)
                        if count==0:
                            gram_inside[L][p]=x
                            count=1
                        else :
                            gram_inside[L][p]=torch.cat((gram_inside[L][p],x),dim=0)
                        
                        # max_indices, maximum=G_p_entire(feat_L,P).view(G_p_entire(feat_L,P).size(0),-1).max(dim=1)
                        # min_indices, minimum=G_p_entire(feat_L,P).view(G_p_entire(feat_L,P).size(0),-1).min(dim=1)
                        # for index, x in enumerate(max_indices):
                            
                        #     gram_values[L][P]=torch.zeros(G_p_entire(feat_L,P).size(0),G_p_entire(feat_L,P).size(1),G_p_entire(feat_L,P).size(2))
                        #     gram_values[L][P][x//G_p_entire(feat_L,P).size(1),x%G_p_entire(feat_L,P).size(1)]=maximum[index]
                        #     gram_index[L][P][x//G_p_entire(feat_L,P).size(1),x%G_p_entire(feat_L,P).size(1)]=1
                        #     gram_values[L][P]=gram_values[L][P].sum(dim=0)
                        #     gram_index[L][P]=gram_index[L][P].sum(dim=0)

                        # gram_entire_values[L][P]+=gram_values[L][P]
                        # gram_entire_indexes[L][P]+=gram_indexes[L][P]

        return gram_inside


def summary_write(summary, writer):
    for key in summary.keys():
        writer.add_scalar(key, summary[key], summary['epoch'])
        

def train_epoch_wo_outlier(model, optimizer, in_loader, loss_func, cur_epoch, op_cfg, writer):
    global global_cfg
    model.train()
    avg_loss = 0
    correct = 0
    in_data_size = len(in_loader.dataset)
    for cur_iter, in_set in enumerate(in_loader):
        #TODO: Dimension of in_set and out_set should be checked!
        # Data to GPU
        data = in_set[0]
        targets = in_set[1]
        if cur_iter == 0:
            writer.add_image('in_dist target {}'.format(targets[0]), data[0], cur_epoch)
        data, targets = data.cuda(), targets.cuda()
        
        # Adjust Learning rate
        lr = get_lr_at_epoch(op_cfg, cur_epoch + float(cur_iter) / in_data_size)
        set_lr(optimizer, lr)
        
        # Foward propagation and Calculate loss
        logits = model(data)
        
        global_cfg['loss']['model'] = model
        global_cfg['loss']['data'] = data
        loss_dict = loss_func(logits, targets, global_cfg['loss'])
        loss = loss_dict['loss']

        
        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate classifier error about in-distribution sample
        num_topks_correct = metrics.topks_correct(logits[:len(targets)], targets, (1,))
        [top1_correct] = [x for x in num_topks_correct]
        
        # Add additional metrics!!!
        
        loss, top1_correct = loss.item(), top1_correct.item()
        avg_loss += loss
        correct += top1_correct
    
    summary = {
        'avg_loss': avg_loss / in_data_size,
        'classifier_acc': correct / in_data_size,
        'lr': get_lr_at_epoch(op_cfg, cur_epoch),
        'epoch': cur_epoch,
    }
    
    return summary
  
    
def valid_epoch_wo_outlier(model, in_loader, loss_func, cur_epoch):
    global global_cfg
    model.eval()
    avg_loss = 0
    correct = 0
    in_data_size = len(in_loader.dataset)
    for cur_iter, in_set in enumerate(in_loader):        
        # Data to GPU
        data = in_set[0]
        targets = in_set[1]
        data, targets = data.cuda(), targets.cuda()
        
        # Foward propagation and Calculate loss
        logits = model(data)

        global_cfg['loss']['model'] = model
        global_cfg['loss']['data'] = data
        loss_dict = loss_func(logits, targets, global_cfg['loss'])
        loss = loss_dict['loss']
        
        # Calculate classifier error about in-distribution sample
        num_topks_correct = metrics.topks_correct(logits[:len(targets)], targets, (1,))
        [top1_correct] = [x for x in num_topks_correct]
        
        # Add additional metrics!!
        
        
        loss, top1_correct = loss.item(), top1_correct.item()
        avg_loss += loss
        correct += top1_correct
    
    summary = {
        'avg_loss': avg_loss / in_data_size,
        'classifier_acc': correct / in_data_size,
        'epoch': cur_epoch,
    }
    
    return summary
    


def train_epoch_w_outlier(model, optimizer, in_loader, out_loader, loss_func, detector_func, cur_epoch, op_cfg, writer):
    global global_cfg
    model.train()
    avg_loss = 0
    correct = 0
    total = 0
    in_data_size = len(in_loader.dataset)
    out_loader.dataset.offset = np.random.randint(len(out_loader.dataset))
    for cur_iter, (in_set, out_set) in enumerate(zip(in_loader, out_loader)):
        #TODO: Dimension of in_set and out_set should be checked!
        
        # Data to GPU
        data = torch.cat((in_set[0], out_set[0]), 0)
        targets = in_set[1]
        if cur_iter == 0:
            writer.add_image('in_dist sample, target:[{}]'.format(targets[0]), in_set[0][0], cur_epoch)
            writer.add_image('out_dist sample', out_set[0][0], cur_epoch)
        data, targets = data.cuda(), targets.cuda()
        
        # Adjust Learning rate
        lr = optim.get_lr_at_epoch(op_cfg, cur_epoch + float(cur_iter) / in_data_size)
        optim.set_lr(optimizer, lr)
        
        # Foward propagation and Calculate loss and confidence
        logits = model(data)
        global_cfg['loss']['model'] = model
        global_cfg['loss']['data'] = data
        global_cfg['detector']['model'] = model
        global_cfg['detector']['data'] = data
        loss_dict = loss_func(logits, targets, global_cfg['loss'])
        loss = loss_dict['loss']
        confidences_dict = detector_func(logits, targets, global_cfg['detector'])
        confidences = confidences_dict['confidences']

        
        # Back propagation 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## METRICS ##
        # Calculate classifier error about in-distribution sample
        num_topks_correct = metrics.topks_correct(logits[:len(targets)], targets, (1,))
        [top1_correct] = [x for x in num_topks_correct]
        
        # Calculate OOD metrics (auroc, aupr, fpr)
        #(auroc, aupr, fpr) = metrics.get_ood_measures(confidences, targets)
        
        # Add additional metrics!!!
        
        
        ## UDATE STATS ##
        loss, top1_correct = loss.item(), top1_correct.item()
        avg_loss += loss
        correct += top1_correct
        total += targets.size(0)
    
    summary = {
        'avg_loss': avg_loss / total,
        'classifier_acc': correct / total,
        'lr': optim.get_lr_at_epoch(op_cfg, cur_epoch),
        'epoch': cur_epoch,
    }
    
    return summary
  
    
def valid_epoch_w_outlier(model, in_loader, out_loader, loss_func, detector_func, cur_epoch):
    global global_cfg  
    model.eval()
    avg_loss = 0
    correct = 0
    total = 0
    max_iter = 0
    avg_auroc = 0
    avg_aupr = 0
    avg_fpr = 0
    in_data_size = len(in_loader.dataset)
    for cur_iter, (in_set, out_set) in enumerate(zip(in_loader, out_loader)):        
        # Data to GPU
        data = torch.cat((in_set[0], out_set[0]), 0)
        targets = in_set[1]
        data, targets = data.cuda(), targets.cuda()
        
        # Foward propagation and Calculate loss and confidence
        logits = model(data)
        global_cfg['loss']['model'] = model
        global_cfg['loss']['data'] = data
        global_cfg['detector']['model'] = model
        global_cfg['detector']['data'] = data
        loss_dict = loss_func(logits, targets, global_cfg['loss'])
        loss = loss_dict['loss']
        confidences_dict = detector_func(logits, targets, global_cfg['detector'])
        confidences = confidences_dict['confidences']
        
        ## METRICS ##
        # Calculate classifier error about in-distribution sample
        num_topks_correct = metrics.topks_correct(logits[:len(targets)], targets, (1,))
        [top1_correct] = [x for x in num_topks_correct]
        
        # Calculate OOD metrics (auroc, aupr, fpr)
        (auroc, aupr, fpr) = metrics.get_ood_measures(confidences, targets)
        
        # Add additional metrics!!!
        
        ## Update stats ##
        loss, top1_correct = loss.item(), top1_correct.item()
        avg_loss += loss
        correct += top1_correct
        total += targets.size(0)
        max_iter += 1
        avg_auroc += auroc
        avg_aupr += aupr
        avg_fpr += fpr
        
    
    summary = {
        'avg_loss': avg_loss / total,
        'classifier_acc': correct / total,
        'AUROC': avg_auroc / max_iter,
        'AUPR' : avg_aupr / max_iter,
        'FPR95': avg_fpr / max_iter,
        'epoch': cur_epoch,
    }
    
    return summary
    

def getOptimizer(model, cfg):
    if cfg['optimizer'] == 'sgd':
        return torch.optim.SGD(model.parameters(),
                               lr=cfg['lr'],
                               momentum=cfg['momentum'],
                               weight_decay=cfg['weight_decay'],
                               nesterov=cfg['nesterov'])
    elif cfg['optimizer'] == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg['lr'],
            betas=(0.9, 0.999),
            weight_decay=cfg['weight_decay']
        )
    else:
        raise NotImplementedError(
            "Does not support '{}' optimizer".format(cfg['optimizer'])
        )

def main(cfg):
    global global_cfg
    # Reproducibility
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    
    # Model & Optimizer

    model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    }

    def _resnet(arch, block, num_classes, layers, pretrained, progress, **kwargs):
        model = ResNet(block, layers, num_classes, **kwargs)
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(model_urls[arch],
                                                progress=progress)
            state_dict['fc.weight']=state_dict['fc.weight'].data[range(num_classes),:]
            state_dict['fc.bias']=state_dict['fc.bias'].data[range(num_classes)]
            state_dict['conv1.weight']=state_dict['conv1.weight'].data[:,:,2:5,2:5]
            model.load_state_dict(state_dict)
        return model

    def resnet34(pretrained=1, num_classes=cfg['in_dataset']['num_classes'],progress=True, **kwargs):
        if pretrained==1:
            pretrained=True
            print("*** Pre Trained : True")
        else: 
            pretrained=False
            print("*** Pre Trained : False")
        return _resnet('resnet34', BasicBlock, num_classes, [3, 4, 6, 3], pretrained, progress,
                    **kwargs)
        # model = ResNet_128(BasicBlock, [3,4,6,3], num_classes=cfg['in_dataset']['num_classes'])
    
    model=resnet34(pretrained=args.pre_trained)
    model.cuda()
    print(model)

    max_epoch = args.max_epoch
    optimizer_option = dict() 
    optimizer_option['max_epoch'] = max_epoch
    optimizer_option['optimizer'] = 'sgd'
    optimizer_option['weight_decay'] = 0.0
    optimizer_option['nesterov'] = True
    optimizer_option['lr'] = 0.01
    optimizer_option['momentum'] = 0.9
    optimizer_option['policy'] = 'cosine'
    optimizer_option['warm_epoch'] = 0
    optimizer_option['warm_lr'] = 0.0
    optimizer = getOptimizer(model, optimizer_option)

    start_epoch = 1
    print('Model and Optimizer Loaded')
    # Load model and optimizer
    # if cfg['load_ckpt'] != '':
    #     checkpoint = torch.load(cfg['load_ckpt'], map_location="cpu")

    #     print("load model on '{}' is complete.".format(cfg['load_ckpt']))
    #     if not cfg['finetuning']:
    #         optimizer.load_state_dict(checkpoint['optimizer_state'])
    #     if 'epoch' in checkpoint.keys() and not cfg['finetuning']:
    #         start_epoch = checkpoint['epoch']
    #         print("Restore epoch {}".format(start_epoch))
    #     else:
    #         start_epoch = 1
    cudnn.benchmark = True
    
    # Data Loader
    print("Loading Training Data")
    in_train_loader = getDataLoader(ds_cfg=cfg['in_dataset'],
                                    dl_cfg=cfg['dataloader'],
                                    split="train")
    print("Loading Validation Data")
    in_valid_loader = getDataLoader(ds_cfg=cfg['in_dataset'],
                                    dl_cfg=cfg['dataloader'],
                                    split="valid")
        
    # Result directory and make tensorboard event file
    exp_dir = os.path.join('pre_trained',args.experiment)
    if os.path.exists(exp_dir) is False:
        os.makedirs(exp_dir)
    writer_train = SummaryWriter(logdir=os.path.join(exp_dir, 'log', 'train'))
    writer_valid = SummaryWriter(logdir=os.path.join(exp_dir, 'log', 'valid'))
    
    # Stats Meters
    #train_meter = TrainMeter()
    #valid_meter = ValidMeter()x
    
    # Loss function
    print("Loading Loss Function")
    def cross_entropy_in_distribution(logits, targets, cfg):
        """
        Cross entropy loss when logits include outlier's logits also.(ignore outlier's)
        """
        return {
            'loss': F.cross_entropy(logits[:len(targets)], targets),
        }

    loss_func = cross_entropy_in_distribution
    global_cfg['loss'] = cfg['loss']

    
    # Outlier detector
    # detector_func = detectors.getDetector(cfg['detector'])
    # global_cfg['detector'] = cfg['detector']
    
    print("============Start training. Result will be saved in {}".format(exp_dir))
    
    for cur_epoch in range(start_epoch, max_epoch + 1):
        train_summary = train_epoch_wo_outlier(model, optimizer, in_train_loader, loss_func, cur_epoch, optimizer_option, writer_train)
        summary_write(summary=train_summary, writer=writer_train)
        print("Training result=========Epoch [{}]/[{}]=========\nlr: {} | loss: {} | acc: {}".format(cur_epoch, args.max_epoch, train_summary['lr'], train_summary['avg_loss'], train_summary['classifier_acc']))
        
        
        if cur_epoch % cfg['valid_epoch'] == 0:
            valid_summary = valid_epoch_wo_outlier(model, in_valid_loader, loss_func, cur_epoch)
            summary_write(summary=valid_summary, writer=writer_valid)
            print("Validate result=========Epoch [{}]/[{}]=========\nloss: {} | acc: {}".format(cur_epoch, args.max_epoch, valid_summary['avg_loss'], valid_summary['classifier_acc']))
        
        if cur_epoch % cfg['ckpt_epoch'] == 0:
            ckpt_dir = os.path.join('pre_trained',args.experiment,"ckpt")
            if os.path.exists(ckpt_dir) is False:
                os.makedirs(ckpt_dir)
            model_state = model.module.state_dict() if cfg['ngpu'] > 1 else model.state_dict()
            checkpoint = {
                "epoch": cur_epoch,
                "model_state": model_state,
                "optimizer_state": optimizer.state_dict(),
            }
            ckpt_name = "checkpoint_epoch_{}".format(cur_epoch)
            ckpt_path = os.path.join(ckpt_dir, ckpt_name + ".pyth")
            torch.save(checkpoint, ckpt_path)
        

if __name__=="__main__":
    print("Setup Training...")

    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',help='Dataset')
    parser.add_argument('--out_target', help='OOD Target')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--valid_epoch',type=int, default=1)
    parser.add_argument('--experiment')
    parser.add_argument('--max_epoch',type=int)
    parser.add_argument('--pre_trained',type=int, default=0)
    args=parser.parse_args()
    
    cfg = dict()
    cfg['mean']=[0,4214,0.4214,0.4214]
    cfg['std']=[0.2355,0.2355,0.2355]

    cfg['in_dataset']=dict()
    cfg['in_dataset']['dataset']=args.dataset
    cfg['in_dataset']['batch_size']=args.batch_size
    if 'DAGM' in args.dataset:
        x=['1','2','3','4','5','6','7','8','9','10']
        x.remove(str(args.out_target))
    elif args.dataset=='Severstal':
        x=['ok','1','2','3','4']
        x.remove(str(args.out_target))
    elif args.dataset=='SDI/34Ah':
        x=['ok','1','2','3','4','5','6','7']
        x.remove(str(args.out_target))
    elif args.dataset=='SDI/37Ah':
        x=['ok','1','2','3','4','5','6']
        x.remove(str(args.out_target))
    elif args.dataset=='SDI/60Ah':
        x=['ok','1','2','5','6','7','8']
        x.remove(str(args.out_target))
    elif args.dataset=='Traffic':
        x=list(range(0,43))
        x.remove(int(args.out_target))
    elif args.dataset=='MNIST':
        x=list(range(0,10))
        x.remove(int(args.out_target))
    elif args.dataset=='mvtec_leather':
        x=['color','cut','fold','glue','good_from_train','poke']
        x.remove(str(args.out_target))
    elif args.dataset=='SVHN':
        x=list(range(1,11))
        x.remove(int(args.out_target))

    

    cfg['in_dataset']['targets']=x
    cfg['in_dataset']['train_transform']=trn.Compose([trn.RandomHorizontalFlip(),trn.ToTensor()])
    cfg['in_dataset']['valid_transform']=trn.Compose([trn.ToTensor()])
    cfg['in_dataset']['data_root']='/HDD0/Saige_Database/ETC/'+args.dataset+'/images'
    cfg['in_dataset']['split_root']='/HDD0/Openset/data_split/'+args.dataset
    cfg['in_dataset']['num_classes']=len(cfg['in_dataset']['targets'])

    cfg['dataloader'] = dict()
    cfg['dataloader']['num_workers'] = 4
    cfg['dataloader']['pin_memory'] = True


    cfg['valid_epoch']=args.valid_epoch
    cfg['ckpt_epoch']=20

    cfg['ngpu']=1
    cfg['seed']=0

    cfg['loss'] = dict()
    cfg['loss']['loss'] = 'cross_entropy_in_distribution'

    print("In-distribution targets: ", cfg['in_dataset']['targets'])
    print("Num-Classes : {}".format(cfg['in_dataset']['num_classes']))
    
    main(cfg)