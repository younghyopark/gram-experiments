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

def G_p(ob, p):
    temp = ob.detach()
    
    temp = temp**p
    temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
    temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1)))).sum(dim=2) 
    temp = (temp.sign()*torch.abs(temp)**(1/p)).reshape(temp.shape[0],-1)
    
    return Variable(temp,requires_grad=True)

def G_p_entire(ob, p):
    temp = ob.detach()
    temp = temp**p
    temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
    temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1))))
    temp = (temp.sign()*torch.abs(temp)**(1/p)).reshape(temp.shape[0],temp.shape[1],-1)
    return temp


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
    def __init__(self, split, targets, size):
        """
            data_root(str) : Root directory of datasets (e.g. "/home/sr2/HDD2/Openset/")
            split_root(str) : Root directroy of split file (e.g. "/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/")
            dataset(str) : dataset name
            split(str) : ['train', 'valid', 'test']
            transform(torchvision transform) : image transform
            targets(list of str) : using targets
        """
        self.data_root = './Imagenet_flickr/imagenet_images/'
        self.split_root = './data_split/LargeCIFAR10/'
        if size=='large':
            self.transform = trn.Compose([trn.Resize([256,256]),trn.CenterCrop([224,224]),trn.ToTensor()])
        else:
            self.transform = trn.Compose([trn.Resize([36,36]),trn.CenterCrop([32,32]),trn.ToTensor()])

        self.targets = targets
        
        self.data_list = []
        f = open(os.path.join(self.split_root, split + ".txt"), "r")
        
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
    elif ds_cfg['dataset'] in ['LargeCIFAR10']:
        dataset = LargeCIFAR10Dataset(split=split,targets=ds_cfg['targets'],size=args.size)
        number= dataset.__len__()
        loader = DataLoader(dataset,
                            batch_size=ds_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset {} ready.'.format(ds_cfg['dataset']))

    elif ds_cfg['dataset'] in ['cifar10']:
        batch_size = args.batch_size
        mean = np.array([[0.4914, 0.4822, 0.4465]]).T

        std = np.array([[0.2023, 0.1994, 0.2010]]).T
        normalize = trn.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        transform_train = trn.Compose([
                trn.RandomCrop(32, padding=4),
                trn.RandomHorizontalFlip(),
                trn.ToTensor(),
                normalize
                
            ])

        transform_test = trn.Compose([
                trn.CenterCrop(size=(32, 32)),
                trn.ToTensor(),
                normalize
            ])

        if split=='train':
            loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('data', train=True, download=True,
                            transform=transform_train),
                batch_size=batch_size, shuffle=True)
        else:
            loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('data', train=False, transform=transform_test),
                batch_size=batch_size)
        return loader

    elif ds_cfg['dataset'] in ['svhn']:
        batch_size = args.batch_size
        mean = np.array([[0.4914, 0.4822, 0.4465]]).T

        std = np.array([[0.2023, 0.1994, 0.2010]]).T
        normalize = trn.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        transform_train = trn.Compose([
                trn.RandomCrop(32, padding=4),
                trn.RandomHorizontalFlip(),
                trn.ToTensor(),
                normalize
                
            ])
        transform_test = trn.Compose([
            trn.CenterCrop(size=(32, 32)),
                trn.ToTensor(),
                normalize
            ])

        if split=='train':
            loader = torch.utils.data.DataLoader(
                datasets.SVHN('data', split="train", download=True,
                            transform=transform_train),
                batch_size=batch_size, shuffle=True)
        else:
            loader = torch.utils.data.DataLoader(
                datasets.SVHN('data', split="test", download=True, transform=transform_test),
                batch_size=batch_size)
        return loader

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

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    
    def forward(self, x):
        t = self.conv1(x)
        out = F.relu(self.bn1(t))
        model.record(t)
        model.record(out)
        t = self.conv2(out)
        out = self.bn2(self.conv2(out))
        model.record(t)
        model.record(out)
        t = self.shortcut(x)
        out += t
        model.record(t)
        out = F.relu(out)
        model.record(out)
        
        return out#, out_list

class ResNet_128(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_128, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        self.collecting = False
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y
    
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
    
    def pre_trained_load(self):
        print('Imagenet Pretrained model Loading!')
        state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth')
        # tm = torch.load(path,map_location="cpu")        
        self.load_state_dict(state_dict)
        print('***Success!***')
    
    def get_min_max(self, data, power):
        mins = []
        maxs = []
        which_layer=1
        for i in range(0,len(data),256):
            batch = data[i:i+256].cuda()
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
        batch = data.cuda()
        feat_list = self.gram_feature_list(batch)

        for L,feat_L in enumerate(feat_list):
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
        which_layer=1
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

def summary_write(summary, writer):
    for key in summary.keys():
        writer.add_scalar(key, summary[key], summary['epoch'])
        

def train_epoch_wo_outlier(model, optimizer, optimizer_2,  in_loader, loss_func, cur_epoch, op_cfg, op_cfg_2, writer):
    global global_cfg
    model.train()
    avg_loss = 0
    avg_celoss =0
    avg_maxminloss=0
    correct = 0
    in_data_size = len(in_loader.dataset)
    for cur_iter, in_set in enumerate(tqdm(in_loader)):
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
        
        lr_2 = get_lr_at_epoch(op_cfg_2, cur_epoch + float(cur_iter) / in_data_size)
        set_lr(optimizer_2, lr_2)

        # Foward propagation and Calculate loss
        logits = model(data)
        mins,maxs=model.get_min_max(data,range(1,11))
        
        global_cfg['loss']['model'] = model
        global_cfg['loss']['data'] = data
        loss_dict = loss_func(logits, targets, global_cfg['loss'], mins, maxs)
        
        if cur_epoch%2 ==0:
            loss = loss_dict['celoss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate classifier error about in-distribution sample
            num_topks_correct = metrics.topks_correct(logits[:len(targets)], targets, (1,))
            [top1_correct] = [x for x in num_topks_correct]

            # Add additional metrics!!!

            loss, celoss, maxminloss, top1_correct = loss.item(), loss_dict['celoss'].item(), loss_dict['maxminloss'].item(), top1_correct.item()
            avg_loss += loss
            avg_celoss += celoss
            avg_maxminloss += maxminloss
            correct += top1_correct

        
            summary = {
                    'avg_loss': avg_loss / in_data_size,
                    'avg_celoss': avg_celoss / in_data_size,
                    'avg_maxminloss': avg_maxminloss / in_data_size,
                    'classifier_acc': correct / in_data_size,
                    'lr': get_lr_at_epoch(op_cfg, cur_epoch),
                    'epoch': cur_epoch,
                }

        else:
            loss = loss_dict['maxminloss']
            optimizer_2.zero_grad()
            loss.backward()
            optimizer_2.step()
            
            # Calculate classifier error about in-distribution sample
            num_topks_correct = metrics.topks_correct(logits[:len(targets)], targets, (1,))
            [top1_correct] = [x for x in num_topks_correct]

            # Add additional metrics!!!

            loss, celoss, maxminloss, top1_correct = loss.item(), loss_dict['celoss'].item(), loss_dict['maxminloss'].item(), top1_correct.item()
            avg_loss += loss
            avg_celoss += celoss
            avg_maxminloss += maxminloss
            correct += top1_correct

            
            summary = {
                    'avg_loss': avg_loss / in_data_size,
                    'avg_celoss': avg_celoss / in_data_size,
                    'avg_maxminloss': avg_maxminloss / in_data_size,
                    'classifier_acc': correct / in_data_size,
                    'lr': get_lr_at_epoch(op_cfg_2, cur_epoch),
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
        mins,maxs=model.get_min_max(data,range(1,11))

        global_cfg['loss']['model'] = model
        global_cfg['loss']['data'] = data
        loss_dict = loss_func(logits, targets, global_cfg['loss'],mins,maxs)
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

def main(model, cfg):
    global global_cfg
    # Reproducibility
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    
    # Model & Optimizer
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
    
    optimizer_option_2 = dict() 
    optimizer_option_2['max_epoch'] = max_epoch
    optimizer_option_2['optimizer'] = 'sgd'
    optimizer_option_2['weight_decay'] = 0.0
    optimizer_option_2['nesterov'] = True
    optimizer_option_2['lr'] = 0.1
    optimizer_option_2['momentum'] = 0.9
    optimizer_option_2['policy'] = 'cosine'
    optimizer_option_2['warm_epoch'] = 0
    optimizer_option_2['warm_lr'] = 0.0
    optimizer_2 = getOptimizer(model, optimizer_option_2)


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
    #valid_meter = ValidMeter()
    
    # Loss function
    print("Loading Loss Function")
    def cross_entropy_in_distribution(logits, targets, cfg):
        """
        Cross entropy loss when logits include outlier's logits also.(ignore outlier's)
        """
        
        return {
            'loss': F.cross_entropy(logits[:len(targets)], targets),
        }

    def maxmin_tight_clustering(logits, targets, cfg, maxs,mins):
        """
        Cross entropy loss when logits include outlier's logits also.(ignore outlier's)
        """
#         loss=0
        loss = 0
        for l in range(96):
            for p in range(10):
                difference = torch.norm(maxs[l][p]-mins[l][p],p=2)
                loss+=difference/960
        return {
            'loss': loss/10+F.cross_entropy(logits[:len(targets)],targets),
            'celoss': F.cross_entropy(logits[:len(targets)],targets),
            'maxminloss': loss
        }

    loss_func = maxmin_tight_clustering
    global_cfg['loss'] = cfg['loss']

    
    # Outlier detector
    # detector_func = detectors.getDetector(cfg['detector'])
    # global_cfg['detector'] = cfg['detector']
    
    print("============Start training. Result will be saved in {}".format(exp_dir))
    
    for cur_epoch in range(start_epoch, max_epoch + 1):
        train_summary = train_epoch_wo_outlier(model, optimizer, optimizer_2, in_train_loader, loss_func, cur_epoch, optimizer_option, optimizer_option_2, writer_train)
        summary_write(summary=train_summary, writer=writer_train)
        print("Training result=========Epoch [{}]/[{}]=========\nlr: {} | loss: {} | ce loss : {} | maxmin loss : {} |acc: {}".format(cur_epoch, args.max_epoch, train_summary['lr'], train_summary['avg_loss'],train_summary['avg_celoss'],train_summary['avg_maxminloss'],train_summary['classifier_acc']))
        
        
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
    import time


    print("Setup Training...")

    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',help='Dataset')
    parser.add_argument('--out_target', help='OOD Target')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--valid_epoch',type=int, default=1)
    parser.add_argument('--experiment')
    parser.add_argument('--max_epoch',type=int)
    parser.add_argument('--pre_trained',type=int, default=0)
    parser.add_argument('--size',type=str, default=0)
    parser.add_argument('--hyper', default=0)

    args=parser.parse_args()
    
    cfg = dict()
    cfg['mean']=[0,4214,0.4214,0.4214]
    cfg['std']=[0.2355,0.2355,0.2355]

    cfg['in_dataset']=dict()
    cfg['in_dataset']['dataset']=args.dataset
    cfg['in_dataset']['batch_size']=args.batch_size

    lmbda = args.hyper

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
    elif args.dataset=='LargeCIFAR10':
        x=['airplane','bird','car','cat','deer','dog','frog','horse','ship','truck']
        x.remove(str(args.out_target))
    else:
        x=None

    cfg['in_dataset']['targets']=x
    cfg['in_dataset']['train_transform']=trn.Compose([trn.RandomHorizontalFlip(),trn.ToTensor()])
    cfg['in_dataset']['valid_transform']=trn.Compose([trn.ToTensor()])
    cfg['in_dataset']['data_root']='/HDD0/Saige_Database/ETC/'+args.dataset+'/images'
    cfg['in_dataset']['split_root']='/HDD0/Openset/data_split/'+args.dataset
    if x is not None : cfg['in_dataset']['num_classes']=len(cfg['in_dataset']['targets'])
    else: cfg['in_dataset']['num_classes']=10

    cfg['dataloader'] = dict()
    cfg['dataloader']['num_workers'] = 8
    cfg['dataloader']['pin_memory'] = True


    cfg['valid_epoch']=args.valid_epoch
    cfg['ckpt_epoch']=1 

    cfg['ngpu']=1
    cfg['seed']=0

    cfg['loss'] = dict()
    cfg['loss']['loss'] = 'maxmin_tight_clustering'

    print("In-distribution targets: ", cfg['in_dataset']['targets'])
    print("Num-Classes : {}".format(cfg['in_dataset']['num_classes']))
    
    model = ResNet_128(BasicBlock, [3,4,6,3], num_classes=cfg['in_dataset']['num_classes'])
    if args.pre_trained == 1:
        tm = torch.load('resnet_svhn.pth',map_location='cpu')
        model.load_state_dict(tm)
        print("loaded pretrained model")
    model.cuda()

    main(model, cfg)