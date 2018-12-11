import os
import os.path as osp
import numpy as np
from glob import glob
from PIL import Image
#from tqdm import tqdm
import torch
#import cPickle as pickle - MWB: couldn't install cPickle
#import pickle
from torchvision import transforms
# from torchvision.transforms import Compose
from transform import ToTensor, RangeNormalize, RandomFlip, Compose, CenterCrop
#import collections
#import random

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, root):
        self.size = (192,192)
        self.root = root
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(root))
        self.img_transform = Compose([
            ToTensor(),
            CenterCrop(self.size),
            # RangeNormalize(min_val=-1,max_val=1),
            # RandomFlip(),
        ])
        #sort file names
        self.input_paths = sorted(glob(os.path.join(self.root, '{}/*.npy'.format("train"))))
        self.name = os.path.basename(root)
        if len(self.input_paths) == 0:
            raise Exception("No inputs are found in {}".format(self.root))
        # imgsets_dir = osp.join(root, "ImageSets/Segmentation/%s.txt" % split)
        # self.files = collections.defaultdict(list)
        # with open(imgsets_dir) as imgset_file:
        #     for name in imgset_file:
        #         name = name.strip()
        #         img_file = osp.join(root, "JPEGImages/%s.jpg" % name)
        #         label_file = osp.join(root, "SegmentationClass/%s.png" % name)
        #         self.files[split].append({
        #             "img": img_file,
        #             "label": label_file
        #         })
        # ])

    def __getitem__(self, index):
        image = np.load(self.input_paths[index])
        # image = np.float32(image)
        # print(np.mean(image))
        image = self.img_transform(image)

        return image

    def __len__(self):
        return len(self.input_paths)


class Dataset_val(torch.utils.data.Dataset):
    def __init__(self, root):
        self.size = (192,192)
        self.root = root
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(root))
        self.img_transform = Compose([
            ToTensor(),
            CenterCrop(self.size),
            # RangeNormalize(min_val=-1,max_val=1),
            # RandomFlip(),
        ])
        #sort file names
        self.input_paths = sorted(glob(os.path.join(self.root, '{}/*.npy'.format("val"))))
        self.name = os.path.basename(root)
        if len(self.input_paths) == 0:
            raise Exception("No validations are found in {}".format(self.root))
        # imgsets_dir = osp.join(root, "ImageSets/Segmentation/%s.txt" % split)
        # self.files = collections.defaultdict(list)
        # with open(imgsets_dir) as imgset_file:
        #     for name in imgset_file:
        #         name = name.strip()
        #         img_file = osp.join(root, "JPEGImages/%s.jpg" % name)
        #         label_file = osp.join(root, "SegmentationClass/%s.png" % name)
        #         self.files[split].append({
        #             "img": img_file,
        #             "label": label_file
        #         })
        # ])

    def __getitem__(self, index):
        image = np.load(self.input_paths[index])
        # print(image)

        image = self.img_transform(image)

        return image

    def __len__(self):
        return len(self.input_paths)



def loader(dataset, batch_size, num_workers=7, shuffle=True):
    input_images = dataset

    input_loader = torch.utils.data.DataLoader(dataset=input_images,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers)
    # gt_loader = torch.utils.data.DataLoader(dataset=gt_image,
    #                                             batch_size=batch_size,
    #                                             shuffle=True,
    #                                             num_workers=num_workers)
    #input_loader.shape = input_image.shape
    #gt_loader.shape = gt_image.shape

    return input_loader#, gt_loader
#import torch.utils.data as utils

# my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])] # a list of numpy arrays
# my_y = [np.array([4.]), np.array([2.])] # another list of numpy arrays (targets)
#
# tensor_x = torch.stack([torch.Tensor(i) for i in my_x]) # transform to torch tensors
# tensor_y = torch.stack([torch.Tensor(i) for i in my_y])
#
# my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
# my_dataloader = utils.DataLoader(my_dataset) # create your dataloader
