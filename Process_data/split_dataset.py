import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,ConcatDataset
from torch.autograd import Variable
import json
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#from skimage import io, transform
from PIL import Image

from collections import Counter
# visualizing data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

# load dataset information~
import yaml

# image writing
import imageio
from skimage import img_as_ubyte


from scipy.io import loadmat
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

parser = argparse.ArgumentParser("ESFPNet based model")
parser.add_argument('--label_json_path', type=str, required=True,
        help='Location of the data directory containing json labels file of each task after combining two json files.json')
parser.add_argument('--path_cancer_imgs', type=str, required=True,
        help='Location of the images of cancer cases)')
parser.add_argument('--path_non_cancer_imgs', type=str, required=True,
        help='Location of the images of non cancer cases)')
parser.add_argument('--path_cancer_masks', type=str, required=True,
        help='Location of the masks of cancer cases for each tasks)')
parser.add_argument('--path_non_cancer_masks', type=str, required=True,
        help='Location of the masks of non cancer cases for each tasks)')
parser.add_argument('--path_dataset', type=str, required=True,
        help='Location of the directory to store the dataset')
parser.add_argument('--n_epochs', type=int, default=500,
        help='Number of epochs for training (default = 500)')
parser.add_argument('--task', type=str, required=True,
        help='Task: Anatomical_landmarks or Lung_lesions')
args = parser.parse_args()

torch.cuda.empty_cache()


class SplittingDataset(Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root):

        with open(args.label_json_path, 'r') as f:
            data = json.load(f)

        object_id = [id['object_id'] for id in data]
        
        self.images = []
        
        for root, dirs, files in os.walk(image_root):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    if os.path.splitext(os.path.basename(os.path.join(root, file)))[0] in object_id:
                        self.images.append(os.path.join(root, file))
        
        self.gts = []

        for root, dirs, files in os.walk(gt_root):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    if os.path.splitext(os.path.basename(os.path.join(root, file)))[0] in object_id:
                        self.gts.append(os.path.join(root, file))

        self.images = [file for file in self.images if file.replace('/imgs/', '/masks_' + args.task + '/') in self.gts]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        name_image = self.images[index].split('/')[-1]


        file_name = os.path.splitext(os.path.basename(self.images[index]))[0]
        
        with open(args.label_json_path, 'r') as f:
            data = json.load(f)

        if args.task == 'Anatomical_landmarks':
            label_list = ['Vocal cords', 'Main carina', 'Intermediate bronchus', 'Right superior lobar bronchus', 'Right inferior lobar bronchus', 'Right middle lobar bronchus', 'Left inferior lobar bronchus', 'Left superior lobar bronchus', 'Right main bronchus', 'Left main bronchus', 'Trachea']
            label_name = [file['label_name'] for file in data if file['object_id'] == file_name]
            
            label_tensor = torch.zeros([11])
            for name in label_name:
                label_tensor[label_list.index(name)] = 1
        else: 
            label_list = ['Muscosal erythema', 'Anthrocosis', 'Stenosis', 'Mucosal edema of carina', 'Mucosal infiltration', 'Vascular growth', 'Tumor']
            label_name = [file['label_name'] for file in data if file['object_id'] == file_name]
            
            label_tensor = torch.zeros([7])
            for name in label_name:
                label_tensor[label_list.index(name)] = 1
                
        str_label = str(label_tensor)

        return self.transform(image), self.transform(gt), str_label, name_image

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
    

def splitDataset():
    split_train_images_save_path = args.path_dataset + '/dataset/' + args.task + '/train/imgs'
    os.makedirs(split_train_images_save_path, exist_ok=True)
    split_train_masks_save_path = args.path_dataset + '/dataset/' + args.task + '/train/masks'
    os.makedirs(split_train_masks_save_path, exist_ok=True)
    
    split_validation_images_save_path = args.path_dataset + '/dataset/' + args.task + '/val/imgs'
    os.makedirs(split_validation_images_save_path, exist_ok=True)
    split_validation_masks_save_path = args.path_dataset + '/dataset/' + args.task + '/val/masks'
    os.makedirs(split_validation_masks_save_path, exist_ok=True)
    
    split_test_images_save_path = args.path_dataset + '/dataset/' + args.task + '/test/imgs'
    os.makedirs(split_test_images_save_path, exist_ok=True)
    split_test_masks_save_path = args.path_dataset + '/dataset/' + args.task + '/test/masks'
    os.makedirs(split_test_masks_save_path, exist_ok=True)
    
    DatasetList = []
    
    images_train_path_1 = Path(args.path_cancer_imgs)
    masks_train_path_1 = Path(args.path_cancer_masks)
    Dataset_part_train_1 = SplittingDataset(images_train_path_1, masks_train_path_1)
    DatasetList.append(Dataset_part_train_1)
    
    images_train_path_2 = Path(args.path_non_cancer_imgs)
    masks_train_path_2 = Path(args.path_non_cancer_masks)
    Dataset_part_train_2 = SplittingDataset(images_train_path_2, masks_train_path_2)
    DatasetList.append(Dataset_part_train_2)

    wholeDataset = ConcatDataset([DatasetList[0], DatasetList[1]])

    imgs_list = []
    masks_list = []
    labels_list = []
    names_list = []

    for iter in list(wholeDataset):
        imgs_list.append(iter[0])
        masks_list.append(iter[1])
        labels_list.append(iter[2])
        names_list.append(iter[3])

    element_counts = {}

    for element in labels_list:
        if element in element_counts:
            element_counts[element] += 1
        else:
            element_counts[element] = 1

    removed_elements = []

    Y_data = [element for element in labels_list if element_counts[element] >= 5]

    for element in labels_list:
        if element_counts[element] < 5:
            removed_elements.append(element)

    combine_list = list(zip(imgs_list, masks_list, names_list, labels_list))
    
    X_data = [tup for tup in combine_list if not any(item in removed_elements for item in tup)]

    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.1, 
                                                        random_state=42, stratify = Y_data) #
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, 
                                                        random_state=42, stratify = y_train) #
    
    for i in X_train:
        image, gt, name, str_label = i[0], i[1], i[2], i[3]
        image_data = image.data.cpu().numpy().squeeze().transpose(1,2,0)
        gt_data = gt.data.cpu().numpy().squeeze()
        imageio.imwrite(split_train_images_save_path + '/' + name,img_as_ubyte(image_data))
        imageio.imwrite(split_train_masks_save_path + '/' + name, img_as_ubyte(gt_data))
    
    for i in X_val:
        image, gt, name, str_label = i[0], i[1], i[2], i[3]
        image_data = image.data.cpu().numpy().squeeze().transpose(1,2,0)
        gt_data = gt.data.cpu().numpy().squeeze()
        imageio.imwrite(split_validation_images_save_path + '/' + name,img_as_ubyte(image_data))
        imageio.imwrite(split_validation_masks_save_path + '/' + name, img_as_ubyte(gt_data))

    for i in X_test:
        image, gt, name, str_label = i[0], i[1], i[2], i[3]
        image_data = image.data.cpu().numpy().squeeze().transpose(1,2,0)
        gt_data = gt.data.cpu().numpy().squeeze()
        imageio.imwrite(split_test_images_save_path + '/' + name,img_as_ubyte(image_data))
        imageio.imwrite(split_test_masks_save_path + '/' + name, img_as_ubyte(gt_data))
             
splitDataset()
