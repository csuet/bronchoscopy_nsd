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

import wandb
wandb.login()
wandb.init(project="ESFPNetclassification_on_Lung_cancer_lesions")

# visualizing data
import matplotlib.pyplot as plt
import numpy as np
import warnings

# load dataset information~
import yaml

# image writing
import imageio
from skimage import img_as_ubyte

from sklearn.model_selection import train_test_split

from scipy.io import loadmat
import matplotlib.pyplot as plt

# Clear GPU cache
torch.cuda.empty_cache()

# configuration


model_type = 'B4'

init_trainsize = 352
batch_size = 9

repeats = 1
n_epochs = 1000
if_renew = False

label_path = './labels/labels_Lung_cancer_lesions_final.json'

class SplittingDataset(Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root):

        with open(label_path, 'r') as f:
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
        self.images = [file for file in self.images if file.replace('/imgs/', '/masks/') in self.gts]
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        name = self.images[index].split('/')[-1]
        return self.transform(image), self.transform(gt), name

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

def splitDataset(renew):
    
    split_train_images_save_path = './dataset/Lung_cancer_lesions/train/imgs'
    os.makedirs(split_train_images_save_path, exist_ok=True)
    split_train_masks_save_path = './dataset/Lung_cancer_lesions/train/masks'
    os.makedirs(split_train_masks_save_path, exist_ok=True)
    
    split_validation_images_save_path = './dataset/Lung_cancer_lesions/val/imgs'
    os.makedirs(split_validation_images_save_path, exist_ok=True)
    split_validation_masks_save_path ='./dataset/Lung_cancer_lesions/val/masks'
    os.makedirs(split_validation_masks_save_path, exist_ok=True)
    
    split_test_images_save_path = './dataset/Lung_cancer_lesions/test/imgs'
    os.makedirs(split_test_images_save_path, exist_ok=True)
    split_test_masks_save_path = './dataset/Lung_cancer_lesions/test/masks'
    os.makedirs(split_test_masks_save_path, exist_ok=True)
    
    if renew == True:
    
        DatasetList = []

        images_train_path = '/workspace/ailab/phucnd/05092023_18_Nhom_benh/imgs'
        masks_train_path = '/workspace/ailab/phucnd/05092023_18_Nhom_benh/masks'
        Dataset_part_train = SplittingDataset(images_train_path, masks_train_path)
        
        imgs_list = []
        masks_list = []
        labels_list = []
        names_list = []

        for iter in list(Dataset_part_train):
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

        Y_data = [element for element in labels_list if element_counts[element] > 4]

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
            
    
    return split_train_images_save_path, split_train_masks_save_path, split_validation_images_save_path, split_validation_masks_save_path, split_test_images_save_path, split_test_masks_save_path

train_images_path, train_masks_path, val_images_path, val_masks_path, test_images_path, test_masks_path = splitDataset(if_renew)

class PolypDataset(Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root,label_root, trainsize, augmentations): #
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.labels = label_root

        self.gts = [gt_root + f for f in os.listdir(gt_root)  if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        if self.augmentations == True:
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0, hue=0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()
                ])
            
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            

    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        file_name = os.path.splitext(os.path.basename(self.images[index]))[0]
        
        with open(label_path, 'r') as f:
            data = json.load(f)

        label_list = ['Muscosal erythema', 'Anthrocosis', 'Stenosis', 'Mucosal edema of carina', 'Mucosal infiltration', 'Vascular growth', 'Tumor']
        label_name = [file['label_name'] for file in data if file['object_id'] == file_name]
        
        label_tensor = torch.zeros([7])
        for name in label_name:
            label_tensor[label_list.index(name)] = 1

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        np.random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        np.random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        
        return image, label_tensor

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

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

class test_dataset:
    def __init__(self, image_root, gt_root,label_root,  testsize): #
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png') or f.endswith('.jpg')]
        self.labels = label_root
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        file_name = os.path.splitext(os.path.basename(self.images[self.index]))[0]

        with open(label_path, 'r') as f:
            data = json.load(f)

        label_list = ['Muscosal erythema', 'Anthrocosis', 'Stenosis', 'Mucosal edema of carina', 'Mucosal infiltration', 'Vascular growth', 'Tumor']
        label_name = [file['label_name'] for file in data if file['object_id'] == file_name]
        
        label_tensor = torch.zeros([7])
        for name in label_name:
            label_tensor[label_list.index(name)] = 1
        

        self.index += 1
        return image, label_tensor

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        
from collections import OrderedDict
import copy

from Encoder import mit
from Decoder import mlp
from mmcv.cnn import ConvModule

class ESFPNetStructure(nn.Module):

    def __init__(self, embedding_dim = 160):
        super(ESFPNetStructure, self).__init__()

        # Backbone
        if model_type == 'B0':
            self.backbone = mit.mit_b0()
        if model_type == 'B1':
            self.backbone = mit.mit_b1()
        if model_type == 'B2':
            self.backbone = mit.mit_b2()
        if model_type == 'B3':
            self.backbone = mit.mit_b3()
        if model_type == 'B4':
            self.backbone = mit.mit_b4()
        if model_type == 'B5':
            self.backbone = mit.mit_b5()

        self._init_weights()  # load pretrain

        # LP Header
        self.LP_1 = mlp.LP(input_dim = self.backbone.embed_dims[0], embed_dim = self.backbone.embed_dims[0])
        self.LP_2 = mlp.LP(input_dim = self.backbone.embed_dims[1], embed_dim = self.backbone.embed_dims[1])
        self.LP_3 = mlp.LP(input_dim = self.backbone.embed_dims[2], embed_dim = self.backbone.embed_dims[2])
        self.LP_4 = mlp.LP(input_dim = self.backbone.embed_dims[3], embed_dim = self.backbone.embed_dims[3])

        # Linear Fuse
        self.linear_fuse34 = ConvModule(in_channels=(self.backbone.embed_dims[2] + self.backbone.embed_dims[3]), out_channels=self.backbone.embed_dims[2], kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse23 = ConvModule(in_channels=(self.backbone.embed_dims[1] + self.backbone.embed_dims[2]), out_channels=self.backbone.embed_dims[1], kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse12 = ConvModule(in_channels=(self.backbone.embed_dims[0] + self.backbone.embed_dims[1]), out_channels=self.backbone.embed_dims[0], kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))

        # Fused LP Header
        self.LP_12 = mlp.LP(input_dim = self.backbone.embed_dims[0], embed_dim = self.backbone.embed_dims[0])
        self.LP_23 = mlp.LP(input_dim = self.backbone.embed_dims[1], embed_dim = self.backbone.embed_dims[1])
        self.LP_34 = mlp.LP(input_dim = self.backbone.embed_dims[2], embed_dim = self.backbone.embed_dims[2])

        # Final Linear Prediction
        self.linear_pred = nn.Conv2d((self.backbone.embed_dims[0] + self.backbone.embed_dims[1] + self.backbone.embed_dims[2] + self.backbone.embed_dims[3]), 1, kernel_size=1)

        #classification layer
        self.norm1 = nn.BatchNorm2d(512, eps=1e-5)
        self.Relu = nn.ReLU(inplace=True)
        self.Dropout = nn.Dropout(p=0.3)
        self.conv1 = nn.Conv2d(512, 256, 1, stride=1, padding=0)
        self.norm2 = nn.BatchNorm2d(256, eps=1e-5)
        self.conv2 = nn.Conv2d(256, 7, 1, stride=1, padding=0, bias=True) # 9 = number of classes
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim = 1)

    def _init_weights(self):

        if model_type == 'B0':
            pretrained_dict = torch.load('./Pretrained/mit_b0.pth')
        if model_type == 'B1':
            pretrained_dict = torch.load('./Pretrained/mit_b1.pth')
        if model_type == 'B2':
            pretrained_dict = torch.load('./Pretrained/mit_b2.pth')
        if model_type == 'B3':
            pretrained_dict = torch.load('./Pretrained/mit_b3.pth')
        if model_type == 'B4':
            pretrained_dict = torch.load('./Pretrained/mit_b4.pth')
        if model_type == 'B5':
            pretrained_dict = torch.load('./Pretrained/mit_b5.pth')


        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        print("successfully loaded!!!!")


    def forward(self, x):

        ##################  Go through backbone ###################

        B = x.shape[0]

        #stage 1
        out_1, H, W = self.backbone.patch_embed1(x)
        for i, blk in enumerate(self.backbone.block1):
            out_1 = blk(out_1, H, W)
        out_1 = self.backbone.norm1(out_1)
        out_1 = out_1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[0], 88, 88)

        # stage 2
        out_2, H, W = self.backbone.patch_embed2(out_1)
        for i, blk in enumerate(self.backbone.block2):
            out_2 = blk(out_2, H, W)
        out_2 = self.backbone.norm2(out_2)
        out_2 = out_2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[1], 44, 44)

        # stage 3
        out_3, H, W = self.backbone.patch_embed3(out_2)
        for i, blk in enumerate(self.backbone.block3):
            out_3 = blk(out_3, H, W)
        out_3 = self.backbone.norm3(out_3)
        out_3 = out_3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[2], 22, 22)

        # stage 4
        out_4, H, W = self.backbone.patch_embed4(out_3)
        for i, blk in enumerate(self.backbone.block4):
            out_4 = blk(out_4, H, W)
        out_4 = self.backbone.norm4(out_4)
        out_4 = out_4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[3], 11, 11)

        #classification
        out2 = self.global_avg_pool(out_4)
        out2 = self.norm1(out2)
        out2 = self.Relu(out2)
        out2 = self.Dropout(out2)
        out2 = self.conv1(out2)
        out2 = self.norm2(out2)
        out2 = self.Relu(out2)
        out2 = self.conv2(out2)

        return out2

def loss_class(pred, target):
    return nn.BCEWithLogitsLoss()(pred, target)


def evaluate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ESFPNet.eval()
    total = 0
    total_correct_predictions = torch.zeros(7).to(device)
    threshold_class = 0.6

    val_loader = test_dataset(val_images_path + '/',val_masks_path + '/', label_path ,init_trainsize) #
    for i in range(val_loader.size):
        image, labels_tensor = val_loader.load_data()#

        image = image.cuda()
        labels_tensor = labels_tensor.to(device)
        pred2 = ESFPNet(image)
        pred2 = np.squeeze(pred2)
        pred2 = torch.unsqueeze(pred2, 0)
        total += 1

        labels_predicted = torch.sigmoid(pred2)
        thresholded_predictions = (labels_predicted >= threshold_class).int()
        correct_predictions = (thresholded_predictions == labels_tensor).sum(dim=0)
        total_correct_predictions += correct_predictions
    acc_1 = total_correct_predictions[0] / total
    print('Muscosal erythema', acc_1.item())
    wandb.log({'Muscosal erythema acc_val' : acc_1.item()})

    acc_2 = total_correct_predictions[1] / total
    print('Anthrocosis', acc_2.item())
    wandb.log({'Anthrocosis acc_val' : acc_2.item()})

    acc_3 = total_correct_predictions[2] / total
    print('Stenosis', acc_3.item())
    wandb.log({'Stenosis acc_val' : acc_3.item()})

    acc_4 = total_correct_predictions[3] / total
    print('Mucosal edema of carina', acc_4.item())
    wandb.log({'Mucosal edema of carina acc_val' : acc_4.item()})

    acc_5 = total_correct_predictions[4] / total
    print('Mucosal infiltration', acc_5.item())
    wandb.log({'Mucosal infiltration acc_val' : acc_5.item()})

    acc_6 = total_correct_predictions[5] / total
    print('Vascular growth', acc_6.item())
    wandb.log({'Vascular growth acc_val' : acc_6.item()})

    acc_7 = total_correct_predictions[6] / total
    print('Tumor', acc_7.item())
    wandb.log({'Tumor acc_val' : acc_7.item()})

    overall_accuracy = torch.mean(total_correct_predictions) / total
    print("acc_val_classification", overall_accuracy.item())
    wandb.log({"acc_val_classification" : overall_accuracy.item()})

    ESFPNet.train()

    return overall_accuracy.item()

def training_loop(n_epochs, ESFPNet_optimizer, numIters):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainDataset = PolypDataset(train_images_path + '/', train_masks_path + '/',label_path, trainsize=init_trainsize, augmentations = True) #
    train_loader = DataLoader(dataset=trainDataset,batch_size=batch_size,shuffle=True)

    segmentation_max = 0
    classification_max = 0
    mean_max = 0
    threshold = 0.6

    for epoch in range(n_epochs):
        loss_class_train = 0.0
        total_correct_predictions = torch.zeros(7).to(device)
        epoch_loss_all = 0.0
        total = 0
        threshold = 0.6

        for data in train_loader:
            images, labels_tensor = data
            
            images = images.to(device)
            labels_tensor = labels_tensor.to(device)
            
            total += labels_tensor.size(0)

            ESFPNet_optimizer.zero_grad()
            pred_labels = ESFPNet(images)
            #classification
            pred_labels = np.squeeze(pred_labels)

            loss_class_train = loss_class(pred_labels, labels_tensor)

            loss_total = loss_class_train
            
            loss_total.backward()
            ESFPNet_optimizer.step()
            epoch_loss_all += loss_total.item()

            labels_predicted = torch.sigmoid(pred_labels)

            thresholded_predictions = (labels_predicted >= threshold).int()
            correct_predictions = (thresholded_predictions == labels_tensor).sum(dim=0)
            total_correct_predictions += correct_predictions
            
        acc_1 = total_correct_predictions[0] / total
        print('Muscosal erythema', acc_1.item())
        wandb.log({'Muscosal erythema acc_train' : acc_1.item()})

        acc_2 = total_correct_predictions[1] / total
        print('Anthrocosis', acc_2.item())
        wandb.log({'Anthrocosis acc_train' : acc_2.item()})

        acc_3 = total_correct_predictions[2] / total
        print('Stenosis', acc_3.item())
        wandb.log({'Stenosis acc_train' : acc_3.item()})

        acc_4 = total_correct_predictions[3] / total
        print('Mucosal edema of carina', acc_4.item())
        wandb.log({'Mucosal edema of carina acc_train' : acc_4.item()})

        acc_5 = total_correct_predictions[4] / total
        print('Mucosal infiltration', acc_5.item())
        wandb.log({'Mucosal infiltration acc_train' : acc_5.item()})

        acc_6 = total_correct_predictions[5] / total
        print('Vascular growth', acc_6.item())
        wandb.log({'Vascular growth acc_train' : acc_6.item()})

        acc_7 = total_correct_predictions[6] / total
        print('Tumor', acc_7.item())
        wandb.log({'Tumor acc_train' : acc_7.item()})

        epoch_loss = epoch_loss_all /  len(train_loader)
        wandb.log({"Loss_train" : epoch_loss})
        print("epoch_loss", epoch_loss)
        #acc_classification
        overall_accuracy = torch.mean(total_correct_predictions) / total
        print("acc_train", overall_accuracy.item())
        wandb.log({"Acc_classification_train" : overall_accuracy*100})

        # print("-Training dataset. Got %d out of %d images correctly (%.3f%%). Epoch loss: %.3f"
        #       % (total_correct_predictions, total, overall_accuracy, epoch_loss))

        classification_acc = evaluate()

        if classification_max < classification_acc:
            data = 'Lung_cancer_lesions'
            classification_max = classification_acc
            save_model_path = './SaveModel/'+data+'/'
            os.makedirs(save_model_path, exist_ok=True)
            print(save_model_path)
            torch.save(ESFPNet, save_model_path + '/Classification_model.pt')               

import torch.optim as optim

for i in range(repeats):
    # Clear GPU cache
    torch.cuda.empty_cache()
    ESFPNet = ESFPNetStructure()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        ESFPNet.to(device)
        print('Models moved to GPU.')
    else:
        print('Only CPU available.')
    print('#####################################################################################')

    # hyperparams for Adam optimizer
    lr=0.0001 #0.0001

    ESFPNet_optimizer = optim.AdamW(ESFPNet.parameters(), lr=lr)

    #losses, coeff_max = training_loop(n_epochs, ESFPNet_optimizer, i+1)
    training_loop(n_epochs, ESFPNet_optimizer, i+1)
    # plt.plot(losses)

    # print('#####################################################################################')
    # print('optimize_m_dice: {:6.6f}'.format(coeff_max))

    # saveResult(i+1)
    # print('#####################################################################################')
    # print('saved the results')
    # print('#####################################################################################')

wandb.finish()