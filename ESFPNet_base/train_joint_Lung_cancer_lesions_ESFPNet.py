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
wandb.init(project="ESFPNet_multimodel_Lung_cancer_lesions")

# visualizing data
import matplotlib.pyplot as plt
import numpy as np
import warnings

# load dataset information~
import yaml

from sklearn.metrics import accuracy_score

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
batch_size = 8

repeats = 1
n_epochs = 1000
if_renew = False#True
data = 'Lung_cancer_lesions'
label_path = './labels/labels_Lung_cancer_lesions_final.json'

class SplittingDataset(Dataset):
    
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
        
        name_image = self.images[index].split('/')[-1]

        file_name = os.path.splitext(os.path.basename(self.images[index]))[0]

        with open(label_path, 'r') as f:
            data = json.load(f)

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


def splitDataset(renew):
    
    split_train_images_save_path = './dataset/' + data + '/train/imgs'
    os.makedirs(split_train_images_save_path, exist_ok=True)
    split_train_masks_save_path = './dataset/' + data + '/train/masks'
    os.makedirs(split_train_masks_save_path, exist_ok=True)
    
    split_validation_images_save_path = './dataset/' + data + '/val/imgs'
    os.makedirs(split_validation_images_save_path, exist_ok=True)
    split_validation_masks_save_path ='./dataset/' + data + '/val/masks'
    os.makedirs(split_validation_masks_save_path, exist_ok=True)
    
    split_test_images_save_path = './dataset/' + data + '/test/imgs'
    os.makedirs(split_test_images_save_path, exist_ok=True)
    split_test_masks_save_path = './dataset/' + data + '/test/masks'
    os.makedirs(split_test_masks_save_path, exist_ok=True)
    
    if renew == True:
        images_train_path = './Segmentation_' + data + '_data/imgs'
        masks_train_path = './Segmentation_' + data + '_data/masks'
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
    return split_train_images_save_path, split_train_masks_save_path, split_validation_images_save_path, split_validation_masks_save_path, split_test_images_save_path, split_test_masks_save_path

train_images_path, train_masks_path, val_images_path, val_masks_path, test_images_path, test_masks_path = splitDataset(if_renew)

class PolypDataset(Dataset):
    
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
        
        return image, gt, label_tensor

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
        return image, gt, label_tensor

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


        #segmentation
        # go through LP Header
        lp_1 = self.LP_1(out_1)
        lp_2 = self.LP_2(out_2)
        lp_3 = self.LP_3(out_3)
        lp_4 = self.LP_4(out_4)

        # linear fuse and go pass LP Header
        lp_34 = self.LP_34(self.linear_fuse34(torch.cat([lp_3, F.interpolate(lp_4,scale_factor=2,mode='bilinear', align_corners=False)], dim=1)))
        lp_23 = self.LP_23(self.linear_fuse23(torch.cat([lp_2, F.interpolate(lp_34,scale_factor=2,mode='bilinear', align_corners=False)], dim=1)))
        lp_12 = self.LP_12(self.linear_fuse12(torch.cat([lp_1, F.interpolate(lp_23,scale_factor=2,mode='bilinear', align_corners=False)], dim=1)))

        # get the final output
        lp4_resized = F.interpolate(lp_4,scale_factor=8,mode='bilinear', align_corners=False)
        lp3_resized = F.interpolate(lp_34,scale_factor=4,mode='bilinear', align_corners=False)
        lp2_resized = F.interpolate(lp_23,scale_factor=2,mode='bilinear', align_corners=False)
        lp1_resized = lp_12

        out1 = self.linear_pred(torch.cat([lp1_resized, lp2_resized, lp3_resized, lp4_resized], dim=1))
        # print(out.shape)


        #classification
        out2 = self.global_avg_pool(out_4)
        out2 = self.norm1(out2)
        out2 = self.Relu(out2)
        out2 = self.Dropout(out2)
        out2 = self.conv1(out2)
        out2 = self.norm2(out2)
        out2 = self.Relu(out2)
        out2 = self.conv2(out2)

        return out1, out2

def ange_structure_loss(pred, mask, smooth=1):
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + smooth)/(union - inter + smooth)
    
    return (wbce + wiou).mean()

def dice_loss_coff(pred, target, smooth = 0.0001):
    
    num = target.size(0)
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    
    return loss.sum()/num

def loss_class(pred, target):
    return nn.BCEWithLogitsLoss()(pred, target)

from torch.autograd import Variable

def evaluate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ESFPNet.eval()
    total = 0
    total_correct_predictions = torch.zeros(7).to(device)
    val = 0
    count = 0
    threshold_class = 0.6

    smooth = 1e-4

    val_loader = test_dataset(val_images_path + '/',val_masks_path + '/', label_path ,init_trainsize) #
    for i in range(val_loader.size):
        image, gt, labels_tensor = val_loader.load_data()#
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        image = image.cuda()
        labels_tensor = labels_tensor.cuda()

        pred1, pred2= ESFPNet(image)
        pred2 = np.squeeze(pred2)
        pred2 = torch.unsqueeze(pred2, 0)
        pred1 = F.upsample(pred1, size=gt.shape, mode='bilinear', align_corners=False)
        pred1 = pred1.sigmoid()
        threshold = torch.tensor([0.5]).to(device)
        pred1 = (pred1 > threshold).float() * 1

        pred1 = pred1.data.cpu().numpy().squeeze()
        pred1 = (pred1 - pred1.min()) / (pred1.max() - pred1.min() + 1e-8)
        target = np.array(gt)

        input_flat = np.reshape(pred1,(-1))
        target_flat = np.reshape(target,(-1))
        intersection = (input_flat*target_flat)
        loss =  (2 * intersection.sum() + smooth) / (pred1.sum() + target.sum() + smooth)

        a =  '{:.4f}'.format(loss)
        a = float(a)

        val = val + a
        count = count + 1

        total = total + 1

        labels_predicted = torch.sigmoid(pred2)
        thresholded_predictions = (labels_predicted >= threshold_class).int()
        correct_predictions = (thresholded_predictions == labels_tensor).sum(dim=0)
        total_correct_predictions += correct_predictions
    
    print("dice_val_segmetation", 100 * val/count)
    wandb.log({"dice_val_segmentation" : val/count})
    
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

    return 100 * val/count,100* overall_accuracy.item()

def training_loop(n_epochs, ESFPNet_optimizer, numIters):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainDataset = PolypDataset(train_images_path + '/', train_masks_path + '/',label_path, trainsize=init_trainsize, augmentations = True) #
    train_loader = DataLoader(dataset=trainDataset,batch_size=batch_size,shuffle=True)

    segmentation_max = 0.0
    classification_max = 0.0
    mean_max = 0
    threshold = 0.6

    for epoch in range(n_epochs):
        loss_seg_train = 0.0
        loss_class_train = 0.0
        total_correct_predictions = torch.zeros(7).to(device)
        epoch_loss_all = 0.0
        total = 0
        threshold = 0.6

        for data in train_loader:
            images, masks, labels_tensor = data
            
            images = images.to(device)
            masks = masks.to(device)
            labels_tensor = labels_tensor.to(device)
            
            total += labels_tensor.size(0)

            ESFPNet_optimizer.zero_grad()
            pred_masks, pred_labels = ESFPNet(images)
            #segmentation
            pred_masks = F.interpolate(pred_masks, scale_factor=4, mode='bilinear', align_corners=False)
            #classification
            pred_labels = np.squeeze(pred_labels)

            loss_seg_train = ange_structure_loss(pred_masks, masks)
            loss_class_train = loss_class(pred_labels, labels_tensor)

            loss_total = loss_seg_train + loss_class_train
            
            loss_total.backward()
            ESFPNet_optimizer.step()
            epoch_loss_all += loss_total.item()

            labels_predicted = torch.sigmoid(pred_labels)

            thresholded_predictions = (labels_predicted >= threshold).int()
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

        epoch_loss = epoch_loss_all /  len(train_loader)
        wandb.log({"Loss_train" : epoch_loss})
        print("epoch_loss", epoch_loss)

        #acc_classification
        overall_accuracy = torch.mean(total_correct_predictions) / total
        print("acc_train", overall_accuracy.item())
        wandb.log({"Acc_classification_train" : overall_accuracy*100})

        # print("-Training dataset. Got %d out of %d images correctly (%.3f%%). Epoch loss: %.3f"
        #       % (total_correct_predictions, total, overall_accuracy.item(), epoch_loss))

        data = 'Lung_cancer_lesions_multimodel'
        segmentation_dice, classification_acc = evaluate()
        if segmentation_max < segmentation_dice:
            segmentation_max = segmentation_dice
            save_model_path = './SaveModel/'+ data+ '/'
            os.makedirs(save_model_path, exist_ok=True)
            print(save_model_path)
            torch.save(ESFPNet, save_model_path + '/Segmentation_best.pt')
            #print('Save Learning Ability Optimized Model at Epoch [{:5d}/{:5d}]'.format(num_epoch, n_epochs))

        if classification_max < classification_acc:
            classification_max = classification_acc
            save_model_path = './SaveModel/'+ data+ '/'
            os.makedirs(save_model_path, exist_ok=True)
            print(save_model_path)
            torch.save(ESFPNet, save_model_path + '/Classification_best.pt')
            #print('Save Learning Ability Optimized Model at Epoch [{:5d}/{:5d}]'.format(num_epoch, n_epochs))
        
        mean_eva = (segmentation_dice + classification_acc) / 2
        if mean_max < mean_eva:
            mean_max = mean_eva
            save_model_path = './SaveModel/'+ data+ '/'
            os.makedirs(save_model_path, exist_ok=True)
            print(save_model_path)
            torch.save(ESFPNet, save_model_path + '/Mean_best.pt')
            #print('Save Learning Ability Optimized Model at Epoch [{:5d}/{:5d}]'.format(num_epoch, n_epochs))
        save_model_path = './SaveModel/'+ data+ '/'
        torch.save(ESFPNet, save_model_path + '/Epoch.pt')

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
    training_loop(n_epochs, ESFPNet_optimizer, i+1, )
    # plt.plot(losses)

    # print('#####################################################################################')
    # print('optimize_m_dice: {:6.6f}'.format(coeff_max))

    # saveResult(i+1)
    # print('#####################################################################################')
    # print('saved the results')
    # print('#####################################################################################')

wandb.finish()