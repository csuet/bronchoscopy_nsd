import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,ConcatDataset
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


#from skimage import io, transform
from PIL import Image  

# visualizing data
import matplotlib.pyplot as plt
import numpy as np
import warnings

from sklearn.model_selection import train_test_split

# load dataset information
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
parser.add_argument('--model_type', type=str, default='B4',
        help='Type of model (default B4)')
parser.add_argument('--init_trainsize', type=int, default=352,
        help='Size of image for training (default = 352)')
parser.add_argument('--batch_size', type=int, default=8,
        help='Batch size for training (default = 8)')
parser.add_argument('--n_epochs', type=int, default=500,
        help='Number of epochs for training (default = 500)')
parser.add_argument('--if_renew', type=bool, default=True,
        help='Check if split data to train_val_test')
parser.add_argument('--task', type=str, default='Anatomical_Landmarks',
        help='Task: Anatomical_Landmarks or Lung_cancer_lesions')
args = parser.parse_args()


# Clear GPU cache
torch.cuda.empty_cache()


import wandb
wandb.login()
wandb.init(project="ESFPNet_segmetation_task_on_" + args.task)

class SplittingDataset(Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root):

        self.images = []
        
        for root, dirs, files in os.walk(image_root):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    self.images.append(os.path.join(root, file))
        
        self.gts = []

        for root, dirs, files in os.walk(gt_root):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    self.gts.append(os.path.join(root, file))
                    
        # or /masks_Anatomical_Landmarks
        mask_dir = "/masks_" + args.task + "/"
        self.images = [file for file in self.images if file.replace('/imgs/', mask_dir) in self.gts]
        
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
    split_train_images_save_path = './dataset/' + args.task+ '/train/imgs'
    os.makedirs(split_train_images_save_path, exist_ok=True)
    split_train_masks_save_path = './dataset/' + args.task + '/train/masks'
    os.makedirs(split_train_masks_save_path, exist_ok=True)
    
    split_validation_images_save_path = './dataset/' + args.task + '/val/imgs'
    os.makedirs(split_validation_images_save_path, exist_ok=True)
    split_validation_masks_save_path ='./dataset/' + args.task + '/val/masks'
    os.makedirs(split_validation_masks_save_path, exist_ok=True)
    
    split_test_images_save_path = './dataset/' + args.task + '/test/imgs'
    os.makedirs(split_test_images_save_path, exist_ok=True)
    split_test_masks_save_path = './dataset/' + args.task + '/test/masks'
    os.makedirs(split_test_masks_save_path, exist_ok=True)
    
    if renew == True:
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

        Y_data = [element for element in labels_list if element_counts[element] > 4]

        for element in labels_list:
            if element_counts[element] < 5:
                removed_elements.append(element)

        combine_list = list(zip(imgs_list, masks_list, names_list, labels_list))
       
        X_data = [tup for tup in combine_list if not any(item in removed_elements for item in tup)]

        
        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.1, 
                                                            random_state=42, stratify = Y_data) 

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, 
                                                            random_state=42, stratify = y_train) 
        
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

train_images_path, train_masks_path, val_images_path, val_masks_path, test_images_path, test_masks_path = splitDataset(args.if_renew)


class PolypDataset(Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize, augmentations): #
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        
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

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        np.random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        np.random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        
        return image, gt

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
    def __init__(self, image_root, gt_root,  testsize): #
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png') or f.endswith('.jpg')]

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
        self.index += 1
        return image, gt

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
        if args.model_type == 'B0':
            self.backbone = mit.mit_b0()
        if args.model_type == 'B1':
            self.backbone = mit.mit_b1()
        if args.model_type == 'B2':
            self.backbone = mit.mit_b2()
        if args.model_type == 'B3':
            self.backbone = mit.mit_b3()
        if args.model_type == 'B4':
            self.backbone = mit.mit_b4()
        if args.model_type == 'B5':
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

    def _init_weights(self):

        if args.model_type == 'B0':
            pretrained_dict = torch.load('./Pretrained/mit_b0.pth')
        if args.model_type == 'B1':
            pretrained_dict = torch.load('./Pretrained/mit_b1.pth')
        if args.model_type == 'B2':
            pretrained_dict = torch.load('./Pretrained/mit_b2.pth')
        if args.model_type == 'B3':
            pretrained_dict = torch.load('./Pretrained/mit_b3.pth')
        if args.model_type == 'B4':
            pretrained_dict = torch.load('./Pretrained/mit_b4.pth')
        if args.model_type == 'B5':
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
        return out1

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

from torch.autograd import Variable

def evaluate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ESFPNet.eval()
    val = 0
    count = 0

    smooth = 1e-4

    val_loader = test_dataset(val_images_path + '/',val_masks_path + '/' ,args.init_trainsize) #
    for i in range(val_loader.size):
        image, gt= val_loader.load_data()#
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        image = image.cuda()
        
        pred1= ESFPNet(image)
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

    ESFPNet.train()

    return val/count

# train the network
def training_loop(n_epochs, ESFPNet_optimizer, numIters):

    # keep track of losses over time
    losses = []
    total = 0
    running_correct = 0.0
    loss_seg_train = 0.0
    loss_class_train = 0.0
    coeff_max = 0
    epoch_loss_all1 = 0.0
    # set up data and then train
    trainDataset = PolypDataset(train_images_path + '/', train_masks_path + '/', trainsize=args.init_trainsize, augmentations = True) #
    train_loader = DataLoader(dataset=trainDataset,batch_size=args.batch_size,shuffle=True)

    iter_X = iter(train_loader)
    steps_per_epoch = len(iter_X)
    num_epoch = 0
    total_steps = (n_epochs+1)*steps_per_epoch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for step in range(1, total_steps):

        # Reset iterators for each epoch
        if step % steps_per_epoch == 0:
            iter_X = iter(train_loader)
            num_epoch = num_epoch + 1

        # make sure to scale to a range -1 to 1
        images, masks= next(iter_X) #

        # move images to GPU if available (otherwise stay on CPU)
        images = images.to(device)
        masks = masks.to(device)

        total += 1

        # ============================================
        #            TRAIN THE NETWORKS
        # ============================================

        ESFPNet_optimizer.zero_grad()

        # 1. Compute the losses from the network
        pred_masks= ESFPNet(images)
        #segmentation
        pred_masks = F.interpolate(pred_masks, scale_factor=4, mode='bilinear', align_corners=False)
        loss = ange_structure_loss(pred_masks, masks)
        loss_seg_train += loss.item()
        loss.backward()
        ESFPNet_optimizer.step()

        # ============================================
        #            TRAIN THE NETWORKS
        # ============================================

        #segmentation
        if step % steps_per_epoch == 0:
            losses.append(loss.item())
            wandb.log({"loss_train" : loss_seg_train/steps_per_epoch})
            print('Epoch [{:5d}/{:5d}] | preliminary loss: {:6.6f} '.format(num_epoch, n_epochs, loss.item()))
            loss_seg_train = 0.0
        if step % steps_per_epoch == 0:
            validation_coeff = evaluate()
            print('Epoch [{:5d}/{:5d}] | validation_coeffient: {:6.6f} '.format(num_epoch, n_epochs, validation_coeff))
            wandb.log({"dice_val" : validation_coeff})
            if coeff_max < validation_coeff:
                coeff_max = validation_coeff
                save_model_path = './SaveModel/' + data +'/'
                os.makedirs(save_model_path, exist_ok=True)
                print(save_model_path)
                torch.save(ESFPNet, save_model_path + '/Segmentation_model.pt')
                print('Save Learning Ability Optimized Model at Epoch [{:5d}/{:5d}]'.format(num_epoch, n_epochs))
                
    return losses, coeff_max


import torch.optim as optim

for i in range(1):
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

    losses, coeff_max = training_loop(args.n_epochs, ESFPNet_optimizer, i+1)

    # plt.plot(losses)

    # print('#####################################################################################')
    # print('optimize_m_dice: {:6.6f}'.format(coeff_max))

    # saveResult(i+1)
    # print('#####################################################################################')
    # print('saved the results')
    # print('#####################################################################################')
wandb.finish()