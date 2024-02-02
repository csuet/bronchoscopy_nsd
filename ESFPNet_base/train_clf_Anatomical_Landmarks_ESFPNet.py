import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,ConcatDataset
import json
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#from skimage import io, transform
from PIL import Image

import wandb
wandb.login()
wandb.init(project="ESFPNet_classification_on_Anatomical_Landmarks")

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
import argparse
from pathlib import Path

parser = argparse.ArgumentParser("ESFPNet based model")
parser.add_argument('--label_json_path', type=str, required=True,
        help='Location of the data directory containing json labels file of each task after combining two json files.json')
parser.add_argument('--dataset', type=str, required=True,
        help='Location of the dataset for each tasks')
parser.add_argument('--model_type', type=str, default='B4',
        help='Type of model (default B4)')
parser.add_argument('--init_trainsize', type=int, default=352,
        help='Size of image for training (default = 352)')
parser.add_argument('--batch_size', type=int, default=8,
        help='Batch size for training (default = 8)')
parser.add_argument('--n_epochs', type=int, default=500,
        help='Number of epochs for training (default = 500)')
args = parser.parse_args()

# Clear GPU cache
torch.cuda.empty_cache()

train_images_path = args.dataset + '/train/imgs'
train_masks_path  = args.dataset + '/train/masks'
val_images_path   = args.dataset + '/val/imgs'
val_masks_path    = args.dataset + '/val/masks'
test_images_path  = args.dataset + '/test/imgs' 
test_masks_path   = args.dataset + '/test/masks'


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
        
        with open(args.label_json_path, 'r') as f:
            data = json.load(f)

        label_list = ['Vocal cords', 'Main carina', 'Intermediate bronchus', 'Right superior lobar bronchus', 'Right inferior lobar bronchus', 'Right middle lobar bronchus', 'Left inferior lobar bronchus', 'Left superior lobar bronchus', 'Right main bronchus', 'Left main bronchus', 'Trachea']
        label_name = [file['label_name'] for file in data if file['object_id'] == file_name]
        
        label_tensor = torch.zeros([11])
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

        with open(args.label_json_path, 'r') as f:
            data = json.load(f)

        label_list = ['Vocal cords', 'Main carina', 'Intermediate bronchus', 'Right superior lobar bronchus', 'Right inferior lobar bronchus', 'Right middle lobar bronchus', 'Left inferior lobar bronchus', 'Left superior lobar bronchus', 'Right main bronchus', 'Left main bronchus', 'Trachea']
        label_name = [file['label_name'] for file in data if file['object_id'] == file_name]
        
        label_tensor = torch.zeros([11])
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

        #classification layer
        self.norm1 = nn.BatchNorm2d(512, eps=1e-5)
        self.Relu = nn.ReLU(inplace=True)
        self.Dropout = nn.Dropout(p=0.3)
        self.conv1 = nn.Conv2d(512, 256, 1, stride=1, padding=0)
        self.norm2 = nn.BatchNorm2d(256, eps=1e-5)
        self.conv2 = nn.Conv2d(256, 11, 1, stride=1, padding=0, bias=True) # 9 = number of classes
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim = 1)

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
    total_correct_predictions = torch.zeros(11).to(device)
    threshold_class = 0.6

    val_loader = test_dataset(val_images_path + '/',val_masks_path + '/', args.label_json_path , args.init_trainsize) #
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
    print('Vocal cords', acc_1.item())
    wandb.log({'Vocal cords acc_val' : acc_1.item()})

    acc_2 = total_correct_predictions[1] / total
    print('Main carina', acc_2.item())
    wandb.log({'Main carina acc_val' : acc_2.item()})

    acc_3 = total_correct_predictions[2] / total
    print('Intermediate bronchus', acc_3.item())
    wandb.log({'Intermediate bronchus acc_val' : acc_3.item()})

    acc_4 = total_correct_predictions[3] / total
    print('Right superior lobar bronchus', acc_4.item())
    wandb.log({'Right superior lobar bronchus acc_val' : acc_4.item()})

    acc_5 = total_correct_predictions[4] / total
    print('Right inferior lobar bronchus', acc_5.item())
    wandb.log({'Right inferior lobar bronchus acc_val' : acc_5.item()})

    acc_6 = total_correct_predictions[5] / total
    print('Right middle lobar bronchus', acc_6.item())
    wandb.log({'Right middle lobar bronchus acc_val' : acc_6.item()})

    acc_7 = total_correct_predictions[6] / total
    print('Left inferior lobar bronchus', acc_7.item())
    wandb.log({'Left inferior lobar bronchus acc_val' : acc_7.item()})

    acc_8 = total_correct_predictions[7] / total
    print('Left superior lobar bronchus', acc_8.item())
    wandb.log({'Left superior lobar bronchus acc_val' : acc_8.item()})
    
    acc_9 = total_correct_predictions[8] / total
    print('Right main bronchus', acc_9.item())
    wandb.log({'Right main bronchus acc_val' : acc_9.item()})

    acc_10 = total_correct_predictions[9] / total
    print('Left main bronchus', acc_10.item())
    wandb.log({'Left main bronchus acc_val' : acc_10.item()})

    acc_11 = total_correct_predictions[10] / total
    print('Trachea', acc_11.item())
    wandb.log({'Trachea acc_val' : acc_11.item()})
    

    overall_accuracy = torch.mean(total_correct_predictions) / total
    print("acc_val_classification", overall_accuracy.item())
    wandb.log({"acc_val_classification" : overall_accuracy.item()})

    ESFPNet.train()

    return overall_accuracy.item()

def training_loop(n_epochs, ESFPNet_optimizer, numIters):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainDataset = PolypDataset(train_images_path + '/', train_masks_path + '/',args.label_json_path, trainsize= args.init_trainsize, augmentations = True) #
    train_loader = DataLoader(dataset=trainDataset,batch_size= args.batch_size,shuffle=True)

    segmentation_max = 0
    classification_max = 0
    mean_max = 0
    threshold = 0.6

    for epoch in range(n_epochs):
        loss_class_train = 0.0
        total_correct_predictions = torch.zeros(11).to(device)
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
        print('Anatomical_LandmarksKQ', acc_1.item())
        wandb.log({'Anatomical_LandmarksKQ acc_train' : acc_1.item()})

        acc_2 = total_correct_predictions[1] / total
        print('Anatomical_Landmarks - DT', acc_2.item())
        wandb.log({'Anatomical_Landmarks - DT acc_train' : acc_2.item()})

        acc_3 = total_correct_predictions[2] / total
        print('Anatomical_Landmarks - CARINA', acc_3.item())
        wandb.log({'Anatomical_Landmarks - CARINA acc_train' : acc_3.item()})

        acc_4 = total_correct_predictions[3] / total
        print('Anatomical_Landmarks - PQGP', acc_4.item())
        wandb.log({'Anatomical_Landmarks - PQGP acc_train' : acc_4.item()})

        acc_5 = total_correct_predictions[4] / total
        print('Anatomical_Landmarks - PQGT', acc_5.item())
        wandb.log({'Anatomical_Landmarks - PQGT acc_train' : acc_5.item()})

        acc_6 = total_correct_predictions[5] / total
        print('Anatomical_Landmarks - PQTTT', acc_6.item())
        wandb.log({'Anatomical_Landmarks - PQTTT acc_train' : acc_6.item()})

        acc_7 = total_correct_predictions[6] / total
        print('Left inferior lobar bronchus', acc_7.item())
        wandb.log({'Left inferior lobar bronchus acc_train' : acc_7.item()})

        acc_8 = total_correct_predictions[7] / total
        print('Left superior lobar bronchus', acc_8.item())
        wandb.log({'Left superior lobar bronchus acc_train' : acc_8.item()})
        
        acc_9 = total_correct_predictions[8] / total
        print('Right main bronchus', acc_9.item())
        wandb.log({'Right main bronchus acc_train' : acc_9.item()})

        acc_10 = total_correct_predictions[9] / total
        print('Left main bronchus', acc_10.item())
        wandb.log({'Left main bronchus acc_train' : acc_10.item()})

        acc_11 = total_correct_predictions[10] / total
        print('Trachea', acc_11.item())
        wandb.log({'Trachea acc_train' : acc_11.item()})

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
            data = 'Anatomical_Landmarks'
            classification_max = classification_acc
            save_model_path = './SaveModel/'+data+'/'
            os.makedirs(save_model_path, exist_ok=True)
            print(save_model_path)
            torch.save(ESFPNet, save_model_path + '/Classification_model.pt')               

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

    #losses, coeff_max = training_loop(n_epochs, ESFPNet_optimizer, i+1)
    training_loop(args.n_epochs, ESFPNet_optimizer, i+1)
    # plt.plot(losses)

    # print('#####################################################################################')
    # print('optimize_m_dice: {:6.6f}'.format(coeff_max))

    # saveResult(i+1)
    # print('#####################################################################################')
    # print('saved the results')
    # print('#####################################################################################')

wandb.finish()