# Create ground truth masks for both tasks
import math
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import cv2
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser("Process data")
parser.add_argument('--data_annots', type=str, required=True,
        help='Location of the data directory containing annotation.json')
parser.add_argument('--data_objects', type=str, required=True,
        help='Location of the data directory containing objects.json')
parser.add_argument('--data_labels', type=str, required=True,
        help='Location of the data directory containing labels.json')
parser.add_argument('--path_save', type=str, required=True,
        help='Location of the saved file (In the same folder of imgs folder) (Recommend name: masks_Lung_cancer_lesions or masks_Anatomical_Landmarks))')
args = parser.parse_args()

path_annots = Path(args.data_annots)
path_objects = Path(args.data_objects)
path_labels = Path(args.data_labels)

# Read json files
annots = open(path_annots)
data_annots = json.load(annots)

objects = open(path_objects)
data_objects = json.load(objects)

labels = open(path_labels)
data_labels = json.load(labels)

shape_x = []
shape_y = []

# Labels of Anatomical landmark = ["Vocal cords", "Main carina","Intermediate bronchus","Right superior lobar bronchus","Right inferior lobar bronchus","Right middle lobar bronchus","Left inferior lobar bronchus","Left superior lobar bronchus","Right main bronchus","Left main bronchus","Trachea"]
label_Anatomical_landmark = ["b2cc70ea-01f6-4389-a3a3-6222600a445a", "ce08e883-ddd2-4639-aa05-9fdac022f545","c93010c7-b77b-44a8-8866-7511f989e97a", "7e3b10f1-c8d2-4bf3-8250-1dc5954f2de5","a9d13324-3c88-4135-88bf-7fbb4ccb4e13", "1578d8da-1512-479c-aed7-ef7cd4fb5541","c115009a-d19a-4337-9151-0dd20a2562e7", "e572ea16-52e8-4404-b866-363eb3f733ce","27cc98cd-7cad-4134-bfb5-b2c74af3326e", "ccca7aa2-1593-4e16-a436-38a5516ce433","0ac584fb-aaeb-44f1-9fa1-fafd973ddac8"]
# Labels of Lesions = ["Mucosal infiltration", "Mucosal edema of carina","Vascular growth","Muscosal erythema","Tumor","Anthrocosis","Stenosis"]
label_Lesions = ["a2ff1ceb-280b-410d-9ddc-82cb4a2e2ccb", "ff9241a0-a760-49d4-b34d-54f20ddedb0c","b96f104d-948d-4507-8b9c-c1dd0734b759", "4001669a-6f32-4a80-91f1-7c0766f19d29","cbed49c9-81ec-45d0-b619-a12c57c1770f", "3f2cce73-e832-4e8c-93f3-984b9e48baf3","cf97e212-5790-4ce0-92b0-5f585cfbbd8c"]


# Create ground truth for segmentation tasks
for i in data_objects:
    for j in i['videos']:
        for k in j['images']:
            # Save file for each task
            save_path = Path(args.path_save)
            path = os.path.join(save_path, i['id'], j['video_id'])
            if not os.path.exists(path):
                os.makedirs(path)
            os.chdir(path)
            mask = np.zeros((480, 480))  # image 480x480
            for l in data_annots:
                if (l['object_id'] == k['image_id']):
                    for x in l['label_ids']:
                        if x in label_Lesions:
                            for p in l['data']:
                                shape_x.append(int(p['x']))
                                shape_y.append(int(p['y']))

                            ab = np.stack((shape_x, shape_y), axis=1)
                            img3 = cv2.drawContours(mask, [ab], -1, 255, -1)

                            cv2.imwrite(l['object_id'] + '.png', mask.astype(np.uint8))
                            shape_x.clear()
                            shape_y.clear()


