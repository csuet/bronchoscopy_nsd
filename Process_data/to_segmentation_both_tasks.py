# Create files for both Anatomical landmark and Lesions Segmentation Tasks
import json
import os
from PIL import Image
import argparse
from pathlib import Path

parser = argparse.ArgumentParser("Process data")
parser.add_argument('--data_json_labels', type=str, required=True,
                    help='Location of the data directory containing json_labels file')
parser.add_argument('--data_imgs', type=str, required=True,
                    help='Location of the data directory containing imgs')
parser.add_argument('--data_objects', type=str, required=True,
                    help='Location of the data directory containing objects.json')
parser.add_argument('--data_masks', type=str, required=True,
                    help='Location of the data directory containing ground truth masks for segmentation')
parser.add_argument('--data_save_imgs', type=str, required=True,
                    help='Location of the data directory containing images saved files')
parser.add_argument('--data_save_masks', type=str, required=True,
                    help='Location of the data directory containing ground truth masks saved files')
args = parser.parse_args()

# Path to save files
path_save_image = Path(args.data_save_imgs)
path_save_masks = Path(args.data_save_masks)

# Path to json files
path_json_label = Path(args.data_json_labels)

json_label = open(path_json_label)
data_json_label = json.load(json_label)

# Path to objects files
path_object = Path(args.data_objects)

objects = open(path_object)
data_objects = json.load(objects)

# Create files
for i in data_json_label:
    for j in data_objects:
        for k in j['videos']:
            for l in k['images']:
                if (i['object_id'] == l['image_id']):
                    img_dir = Path(args.data_imgs)
                    mask_dir = Path(args.data_masks)
                    directory_img = os.path.join(img_dir, j['id'], k['video_id'], f'{l["image_id"]}.png')
                    directory_mask = os.path.join(mask_dir, j['id'], k['video_id'], f'{l["image_id"]}.png')
                    new_path_img = os.path.join(path_save_image, f'{l["image_id"]}.png')
                    new_path_mask = os.path.join(path_save_masks, f'{l["image_id"]}.png')
                    # Save files
                    with Image.open(directory_img) as img:
                        img.save(new_path_img)
                    with Image.open(directory_mask) as img:
                        img.save(new_path_mask)



