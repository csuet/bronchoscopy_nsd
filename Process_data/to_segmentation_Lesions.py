# Create files for Lesions Segmentation Tasks
import json
import os
from PIL import Image

# Path to save files
path_image = '/workspace/ailab/phucnd/Segmentation_TTKV_data/imgs'
path_masks = '/workspace/ailab/phucnd/Segmentation_TTKV_data/masks'

# Path to json files
path_json_label = '/workspace/ailab/phucnd/04102023_Nhom_chung/json_label_TTKV_final.json'
# path_json_label = '/workspace/ailab/phucnd/04102023_Nhom_benh/json_label_TTKV_final.json'

json_label = open(path_json_label)
data_json_label = json.load(json_label)

path_object = '/workspace/ailab/phucnd/04102023_Nhom_chung/objects_final.json'
# path_object = '/workspace/ailab/phucnd/04102023_Nhom_benh/objects_final.json'

objects = open(path_object)
data_objects = json.load(objects)

# Create files
for i in data_json_label:
    for j in data_objects:
        for k in j['videos']:
            for l in k['images']:
                if (i['object_id'] == l['image_id']):
                    directory_img = os.path.join('/workspace/ailab/phucnd/04102023_Nhom_chung/imgs', j['id'],
                                                 k['video_id'], f'{l["image_id"]}.png')
                    directory_mask = os.path.join('/workspace/ailab/phucnd/04102023_Nhom_chung/masks_TTKV', j['id'],
                                                  k['video_id'], f'{l["image_id"]}.png')
                    new_path_img = os.path.join(path_image, f'{l["image_id"]}.png')
                    new_path_mask = os.path.join(path_masks, f'{l["image_id"]}.png')
                    # Save files
                    with Image.open(directory_img) as img:
                        img.save(new_path_img)
                    with Image.open(directory_mask) as img:
                        img.save(new_path_mask)


