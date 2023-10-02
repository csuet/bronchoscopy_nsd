import json
import os
from PIL import Image

path_image = '/workspace/ailab/phucnd/Segmentation_MGPKV_final_data/imgs'
path_masks = '/workspace/ailab/phucnd/Segmentation_MGPKV_final_data/masks'

path_json_label = '/workspace/ailab/phucnd/25092023_112_Nhom_benh/json_label_MGPKV_final.json'
# path_json_label = '/workspace/ailab/phucnd/25092023_109_Nhom_chung/json_label_MGPKV_final.json'


json_label = open(path_json_label)
data_json_label = json.load(json_label)

path_object = '/workspace/ailab/phucnd/25092023_112_Nhom_benh/objects_new.json'
# path_object = '/workspace/ailab/phucnd/25092023_109_Nhom_chung/objects_new.json'

objects = open(path_object)
data_objects = json.load(objects)

for i in data_json_label:
    for j in data_objects:
        for k in j['series']:
            for l in k['images']:
                if (i['object_id'] == l['object_id']):
                    directory_img = os.path.join('/workspace/ailab/phucnd/25092023_112_Nhom_benh/imgs',j['object_id'],k['object_id'],f'{l["object_id"]}.png')
                    directory_mask = os.path.join('/workspace/ailab/phucnd/25092023_112_Nhom_benh/masks_MGPKV',j['object_id'],k['object_id'],f'{l["object_id"]}.png')
                    new_path_img = os.path.join(path_image,f'{l["object_id"]}.png')
                    new_path_mask = os.path.join(path_masks,f'{l["object_id"]}.png')
                    with Image.open(directory_img) as img:
                        img.save(new_path_img)
                    with Image.open(directory_mask) as img:
                        img.save(new_path_mask)
                        print('donejson')

                    
                    
