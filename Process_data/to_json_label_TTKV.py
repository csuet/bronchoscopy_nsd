# Create json files for Lesions task
import json

# Path to json files
path_annots = '/workspace/ailab/phucnd/04102023_Nhom_chung/annotation_final.json'
path_objects = '/workspace/ailab/phucnd/04102023_Nhom_chung/objects_final.json'
path_labels = '/workspace/ailab/phucnd/04102023_Nhom_chung/labels_final.json'

# path_annots = '/workspace/ailab/phucnd/04102023_Nhom_benh/annotation_final.json'
# path_objects = '/workspace/ailab/phucnd/04102023_Nhom_benh/objects_final.json'
# path_labels = '/workspace/ailab/phucnd/04102023_Nhom_benh/labels_final.json'

# Read json files
annots = open(path_annots)
data_annots = json.load(annots)

objects = open(path_objects)
data_objects = json.load(objects)

labels = open(path_labels)
data_labels = json.load(labels)

# path_save = '/workspace/ailab/phucnd/04102023_Nhom_benh/json_label_TTKV_final.json'
path_save = '/workspace/ailab/phucnd/04102023_Nhom_chung/json_label_TTKV_final_test.json'

try:
    with open(path_save, 'r') as json_file:
        existing_data = json.load(json_file)
except json.decoder.JSONDecodeError:
    existing_data = []

# Labels of Lesions
label_id = ["a2ff1ceb-280b-410d-9ddc-82cb4a2e2ccb", "ff9241a0-a760-49d4-b34d-54f20ddedb0c",
            "b96f104d-948d-4507-8b9c-c1dd0734b759", "4001669a-6f32-4a80-91f1-7c0766f19d29",
            "cbed49c9-81ec-45d0-b619-a12c57c1770f", "3f2cce73-e832-4e8c-93f3-984b9e48baf3",
            "cf97e212-5790-4ce0-92b0-5f585cfbbd8c"]

# Create json files contains image and labels of image
for i in data_annots:
    for k in data_labels:
        for x in i['label_ids']:
            if (x in label_id):
                if (k['id'] == x):
                    new_data = {
                        "object_id": i['object_id'],
                        "label_id": x,
                        "label_name": k['name']
                    }
                    if new_data not in existing_data:
                        existing_data.append(new_data)

                    # Save file
                    with open(path_save, 'w') as json_file:
                        json.dump(existing_data, json_file, ensure_ascii=False)


