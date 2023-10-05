# Create json files for Lesions task
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser("Preprocess data")
parser.add_argument('--data_annots', type=str, required=True,
        help='Location of the data directory containing annotation.json')
parser.add_argument('--data_objects', type=str, required=True,
        help='Location of the data directory containing labels.json')
parser.add_argument('--data_labels', type=str, required=True,
        help='Location of the data directory containing objects.json')
parser.add_argument('--path_save', type=str, required=True,
        help='Location of the saved file)')
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

path_save = Path(args.path_save)

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


