# Create json files for Anatomical landmark task
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
# Labels of Anatomical landmark
label_id = ["b2cc70ea-01f6-4389-a3a3-6222600a445a", "ce08e883-ddd2-4639-aa05-9fdac022f545",
            "c93010c7-b77b-44a8-8866-7511f989e97a", "7e3b10f1-c8d2-4bf3-8250-1dc5954f2de5",
            "a9d13324-3c88-4135-88bf-7fbb4ccb4e13", "1578d8da-1512-479c-aed7-ef7cd4fb5541",
            "c115009a-d19a-4337-9151-0dd20a2562e7", "e572ea16-52e8-4404-b866-363eb3f733ce",
            "27cc98cd-7cad-4134-bfb5-b2c74af3326e", "ccca7aa2-1593-4e16-a436-38a5516ce433",
            "0ac584fb-aaeb-44f1-9fa1-fafd973ddac8"]

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



