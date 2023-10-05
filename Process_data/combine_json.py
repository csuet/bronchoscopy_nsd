# Combine two json files
import json
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser("Preprocess data")
parser.add_argument('--data_json_labels_cancer', type=str, required=True,
        help='Location of the data directory containing json_label files of cancer cases')
parser.add_argument('--data_json_labels_non_cancer', type=str, required=True,
        help='Location of the data directory containing json_label files of non cancer cases')
parser.add_argument('--path_save', type=str, required=True,
        help='Location of the saved file)')
args = parser.parse_args()

# Paths to the input JSON files
file1_path = Path(args.data_json_labels_cancer)
file2_path = Path(args.data_json_labels_non_cancer)

# Path to the output JSON file
output_save_path = Path(args.path_save)
if not os.path.exists(output_save_path):
    os.makedirs(output_save_path)

with open(file1_path, 'r') as file1:
    data1 = json.load(file1)

with open(file2_path, 'r') as file2:
    data2 = json.load(file2)

# Merge the dictionaries from the two files
combined_data = data1 + data2

# Write the combined data to a new JSON file
with open(output_save_path, 'w') as output_file:
    json.dump(combined_data, output_file, indent=4)