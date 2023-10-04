import json
# Combine two json files
# Paths to the input JSON files
file1_path = '/workspace/ailab/phucnd/04102023_Nhom_benh/json_label_TTKV_final.json'
file2_path = '/workspace/ailab/phucnd/04102023_Nhom_chung/json_label_TTKV_final.json'

# Path to the output JSON file
output_path = '/workspace/ailab/phucnd/labels_TTKV.json'

with open(file1_path, 'r') as file1:
    data1 = json.load(file1)

with open(file2_path, 'r') as file2:
    data2 = json.load(file2)

# Merge the dictionaries from the two files
combined_data = data1 + data2

# Write the combined data to a new JSON file
with open(output_path, 'w') as output_file:
    json.dump(combined_data, output_file, indent=4)