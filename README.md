# BM-BronchoLC

This is the codebase for preprocessing data and training
unitask models and multitasks models.
The data is host at figshare with the following link:

+ The `Process_data` folder contains the code for 
reading the raw input images and linking with the three
metadata files to create the labels for anatomical landmarks
and lessons.
+ The `Unet2+_base` data folder contains the code for
training unitask models and multitasks models with Unet2+
architecture
+ The `ESFPNet_base` data folder contains the code for
training unitask models and multitasks models with Unet2+
architecture

# Process data
### Step 1: Utilize the script `to_json_label_{Anatomical landmark, Lesions}.py` for generating labels from datasets for both cancer and non-cancer types

The input for the script includes the following files:
+ `annotation.json`
+ `labels.json`
+ `objects.json`\
Generate two separate sets of labels for the tasks:
+ `Anatomical Landmarks`
+ `Lesions`\
Categorize the labels based on cancer and non-cancer types.\
Save the outputs in the following JSON files:
+ `labels_Lung_lesions.json`
+ `labels_Anatomical_landmarks.json`

Example scripts:
```bash
python to_json_label_Anatomical_landmark.py  --data_annots ./data/Lung_cancer/annotation.json --data_objects ./data/Lung_cancer/objects.json --data_labels ./data/Lung_cancer/labels.json  --path_save ./data/Lung_cancer/labels_Anatomical_landmarks.json
```

### Step 2: Execute the script `combine_json.py` to combine labels from both cancer and non-cancer cases for Lesions and Anatomical Landmarks tasks.

The script requires the following input JSON files:
+ `labels_Lung_lesions.json (cancer)`
+ `labels_Lung_lesions.json (non-cancer)`
+ `labels_Anatomical_landmarks.json (cancer)`
+ `labels_Anatomical_landmarks.json (non-cancer)`\

Merge the labels for both cancer and non-cancer cases for each task.\

Save the combined outputs in the following JSON files:
+ `labels_Lung_lesions_final.json`
+ `labels_Anatomical_landmarks_final.json`

Example scripts:
```bash
python combine_json.py --data_json_labels_cancer ./data/Lung_cancer/labels_Lung_lesions.json --data_json_labels_non_cancer ./data/Non_lung_cancer/labels_Lung_lesions.json --path_save ./data/labels_Lung_lesions_final.json
```

### Step 3: Run the script `annots_to_masks.py` to convert annotations into ground truth images for both Lesions and Anatomical Landmarks tasks, considering cancer and non-cancer types.

The script requires the following inputs:
+ `annotation.json`
+ `labels.json`
+ `objects.json`
+ `Type of tasks` (specify either "lesions" or "anatomical landmarks")\
Based on the specified task type, generate masks (ground truth) for image segmentations (both cancer and non-cancer cases)\
Save the resulting masks as outputs, representing the ground truth for the segmentation of images.\
```bash
|-- Lung_cancer
|   |-- imgs
|   |   |-- images
|   |-- masks_Lung_lesions                  
|   |   |-- masks
|   |-- masks_Anatomical_landmarks                 
|   |   |-- masks
|-- Non_lung_cancer
|   |-- imgs
|   |   |-- images
|   |-- masks_Lung_lesions                   
|   |   |-- masks
|   |-- masks_Anatomical_landmarks                  
|   |   |-- masks
```

Example scripts:
```bash
python annots_to_mask.py --data_annots ./data/Lung_cancer/annotation.json --data_objects ./data/Lung_cancer/objects.json --data_labels ./data/Lung_cancer/labels.json --path_save ./data/Lung_cancer/masks_Lung_lesions --type label_Lesions
```

After all steps in the process data phase, your data structure looks like this:

```bash
|-- Lung_cancer
|   |-- imgs
|   |   |-- images
|   |-- masks_Lung_lesions                          <-- After 3rd step
|   |   |-- masks
|   |-- masks_Anatomical_landmarks                  <-- After 3rd step
|   |   |-- masks
|   |-- labels_Lung_lesions.json                    <-- After 1st step
|   |-- labels_Anatomical_landmarks.json            <-- After 1st step
|   |-- annotations.json
|   |-- objects.json
|   |-- labels.json
|-- Non_lung_cancer
|   |-- imgs
|   |   |-- images
|   |-- masks_Lung_lesions                          <-- After 3rd step
|   |   |-- masks
|   |-- masks_Anatomical_landmarks                  <-- After 3rd step
|   |   |-- masks
|   |-- labels_Lung_lesions.json                    <-- After 1st step
|   |-- labels_Anatomical_landmarks.json            <-- After 1st step
|   |-- annotations.json
|   |-- objects.json
|   |-- labels.json
|-- labels_Lung_lesions_final.json                  <-- After 2nd step
|-- labels_Anatomical_landmarks_final.json          <-- After 2nd step
```

### Step 4: Execute the script `split_dataset.py` to perform the dataset split for images and masks related to Anatomical Landmarks or Lung Cancer Lesions.
The script requires the following input parameters:
+ `Labels JSON file` (`labels_Lung_lesions_final.json` or `labels_Anatomical_landmarks_final.json`)
+ `Folder containing cancer images` (`./Lung_cancer/imgs`)
+ `Folder containing cancer masks` (`./Lung_cancer/masks_Lung_lesions` or `./Lung_cancer/masks_Anatomical_landmarks`)
+ `Folder containing non-cancer images` (`./Non_lung_cancer/imgs`)
+ `Folder containing non-cancer masks` (`./Non_lung_cancer/masks_Lung_lesions` or `./Non_lung_cancer/masks_Anatomical_landmarks`)\
The dataset will be split into training, validation, and test sets\
Organize the outputs into a "dataset" folder, which includes subfolders for train, val, and test. Each of these subfolders comprise two subdirectories: one for images and another for masks.


# Training with Unet2+ models

- Using the raw imgs input and the output label files in the `Process data`
folder for training with `train_clf_*.py`. And using the output masks for training with `train_segment_Unet2+.py`

- For the joint training of `train_joint_*.py`, it is required to use both the output label json and the output masks for training

- The `infer_*.py` is used to extract the correspondent output from the trained model.
# Training with ESFPNet models

- Using the raw imgs input and the output label files in the `Process data`
folder for training with `train_clf_*.py`. And using the output masks for training with `train_segment_ESFPNet.py`

- For the joint training of `train_joint_*.py`, it is required to use both the output label json and the output masks for training

- The `infer_*.py` is used to extract the correspondent output from the trained model.

