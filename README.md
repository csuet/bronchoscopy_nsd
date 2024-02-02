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
## Step 1: Utilize the script `to_json_label_{Anatomical landmark, Lessons}.py` for generating labels from datasets for both cancer and non-cancer types

+ The input for the script includes the following files:\
`annotation.json`\
`labels.json`\
`objects.json`\
+ Generate two separate sets of labels for the tasks:
`Anatomical Landmarks`\
`Lesions`\
+ Categorize the labels based on cancer and non-cancer types.
+ Save the outputs in the following JSON files:
`labels_Lung_cancer_lesions.json`\
`labels_Anatomical_landmarks.json`\

Example scripts:
```bash
python to_json_label_Anatomical_landmark.py  --data_annots ./data/Data_paper_lung_cancer_test/annotation.json --data_objects ./data/Data_paper_lung_cancer_test/objects.json --data_labels ./data/Data_paper_lung_cancer_test/labels.json  --path_save ./data/Data_paper_lung_cancer_test/labels_Anatomical_landmarks.json

```

- Step 2: The output json from the Step 1 are used as input of the
`combine_json.py` for combining into the full list of image labels.

- Step 3: Use `annots_to_mask.py` with the combined json
and the input images to create the correspondent masks.

After all steps in the process data phase, your data structure looks like this:

```bash
|-- Lung_cancer
|   |-- imgs
|   |   |-- images
|   |-- masks_Lung_cancer_lesions                   <-- After 3rd step
|   |   |-- masks
|   |-- masks_Anatomical_landmarks                  <-- After 3rd step
|   |   |-- masks
|   |-- labels_Lung_cancer_lesions.json             <-- After 1st step
|   |-- labels_Anatomical_landmarks.json            <-- After 1st step
|   |-- annotations.json
|   |-- objects.json
|   |-- labels.json
|-- Non_lung_cancer
|   |-- imgs
|   |   |-- images
|   |-- masks_Lung_cancer_lesions                   <-- After 3rd step
|   |   |-- masks
|   |-- masks_Anatomical_landmarks                  <-- After 3rd step
|   |   |-- masks
|   |-- labels_Lung_cancer_lesions.json             <-- After 1st step
|   |-- labels_Anatomical_landmarks.json            <-- After 1st step
|   |-- annotations.json
|   |-- objects.json
|   |-- labels.json
|-- labels_Lung_cancer_lesions_final.json           <-- After 2nd step
|-- labels_Anatomical_landmarks_final.json          <-- After 2nd step
```


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

