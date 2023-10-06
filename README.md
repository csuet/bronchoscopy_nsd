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
- Step 1: To create the data for further training and testing, firstly,
using the file `to_json_label_{Anatomical landmark,Lessons}.py`
to create json for Lung cancer type and Non lung cancer
by providing the metadata `annotation.json`, `labels.json`, `objects.json`

- Step 2: The output json from the Step 1 are used as input of the
`combine_json.py` for combining into the full list of image labels.

- Step 3: Use `annots_to_mask.py` with the combined json
and the input images to create the correspondent masks.

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

