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
To create the data for further training and testing, firstly,
using the file `to_json_label`

