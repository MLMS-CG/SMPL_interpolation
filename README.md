# SMPL_interpolation
This repo provides a torch implementation of the SMPL model with the interpolation from the T pose to any pose.

## Installation
```bash
pip install torch
pip install numpy
pip install pickle
pip install joblib
pip install tqdm
pip install trimesh
```

## Download data
SMPL data
Download the auxiliary data for SMPL using this [link](https://toNas). Unpack it anywhere you want and set the ROOT global variable to the path of the unpacked folder in global_var.py. 

## Example: generate a sequence of SMPL meshes & interpolate from the T pose to the pose of its first frame
The motion sequences are provided in the form of SMPL parameters (thetas for pose and betas for shape) in samples_test.pt
```bash
python interpolate_to_T.py 
```




