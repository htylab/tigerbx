## Background
This package provides deep-learning segmentation models

## Tutorial using tigerseg

### Install package
    
    pip install git+https://github.com/htylab/tigerseg

## Usage

### As a command line tool:

    tigerseg -i INPUT_FILE -o OUTPUT_DIR --model model --GPU True --report True

INPUT_FILE: For example, t1.nii.gz. A wildcard is allowed. For example, you can use 

    tigerseg -i c:\data\*.nii.gz -o c:\output --model model


For subcortical segmentation:

    aseg -i c:\data\*.nii.gz -o c:\output

For cine cardiac MRI segmentation:

    cine4d -i c:\data\*.nii.gz -o c:\output


### As a python module:

```
from tigerseg import segment

import segment
result = segment.apply_files('cine4d_v0002_xyz_mms12acdc', r"C:\sample\*o.nii")

```
