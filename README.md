## Background
This package provides deep-learning segmentation models

## Tutorial using tigerseg

### Install package

    pip install git+https://github.com/htylab/tigerseg

## Usage

### As a command line tool:

    tigerseg INPUT_FILE OUTPUT_DIR --modelname model --GPU True --report True

INPUT_FILE: For example, t1.nii.gz. A wildcard is allowed. For example, you can use 

    tigerseg c:\data\*.nii.gz c:\output --modelname model


For subcortical segmentation:

    aseg c:\data\*.nii.gz c:\output

For cine cardiac MRI segmentation:

    cine4d c:\data\*.nii.gz c:\output


### As a python module:

```
from tigerseg import segment

import segment
result = segment.apply('cine4d_v0002_xyz_mms12acdc', r"C:\expdata\nchu_cine\sample\*o.nii")

```
