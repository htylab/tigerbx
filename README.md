# Automated subcortical brain segmentation pipeline
![subcortical Segmentation Example](doc/tumor_segmentation_illusatration.gif)

## Background
This package provides trained 3D U-Net model for subcortical brain segmentation


## Tutorial using SubBrainSegment

### Install package

    pip install tigerseg 

### Install Python 3 and dependencies:
* numpy>=1.16.0
* nibabel>=2.5.1
* tables>=3.6.1
* tensorflow-gpu==2.1.0
* h5py==2.10.0
* [SimpleITK](https://simpleitk.readthedocs.io/en/master/gettingStarted.html)


## Usage

### As a command line tool:

    tigerseg INPUT OUTPUT

If INPUT points to a file, the file will be processed. If INPUT points to a directory, the directory will be searched for the specific format(nii.gz).
OUTPUT is the output directory.

For additional options type:

    tigerseg -h



### As a python module:

```
import tigerseg.segment

input_dir = /your/input/directory
output_dir = /your/output/directory

tigerseg.segment.apply(input=input_dir,output=output_dir)
```
