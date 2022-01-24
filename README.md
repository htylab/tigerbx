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
* h5py==2.10.0
* onnxruntime>=1.8.0
* [SimpleITK](https://simpleitk.readthedocs.io/en/master/gettingStarted.html)


## Usage

### As a command line tool:

    tigerseg INPUT OUTPUT --CPU CPU --onnx ONNX --seg_mode SEG_MODE --report REPORT

If INPUT points to a file, the file will be processed. If INPUT points to a directory, the directory will be searched for the specific format(nii.gz).
OUTPUT is the output directory.
If you don't want to use onnxruntime version, you can set ONNX to False.
SEG_MODE: 0: Subcortical Brain Segmentation mode. 1: Brain Tumor Segmentation mode. 2: Nasopharyngeal Carcinoma Segmentation mode.
If you need some report of the Segmentation, set the REPORT to True.

For additional options type:

    tigerseg -h



### As a python module:

```
import tigerseg.segment

input_dir = /your/input/directory
output_dir = /your/output/directory

tigerseg.segment.apply(input=input_dir,output=output_dir)   # For tensorflow

tigerseg.segment.onnx_apply(input=input_dir,output=output_dir,only_CPU=False,seg_mode=0,report_enabled=True) # For onnxruntime
```
