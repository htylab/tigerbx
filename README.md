### TigerBx: Tissue Mask Generation for Brain Extration

## Background

* This repo provides deep learning methods with pretrained models for brain extraction.
* We also provided the stand-alone application working on Windows, Mac, and Linux.
* The software has been exclusively designed for research purposes and is not intended for any commercial application.
* The software should not be used in clinical applications.

![tigerbet](./doc/tigerbx.png)

### Install stand-alone version
https://github.com/htylab/tigerbx/releases

### Usage

    tigerbx -bmad c:\data\*.nii.gz -o c:\output
    tigerbx -c c:\data\*.nii.gz -o c:\output

### As a python package

    pip install onnxruntime #for gpu version: onnxruntime-gpu
    pip install https://github.com/htylab/tigerbx/archive/release.zip

### For archived versions
    pip install https://github.com/htylab/tigerbx/archive/refs/tags/v0.1.15.tar.gz

### As a python package

    import tigerbx
    tigerbx.run('bmadk', r'C:\T1w_dir', r'C:\output_dir')
    tigerbx.run('bmadk', r'C:\T1w_dir\**\*.nii.gz', r'C:\output_dir')
    tigerbx.run('bmadk', r'C:\T1w_dir\**\*.nii.gz') # storing output in the same dir
    tigerbx.run('dg', r'C:\T1w_dir') # Producing deep-gray-matter masks with GPU


** Mac and Windows  are supported.**

** Ubuntu (version >20.04)  are supported.**

** Typically requires about 1 minute to obtain deep gray matter segmenation without GPU**

```
tigerbx -bmad c:\data\**\*T1w.nii -o c:\outputdir
-m: Produces the brain mask.
-a: Produces the aseg mask.
-b: Produces the extracted brain.
-B: Produces brain age mapping (WIP).
-d: Produces the deep gray matter mask.
-k: Produces the DKT mask (WIP).
-c: Produces the cortical thickness map.
-C: Produces the FSL-style PVEs of CSF, GM, and WM (WIP).
-S: Produces the aseg mask using the SynthSeg-like method (WIP).
-t: Produces the tumor mask (T1 and T1c) (WIP).
-w: Produces the white matter parcellation (WIP).
-W: Produces the white matter hypointensity mask (WIP).
-q: Saves the QC score. Pay attention to QC scores below 50.
-z: Forces storing in nii.gz format.
-A: Affines images to MNI152.
-r: Registers images to MNI152.
```
## Citation

* If you use this application, cite the following paper:

1. Weng JS, Huang TY. Deriving a robust deep-learning model for subcortical brain segmentation by using a large-scale database: Preprocessing, reproducibility, and accuracy of volume estimation. NMR Biomed. 2022 Nov 23:e4880. doi: 10.1002/nbm.4880. (https://doi.org/10.1002/nbm.4880)
2. Chen CS, Huang TY "Accelerated Cortical Thickness Mapping Using Deep Learning", ISMRM 2024
3. Wang HC, Chen CS, Kuo CC, Huang TY, Kuo KH, Chuang TC, Lin YR, Chung HW (2024) “Comparative Assessment of Established and Deep Learning Segmentation Methods for Hippocampal Volume Estimation in Brain MRI Analysis” NMR in Biomedicine. 2024;e5169. doi:10.1002/nbm.5169

## Label definitions

For label definitions, please check here. [Label definitions](doc/seglabel.md)

## Validation methods

For validation methods, please check here. [Validation](doc/validation.md)

## Disclaimer

The software has been exclusively designed for research purposes and has not undergone review or approval by the Food and Drug Administration or any other agency. By using this software, you acknowledge and agree that it is not recommended nor advised for clinical applications.  You also agree to use, reproduce, create derivative works of, display, and distribute the software in compliance with all applicable governmental laws, regulations, and orders, including but not limited to those related to export and import control.

The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the contributors be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software. Use of the software is at the recipient's own risk.


