### TigerBx: Tissue Mask Generation for Brain Extration

## Background


* This repo provides deep learning methods with pretrained models for brain extraction.
* We also provided the stand-alone application working on Windows, Mac, and Linux.
* The software has been exclusively designed for research purposes. 
* It is not recommended nor advised for clinical applications.

![tigerbet](./doc/tigerbx.png)

### Install stand-alone version
https://github.com/htylab/tigerbx/releases

### Usage

    tigerbx -bmad c:\data\*.nii.gz -o c:\output
    tigerbx -c c:\data\*.nii.gz -o c:\output

### As a python package

    pip install onnxruntime #for gpu version: onnxruntime-gpu
    pip install https://github.com/htylab/tigerbx/archive/release.zip

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
>>tigerbet  c:\data\**\*T1w.nii -o c:\outputdir -b -m -a -d -f
-b: Producing the extracted brain.
-m: Producing the brain mask.
-a: Producing the aseg mask.
-d: Producing the deep gray matter mask.
-k: Producing the DKT mask (work in progress).
-c: Producing the cortical thickness map.
-C: Producing the FSL-style PVEs of CSF, GM, WM.
-w: Producing the white matter parcellation (work in progress).
-l: Producing the white-matter hyperintensity mask (WMH) (WIP).
-q: Save the QC score. Pay attention to QC scores less than 50.
-z: Force storing in nii.gz format.
```
## Citation

* If you use this application, cite the following paper:

1. Weng JS, Huang TY. Deriving a robust deep-learning model for subcortical brain segmentation by using a large-scale database: Preprocessing, reproducibility, and accuracy of volume estimation. NMR Biomed. 2022 Nov 23:e4880. doi: 10.1002/nbm.4880. (https://doi.org/10.1002/nbm.4880)


## Label definitions

For label definitions, please check here. [Label definitions](doc/seglabel.md)

## Disclaimer

The software has been exclusively designed for research purposes and has not undergone review or approval by the Food and Drug Administration or any other agency. By using this software, you acknowledge and agree that it is not recommended nor advised for clinical applications.  You also agree to use, reproduce, create derivative works of, display, and distribute the software in compliance with all applicable governmental laws, regulations, and orders, including but not limited to those related to export and import control.

The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the contributors be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software. Use of the software is at the recipient's own risk.


