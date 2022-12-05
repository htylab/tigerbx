## Background


* This repo provides deep-learning methods for brain extration.
* We also provided the stand-alone application working on Windows, Mac, and Linux.

## Citation

* If you use this application, cite the following paper:

1. Weng JS, Huang TY. Deriving a robust deep-learning model for subcortical brain segmentation by using a large-scale database: Preprocessing, reproducibility, and accuracy of volume estimation. NMR Biomed. 2022 Nov 23:e4880. doi: 10.1002/nbm.4880.

![tigerbet](./doc/tigerbet2.png)

### Install stand-alone version
    https://github.com/htylab/tigerseg/releases

### Usage

    tigerbx -bmadf c:\data\*.nii.gz -o c:\output

** Mac and Windows  are supported.**
** Ubuntu (version >18.04)  are supported.**
** Typically requires about 1 minute to obtain deep gray matter segmenation without GPU**


```tigerbet  c:\data\**\*T1w.nii -o c:\outputdir -b -m -a -d -f```
-b: producing extracted brain
-m: storing the brain mask
-a: producing the aseg mask
-d: producing the deep gray mater mask
-f: faster operation with low-resolution models



## ASEG43
| Label | Structure              | Label | Structure               |
| ----- | ---------------------- | ----- | ----------------------- |
| 2     | Left Cerebral WM       | 41    | Right Cerebral WM       |
| 3     | Left Cerebral Cortex   | 42    | Right Cerebral Cortex   |
| 4     | Left Lateral Ventricle | 43    | Right Lateral Ventricle |
| 5     | Left Inf Lat Vent      | 44    | Right Inf Lat Vent      |
| 7     | Left Cerebellum WM     | 46    | Right Cerebellum WM     |
| 8     | Left Cerebellum Cortex | 47    | Right Cerebellum Cortex |
| 10    | Left Thalamus          | 49    | Right Thalamus          |
| 11    | Left Caudate           | 50    | Right Caudate           |
| 12    | Left Putamen           | 51    | Right Putamen           |
| 13    | Left Pallidum          | 52    | Right Pallidum          |
| 14    | 3rd Ventricle          | 53    | Right Hippocampus       |
| 15    | 4th Ventricle          | 54    | Right Amygdala          |
| 16    | Brain Stem             | 58    | Right Accumbens area    |
| 17    | Left Hippocampus       | 60    | Right VentralDC         |
| 18    | Left Amygdala          | 62    | Right vessel            |
| 24    | CSF                    | 63    | Right choroid plexus    |
| 26    | Left Accumbens area    | 77    | WM hypointensities      |
| 28    | Left VentralDC         | 85    | Optic Chiasm            |
| 30    | Left vessel            | 251   | CC Posterior            |
| 31    | Left choroid plexus    | 252   | CC Mid Posterior        |
|       |                        | 253   | CC Central              |
|       |                        | 254   | CC Mid Anterior         |
|       |                        | 255   | CC Anterior             |

## DeepGM: Deep gray-matter structures
| Label No. | Structure Name       | Label No. | Structure Name        |
| --------- | -------------------- | --------- | --------------------- |
| 1         | Left-Thalamus-Proper | 2         | Right-Thalamus-Proper |
| 3         | Left-Caudate         | 4         | Right-Caudate         |
| 5         | Left-Putamen         | 6         | Right-Putamen         |
| 7         | Left-Pallidum        | 8         | Right-Pallidum        |
| 9         | Left-Hippocampus     | 10        | Right-Hippocampus     |
| 11        | Left-Amygdala        | 12        | Right-Amygdala        |
