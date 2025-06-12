# TigerBx: Brain Image Registration and VBM Pipeline

This module offers flexible tools for brain image registration and voxel‑based morphometry (VBM). It supports a variety of affine and nonlinear registration approaches, including both deep‑learning and classical algorithms.
The VBM and registration pipeline was developed by **Pei-Mao Sun**.

---

## 🔧 Command‑line Options

| Option | Description                                          |
| ------ | ---------------------------------------------------- |
| `-b`   | Generate a brain mask                                |
| `-z`   | Save output as `.nii.gz`                             |
| `-A`   | Affine registration to template (default: MNI152)    |
| `-r`   | **VMnet** registration to template (default: MNI152) |
| `-s`   | **SyN** registration to template (antspyx‑based)   |
| `-S`   | **SyNCC** registration to template (antspyx‑based) |
| `-F`   | **FuseMorph** registration to template               |
| `-T`   | Specify a custom template filename                   |
| `-R`   | Rigid registration                                   |
| `-v`   | Run the full VBM analysis pipeline                   |

---

## 🧪 Example Usage

```python
# 1. VMnet registration with C2FViT affine initialization
tigerbx.reg(
    'r',
    r'C:\\T1w_dir',
    r'C:\\output_dir',
    template='template.nii.gz',
    save_displacement=False,
    affine_type='C2FViT'
)

# 2. FuseMorph registration with ANTs affine initialization
tigerbx.reg(
    'F',
    r'C:\\T1w_dir',
    r'C:\\output_dir',
    save_displacement=False,
    affine_type='ANTs'
)

# 3. VBM pipeline with C2FViT affine preprocessing
tigerbx.reg(
    'v',
    r'C:\\T1w_dir\\**\\*.nii.gz',
    r'C:\\output_dir',
    affine_type='C2FViT'
)

# 4. Apply a warp field to a new image
tigerbx.transform(
    r'C:\\T1w_dir\\moving.nii.gz',
    r'C:\\T1w_dir\\warp.npz',
    r'C:\\output_dir',
    interpolation='nearest'
)
```
### CLI Example

```bash
tiger reg -r C:\T1w_dir -o C:\output_dir -T template.nii.gz --affine_type C2FViT
tiger reg -F C:\T1w_dir -o C:\output_dir --affine_type ANTs
tiger reg -v C:\T1w_dir\**\*.nii.gz -o C:\output_dir --affine_type C2FViT
```

---

## ⚙️ Additional Notes

* **Custom template**
  Use `--template your_template.nii.gz` to specify a custom registration template.
* **Saving displacement fields**
  Set `--save_displacement=True` to store displacement fields during registration.

  * ⚠️ The `-s` and `-S` methods save displacement fields by **file path**, not as arrays—take extra care.
* **Affine type** (`--affine_type`)

  * Choices: `C2FViT` (default) or `ANTs`
  * Affects the affine preprocessing step of `-r`, `-F`, and `-v`.
    Methods `-s` and `-S` include their own affine preprocessing and are therefore unaffected.
* **Interpolation**
  The `transform` function currently supports two modes: `nearest` (default) and `linear`.
* **Rigid transform compatibility**
  The `-R` (rigid) option is currently handled independently. When generating displacement fields, do not combine rigid with other methods in the same operation. If you only want to obtain the registered image without saving the displacement field, then combining rigid with other methods is acceptable.

---

## 📚 Recommended Templates

* **MNI152 1 mm**: `mni152_1mm.nii.gz`
* **Custom template**: `--template <your_template>.nii.gz`

---

> For more examples, advanced configuration, and performance tips, please refer to the project documentation or open an issue.
