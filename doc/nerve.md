# TigerBx: `nerve` Module

NERVE (Neuroimaging Embedding Resource for VAE Encoders) was developed by **Pei-Shin Chen**. The project aims to create an open, general-purpose brain MRI embedding model. Large MRI datasets were collected and hippocampus and amygdala patches were extracted to train variational autoencoders. The resulting embeddings (NERVE) can be used for downstream tasks such as Alzheimer’s disease detection using a ResNet classifier.

## Function

`nerve(argstring, input, output=None, model=None, method='NERVE')`

The module extracts ROI patches, encodes them into latent vectors, and optionally decodes them back to NIfTI images.

### Flags

- `e` – Encode input volumes to latent `.npz` files
- `d` – Decode latent files back to patches
- `p` – Save intermediate patch images
- `g` – Use GPU during inference
- `v` – Evaluate reconstruction quality (MAE/MSE/PSNR/SSIM)

### Example

```python
import tigerbx

# Encode and then decode a single T1-weighted scan
# Output files will be placed in the specified directory
result = tigerbx.nerve('e', 'T1w.nii.gz', 'nerve_out')
```
---

Additional Examples
-------------------

```python
# Encode an entire folder on GPU
files = tigerbx.nerve('eg', '/data/T1w/*.nii.gz', 'latent_dir')

# Evaluate reconstruction accuracy from NPZ files
metrics = tigerbx.nerve('v', '/data/npz_dir', 'eval_dir')
```

These examples show how to process multiple inputs, enable GPU inference, and generate quality metrics.
