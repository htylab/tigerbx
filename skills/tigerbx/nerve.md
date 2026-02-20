# `tigerbx.nerve()` — Hippocampus/Amygdala VAE Embedding

Extracts hippocampus and amygdala ROI patches and encodes them to latent vectors using a
variational autoencoder (VAE). Embeddings can be used for downstream tasks such as
Alzheimer's disease detection.

Developed by Pei-Shin Chen.

```python
import tigerbx

tigerbx.nerve(argstring, input, output=None, model=None, method='NERVE')
```

| Parameter   | Type   | Default   | Description |
|-------------|--------|-----------|-------------|
| `argstring` | `str`  | —         | Operation flags (see table) |
| `input`     | `str`  | —         | NIfTI file, directory, or path to `.npz` files (for decode) |
| `output`    | `str`  | `None`    | Output directory |
| `model`     | `str`  | `None`    | Custom model path |
| `method`    | `str`  | `'NERVE'` | Model variant |

---

## `argstring` flags

| Flag | Description |
|------|-------------|
| `e`  | Encode — produce latent `.npz` files |
| `d`  | Decode — reconstruct patch images from `.npz` files |
| `p`  | Save ROI patch images |
| `v`  | Full evaluation (encode + decode + patches) |
| `s`  | Variable (sigma) reconstruction |
| `g`  | GPU |

---

## Examples

```python
import tigerbx
import numpy as np

# Encode hippocampus/amygdala to latent vectors
tigerbx.nerve('e', 'T1w.nii.gz', 'output/')
# → output/T1w_nerve_L.npz, output/T1w_nerve_R.npz

# Load and inspect a latent vector
import numpy as np
npz = np.load('output/T1w_nerve_L.npz')
latent = npz['z']          # shape: (latent_dim,)
sigma  = npz['sigma']      # uncertainty estimate

# Encode and save ROI patch images
tigerbx.nerve('ep', 'T1w.nii.gz', 'output/')

# Full evaluation (encode + decode + patches)
tigerbx.nerve('v', 'T1w.nii.gz', 'output/')

# Decode previously saved .npz files back to patch images
tigerbx.nerve('d', '/data/nerve_out/', '/data/recon/')

# GPU, whole directory
tigerbx.nerve('eg', '/data/T1w_dir/', '/data/output/')
```

---

## Output file naming

For input `sub-001_T1w.nii.gz`:

| Flag | Output |
|------|--------|
| `e`  | `sub-001_T1w_nerve_L.npz`, `sub-001_T1w_nerve_R.npz` |
| `p`  | `sub-001_T1w_nerve_patch_L.nii.gz`, `sub-001_T1w_nerve_patch_R.nii.gz` |
| `d`  | `sub-001_T1w_nerve_recon_L.nii.gz`, `sub-001_T1w_nerve_recon_R.nii.gz` |

---

## CLI (for simple one-off tasks)

```bash
tiger nerve T1w.nii.gz -e -o output/
tiger nerve T1w.nii.gz -e -p -o output/
tiger nerve T1w.nii.gz -v -o output/
tiger nerve /data/nerve_out/ -d -o /data/recon/
```
