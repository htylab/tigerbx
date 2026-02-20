# `tiger nerve` — Hippocampus/Amygdala VAE Embedding

Extracts hippocampus and amygdala ROI patches and encodes them to latent vectors using a
variational autoencoder (VAE). Embeddings can be used for downstream tasks such as
Alzheimer's disease detection.

Developed by Pei-Shin Chen.

```
tiger nerve <input> [input ...] [-o OUTPUT] [-e] [-d] [-p] [-v] [-s] [-g]
```

---

## Flags

| Flag | Description |
|------|-------------|
| `-e` | Encode — produce latent `.npz` files |
| `-d` | Decode — reconstruct patch images from `.npz` files |
| `-p` | Save ROI patch images |
| `-v` | Full evaluation (encode + decode + patches) |
| `-s` | Variable (sigma) reconstruction |
| `-g` | GPU |

---

## Examples

```bash
# Encode to latent vectors
tiger nerve T1w.nii.gz -e -o output/

# Encode + save patch images
tiger nerve T1w.nii.gz -e -p -o output/

# Full evaluation (encode + decode + patches)
tiger nerve T1w.nii.gz -v -o output/

# GPU, whole directory
tiger nerve /data/T1w_dir -e -g -o /data/output/

# Decode previously saved .npz files
tiger nerve /data/nerve_out -d -o /data/recon/
```

---

## Output naming

For input `sub-001_T1w.nii.gz`:

| Flag | Output |
|------|--------|
| `-e` | `sub-001_T1w_nerve_L.npz`, `sub-001_T1w_nerve_R.npz` (left/right latent vectors) |
| `-p` | `sub-001_T1w_nerve_patch_L.nii.gz`, `_patch_R.nii.gz` (ROI patches) |
| `-d` | `sub-001_T1w_nerve_recon_L.nii.gz`, `_recon_R.nii.gz` (reconstructions) |
