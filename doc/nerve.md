# TigerBx: `nerve` Module

NERVE (Neuroimaging Embedding Resource for VAE Encoders) was developed by **Pei-Shin Chen**. It extracts hippocampus and amygdala patches from T1-weighted images and encodes them into compact latent vectors using a variational autoencoder (VAE). The embeddings can be used for downstream tasks such as Alzheimer's disease detection.

---

## Python API

```python
tigerbx.nerve(argstring, input, output=None, model=None, method='NERVE')
```

| Parameter   | Type            | Default    | Description |
|-------------|-----------------|------------|-------------|
| `argstring` | `str`           | —          | One or more flag characters (see table below) |
| `input`     | `str` or `list` | —          | Input NIfTI file, directory, or glob pattern; or a directory of `.npz` files when decoding |
| `output`    | `str`           | `None`     | Output directory; if `None`, saves next to each input file |
| `model`     | `str` or `dict` | `None`     | Custom model override; `None` uses bundled defaults |
| `method`    | `str`           | `'NERVE'`  | Embedding method |

---

## CLI Usage

```
tiger nerve <input> [input ...] [-o OUTPUT] [-e] [-d] [-p] [-v] [-g] [-s]
```

---

## Flags

| API flag | CLI flag | Description |
|----------|----------|-------------|
| `e`      | `-e`     | Encode input volumes to latent `.npz` files |
| `d`      | `-d`     | Decode latent `.npz` files back to patch images |
| `p`      | `-p`     | Save intermediate ROI patch images (`.nii.gz`) |
| `v`      | `-v`     | Evaluate reconstruction quality — automatically enables `e`, `d`, and `p` |
| `g`      | `-g`     | Use GPU for inference |
| `s`      | `-s`     | Use variable (sigma) reconstruction |

Additional CLI-only options:

| CLI flag          | Description |
|-------------------|-------------|
| `--model MODEL`   | Specify a custom ONNX model |
| `--method METHOD` | Specify the embedding method (default: `NERVE`) |

---

## Examples

### Python API

```python
import tigerbx

# Encode a single scan to latent vectors
tigerbx.nerve('e', 'T1w.nii.gz', 'output_dir')

# Encode and save ROI patch images
tigerbx.nerve('ep', 'T1w.nii.gz', 'output_dir')

# Encode then decode (round-trip) and save patches
tigerbx.nerve('edp', 'T1w.nii.gz', 'output_dir')

# Evaluate reconstruction quality (automatically enables encode, decode, and patch saving)
tigerbx.nerve('v', 'T1w.nii.gz', 'output_dir')

# Encode an entire directory using GPU
tigerbx.nerve('eg', '/data/T1w_dir', '/data/nerve_out')

# Decode previously saved latent .npz files
tigerbx.nerve('d', '/data/nerve_out', '/data/recon_out')
```

### CLI

```bash
# Encode a single scan to latent vectors
tiger nerve T1w.nii.gz -e -o output_dir

# Encode and save ROI patch images
tiger nerve T1w.nii.gz -e -p -o output_dir

# Encode then decode (round-trip) and save patches
tiger nerve T1w.nii.gz -e -d -p -o output_dir

# Evaluate reconstruction quality
tiger nerve T1w.nii.gz -v -o output_dir

# Encode an entire directory using GPU
tiger nerve /data/T1w_dir -e -g -o /data/nerve_out

# Decode previously saved .npz files
tiger nerve /data/nerve_out -d -o /data/recon_out
```

---

## Output Files

For an input named `sub-001_T1w.nii.gz`:

| Flag | Output file |
|------|-------------|
| `-e` | `sub-001_T1w_nerve.npz` (latent vectors for left and right ROIs) |
| `-p` | ROI patch images (`.nii.gz`) for hippocampus and amygdala |
| `-d` | Reconstructed patch images (`.nii.gz`) |
