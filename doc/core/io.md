# `tigerbx.core.io` — NIfTI I/O and Path Utilities

Shared I/O helpers for input resolution, output path templating, and NIfTI file writing.

---

## Input Resolution

### `resolve_nifti_inputs(input_arg)`

Resolve an input argument to a list of NIfTI file paths.

| Parameter   | Type   | Description |
|-------------|--------|-------------|
| `input_arg` | `list` | A list containing file paths, a directory path, or a glob pattern |

**Behaviour:**
- If `input_arg[0]` is a directory, globs for `*.nii` and `*.nii.gz` inside it (single level only).
- If `input_arg[0]` contains `*`, expands the glob pattern.
- Otherwise, returns `input_arg` as-is.

**Returns:** `list[str]` — resolved file paths.

---

### `detect_common_folder(input_file_list)`

When multiple input files share the same basename (e.g. files from different subject directories), returns their `os.path.commonpath`. This prefix is prepended to output filenames to avoid collisions.

| Parameter         | Type         | Description |
|-------------------|--------------|-------------|
| `input_file_list` | `list[str]`  | List of input file paths |

**Returns:** `str | None` — common path if duplicate basenames exist, else `None`.

---

### `resolve_inputs(input_arg)`

Convenience wrapper: calls `resolve_nifti_inputs` then `detect_common_folder`.

**Returns:** `(list[str], str | None)` — `(input_file_list, common_folder)`.

```python
from tigerbx.core.io import resolve_inputs

file_list, cf = resolve_inputs(['/data/T1w_dir'])
```

---

## Output Path Template

### `get_template(f, output_dir, gz=None, common_folder=None)`

Build an output path template containing the `@@@@` placeholder for `save_nib`.

| Parameter       | Type         | Default | Description |
|-----------------|--------------|---------|-------------|
| `f`             | `str`        | —       | Input file path |
| `output_dir`    | `str | None` | —       | Output directory; `None` = same directory as input |
| `gz`            | `bool | None`| `None`  | `True`/`False`: include `.nii[.gz]` extension. `None`: return stem only (no extension) |
| `common_folder` | `str | None` | `None`  | Common path prefix for deduplication |

**Returns:** `(str, str)` — `(ftemplate, f_output_dir)`.

- `ftemplate` is a **full path** (includes `f_output_dir`), e.g. `/output/sub-001_T1w_@@@@.nii.gz`
- `f_output_dir` is the resolved output directory

```python
from tigerbx.core.io import get_template

ftemplate, out_dir = get_template('/data/sub-001_T1w.nii.gz', '/output', gz=True)
# ftemplate = '/output/sub-001_T1w_@@@@.nii.gz'
# out_dir   = '/output'
```

---

### `save_nib(data_nib, ftemplate, postfix)`

Save a nibabel image by replacing `@@@@` in the template with the given postfix.

| Parameter  | Type              | Description |
|------------|-------------------|-------------|
| `data_nib` | `nib.Nifti1Image` | Image to save |
| `ftemplate`| `str`             | Path template from `get_template()` |
| `postfix`  | `str`             | Label to replace `@@@@` (e.g. `'tbx'`, `'tseg'`) |

**Returns:** `str` — the output file path.

```python
from tigerbx.core.io import get_template, save_nib

ftemplate, _ = get_template(f, output_dir, gz=True)
fn = save_nib(result_nib, ftemplate, 'tbxmask')
# → '/output/sub-001_T1w_tbxmask.nii.gz'
```
