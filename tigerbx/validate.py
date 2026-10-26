from os.path import basename, join, dirname
import numpy as np
import glob
import nibabel as nib
import tigerbx
from tigerbx import lib_reg
from tigerbx.core.io import resolve_nifti_inputs
import sys
import os
import csv
import json
import random
import tempfile

# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(os.path.abspath(__file__))

_LITE_N = 20    # default files per dataset in lite mode
_FAILED = []    # (file, error) of the dataset being validated; one bad file no longer drops the whole dataset


# ── metric helpers ────────────────────────────────────────────────────────────

def getdice(mask1, mask2):
    denom = np.sum(mask1) + np.sum(mask2)
    if denom == 0:
        return 1.0          # both empty → perfect agreement by convention
    return 2 * np.sum(mask1 & mask2) / denom


def get_dice12(gt, pd_arr, model_str):
    iigt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    if model_str == 'dgm':
        iipd = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    else:
        iipd = [10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54]
    return np.array([getdice(gt == a, pd_arr == b) for a, b in zip(iigt, iipd)])


def get_dice26(gt, pd_arr):
    iigt = [2, 41, 3, 42, 4, 43, 7, 46, 8, 47, 10, 49, 11, 50,
            12, 51, 13, 52, 17, 53, 18, 54, 28, 60, 16, 24]
    return np.array([getdice(gt == lbl, pd_arr == lbl) for lbl in iigt])


# ── loop / IO helpers ─────────────────────────────────────────────────────────

def _run_loop(ffs, compute_fn, debug=False, files_filter=None):
    """
    Iterate over *ffs* calling ``compute_fn(f, tmp_dir)`` for each file.

    - files_filter : set of absolute paths; restricts *ffs* to these files.
                     Takes priority over *debug*.
    - debug        : if True (and files_filter is None), limit to first 5 files.

    All prediction outputs written to *tmp_dir* are deleted automatically.
    Returns *(f_list, results_list)*.
    """
    if files_filter is not None:
        ffs = [f for f in ffs if os.path.abspath(f) in files_filter]
    elif debug:
        ffs = ffs[:5]
    print(f'Total files: {len(ffs)}')
    f_list, results = [], []
    with tempfile.TemporaryDirectory() as _tmp:
        for f in ffs:
            try:
                val = compute_fn(f, _tmp)
            except Exception as exc:
                _record_failure(f, exc)
                continue
            f_list.append(f)
            results.append(val)
    return f_list, results


def _record_failure(f, exc):
    print(f'  FAILED {basename(f)}: {type(exc).__name__}: {exc}')
    _FAILED.append((f, f'{type(exc).__name__}: {exc}'))


def _nib_data(res, key):
    """Return fdata from a result entry that is either a nib object or a file path."""
    val = res[key]
    return nib.load(val).get_fdata() if isinstance(val, str) else val.get_fdata()


def _run_bx_batch(ffs, argstr, GPU, model, compute_metrics_fn,
                  debug=False, files_filter=None):
    """
    Run all files in one ``tigerbx.run()`` call (session-cache benefit),
    then compute metrics per file via *compute_metrics_fn(f, res)*.

    *res* is result_filedict (paths) for N>1 files, result_dict (nib objects)
    for a single file — ``_nib_data()`` handles both transparently.

    Returns *(f_list, results_list)*.
    """
    if files_filter is not None:
        ffs = [f for f in ffs if os.path.abspath(f) in files_filter]
    elif debug:
        ffs = ffs[:5]
    print(f'Total files: {len(ffs)}')
    if not ffs:
        return [], []
    f_list, results = [], []
    with tempfile.TemporaryDirectory() as _tmp:
        try:
            batch = tigerbx.run(argstr, ffs, _tmp, model=model)
            if not isinstance(batch, list):
                batch = [batch]   # single-file path returns dict, not list
        except Exception as exc:
            # one file broke the batch run: redo file by file so the others still count
            print(f'  batch run failed ({type(exc).__name__}: {exc}); retrying file by file')
            batch = []
            for f in ffs:
                try:
                    batch.append(tigerbx.run(argstr, [f], _tmp, model=model))
                except Exception as exc_f:
                    _record_failure(f, exc_f)
                    batch.append(None)
        for f, res in zip(ffs, batch):
            if res is None:
                continue
            try:
                val = compute_metrics_fn(f, res)
            except Exception as exc:
                _record_failure(f, exc)
                continue
            f_list.append(f)
            results.append(val)
    return f_list, results


def _write_csv(output_dir, filename, header, f_list, rows):
    """Write a CSV using the stdlib *csv* module (no pandas required)."""
    if output_dir is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    with open(join(output_dir, filename), 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['Filename'] + header)
        for fname, row in zip(f_list, rows):
            if isinstance(row, (list, tuple)):
                vals = list(row)
            elif hasattr(row, '__len__'):
                vals = np.asarray(row).tolist()
            else:
                vals = [float(row)]
            writer.writerow([fname] + vals)


def _print_col_means(column_names, mean_per_column):
    """Print per-column mean Dice in a compact aligned format."""
    w = max(len(c) for c in column_names)
    for col in column_names:
        print(f'  {col:<{w}} : {mean_per_column[col]:.4f}')
    overall = float(np.mean(list(mean_per_column.values())))
    print(f'  {"Overall mean":<{w}} : {overall:.4f}')


# ── registration pipeline helper ──────────────────────────────────────────────

# ── individual validation functions ──────────────────────────────────────────

def val_tbx_strip2(input_dir, output_dir=None, GPU=False,
                     debug=False, files_filter=None,
                     tbx_model=None, seg_model=None, **kwargs):
    ffs = resolve_nifti_inputs([join(input_dir, '*', 'image.nii.gz')])
    run_model = {'tbx': tbx_model} if tbx_model else None
    argstr = ('g' if GPU else '') + 'm'

    tt_list, cat_list = [], []

    def compute_metrics(f, res):
        tt_list.append(basename(dirname(f)).split('_')[1])
        cat_list.append('_'.join(basename(dirname(f)).split('_')[:2]))
        mask_pred = _nib_data(res, 'tbxmask')
        mask_gt   = nib.load(f.replace('image.nii.gz', 'mask.nii.gz')).get_fdata()
        return getdice((mask_pred > 0).flatten(), (mask_gt > 0).flatten())

    f_list, dsc_list = _run_bx_batch(ffs, argstr, GPU, run_model, compute_metrics,
                                     debug=debug, files_filter=files_filter)

    _write_csv(output_dir, 'val_tbx_strip2.csv',
               ['type', 'category', 'DICE'],
               f_list,
               [[t, c, d] for t, c, d in zip(tt_list, cat_list, dsc_list)])

    # per-category mean Dice (stdlib + numpy; no pandas required)
    from collections import defaultdict
    cat_dice = defaultdict(list)
    for cat, dice in zip(cat_list, dsc_list):
        cat_dice[cat].append(dice)
    for cat in sorted(cat_dice):
        print(f'  {cat}: {np.mean(cat_dice[cat]):.4f}')
    metric = float(np.mean(dsc_list))
    print(f'mean Dice of all data: {metric:.4f}')
    data = {'Filename': f_list, 'type': tt_list,
            'category': cat_list, 'DICE': dsc_list}
    return data, metric


def val_tbx_strip1(input_dir, output_dir=None, GPU=False,
                 debug=False, files_filter=None,
                 tbx_model=None, seg_model=None, **kwargs):
    ffs = resolve_nifti_inputs([join(input_dir, '*', '*T1w.nii.gz')])
    run_model = {'tbx': tbx_model} if tbx_model else None
    argstr = ('g' if GPU else '') + 'm'

    def compute_metrics(f, res):
        mask_pred = _nib_data(res, 'tbxmask')
        mask_gt   = nib.load(f.replace('T1w.nii.gz', 'T1w_brainmask.nii.gz')).get_fdata()
        return getdice((mask_pred > 0).flatten(), (mask_gt > 0).flatten())

    f_list, dsc_list = _run_bx_batch(ffs, argstr, GPU, run_model, compute_metrics,
                                     debug=debug, files_filter=files_filter)

    _write_csv(output_dir, 'val_tbx_strip1.csv', ['DICE'], f_list, [[d] for d in dsc_list])

    metric = float(np.mean(dsc_list))
    print(f'mean Dice of all data: {metric:.4f}')
    return {'Filename': f_list, 'DICE': dsc_list}, metric


def _val_seg_123(model_str, run_option, input_dir, output_dir=None,
                 GPU=False, debug=False, files_filter=None,
                 tbx_model=None, seg_model=None, **kwargs):
    column_names = ['Left-Thalamus',    'Right-Thalamus',
                    'Left-Caudate',     'Right-Caudate',
                    'Left-Putamen',     'Right-Putamen',
                    'Left-Pallidum',    'Right-Pallidum',
                    'Left-Hippocampus', 'Right-Hippocampus',
                    'Left-Amygdala',    'Right-Amygdala']
    ffs = resolve_nifti_inputs([join(input_dir, 'raw123', '*.nii.gz')])
    run_model = {}
    if tbx_model: run_model['tbx'] = tbx_model
    if seg_model: run_model[model_str] = seg_model
    run_model = run_model or None
    argstr = ('g' if GPU else '') + run_option

    def compute_metrics(f, res):
        mask_pred = _nib_data(res, model_str).astype(int)
        mask_gt   = nib.load(f.replace('raw123', 'label123')).get_fdata().astype(int)
        return get_dice12(mask_gt, mask_pred, model_str)

    f_list, dsc_list = _run_bx_batch(ffs, argstr, GPU, run_model, compute_metrics,
                                     debug=debug, files_filter=files_filter)
    _write_csv(output_dir, f'val_{model_str}_123.csv', column_names, f_list, dsc_list)

    dsc_array       = np.array(dsc_list)
    mean_per_column = dict(zip(column_names, dsc_array.mean(axis=0).tolist()))
    _print_col_means(column_names, mean_per_column)
    data = {'Filename': f_list,
            **{col: dsc_array[:, i].tolist() for i, col in enumerate(column_names)}}
    return data, mean_per_column


# seg18 manual labels (FreeSurfer numbers) vs the tseg prediction: structure -> (seg18 labels, predicted labels).
# seg18 cerebral WM includes the corpus callosum and has no WM-hypointensity class, so WM is scored bilaterally
# with 77 and 251-255; seg18 has no choroid plexus (31 / 63 count as lateral ventricle); optic chiasm (85) is
# background. CSF and vessels (*) are protocol-dependent: reported, left out of the 30-structure mean.
SEG18_STRUCTURES = {
    'Cerebral-WM': ((2, 41), (2, 41, 77, 251, 252, 253, 254, 255)),
    'L-Cortex': ((3,), (3,)), 'R-Cortex': ((42,), (42,)),
    'L-Lat-Vent': ((4,), (4, 31)), 'R-Lat-Vent': ((43,), (43, 63)),
    'L-Inf-Lat-Vent': ((5,), (5,)), 'R-Inf-Lat-Vent': ((44,), (44,)),
    'L-Cblm-WM': ((7,), (7,)), 'R-Cblm-WM': ((46,), (46,)),
    'L-Cblm-Cortex': ((8,), (8,)), 'R-Cblm-Cortex': ((47,), (47,)),
    'L-Thalamus': ((10,), (10,)), 'R-Thalamus': ((49,), (49,)),
    'L-Caudate': ((11,), (11,)), 'R-Caudate': ((50,), (50,)),
    'L-Putamen': ((12,), (12,)), 'R-Putamen': ((51,), (51,)),
    'L-Pallidum': ((13,), (13,)), 'R-Pallidum': ((52,), (52,)),
    '3rd-Vent': ((14,), (14,)), '4th-Vent': ((15,), (15,)), 'Brain-Stem': ((16,), (16,)),
    'L-Hippocampus': ((17,), (17,)), 'R-Hippocampus': ((53,), (53,)),
    'L-Amygdala': ((18,), (18,)), 'R-Amygdala': ((54,), (54,)),
    'L-Accumbens': ((26,), (26,)), 'R-Accumbens': ((58,), (58,)),
    'L-VentralDC': ((28,), (28,)), 'R-VentralDC': ((60,), (60,)),
    'CSF*': ((24,), (24,)), 'L-Vessel*': ((30,), (30,)), 'R-Vessel*': ((62,), (62,)),
}


def val_tseg_18(input_dir, output_dir=None, GPU=False, debug=False, files_filter=None,
                    tbx_model=None, seg_model=None, **kwargs):
    """TSEG on seg18: full pipeline (tBX + TSEG) on raw18/<name>.nii.gz, Dice per structure against
    label18/<name>.nii.gz after reorienting the prediction to RAS (the label grid). Returns the mean over subjects
    and over the 30 structures without '*' (tigerbx 0.3.0 default model: 0.8131 with full=True)."""
    names = list(SEG18_STRUCTURES)
    ffs = resolve_nifti_inputs([join(input_dir, 'raw18', '*.nii.gz')])
    run_model = {}
    if tbx_model: run_model['tbx'] = tbx_model
    if seg_model: run_model['tseg'] = seg_model
    argstr = ('g' if GPU else '') + 'mt'

    def compute_metrics(f, res):
        val = res['tseg']
        img = nib.load(val) if isinstance(val, str) else val
        pred = nib.Nifti1Image(np.asarray(img.dataobj).astype(np.int16), img.affine)
        pred = np.asarray(nib.as_closest_canonical(pred).dataobj).astype(int)
        gt = np.asarray(nib.load(f.replace('raw18', 'label18')).dataobj).astype(int)
        if pred.shape != gt.shape:
            raise ValueError(f'prediction {pred.shape} and label {gt.shape} grids differ for {basename(f)}')
        out = []
        for g, p in SEG18_STRUCTURES.values():
            a, b = np.isin(gt, g), np.isin(pred, p)
            n = a.sum() + b.sum()
            out.append(2 * np.logical_and(a, b).sum() / n if n else np.nan)
        return out

    f_list, dsc_list = _run_bx_batch(ffs, argstr, GPU, run_model or None, compute_metrics,
                                     debug=debug, files_filter=files_filter)
    _write_csv(output_dir, 'val_tseg_18.csv', names, f_list, dsc_list)
    if not dsc_list:
        return {'Filename': []}, float('nan')
    per_struct = np.nanmean(np.array(dsc_list, float), axis=0)
    for n, v in zip(names, per_struct):
        print(f'  {n:<15} : {v:.4f}')
    metric = float(np.nanmean([v for n, v in zip(names, per_struct) if '*' not in n]))
    print(f'mean Dice of the 30 structures: {metric:.4f}')
    data = {'Filename': f_list, **{n: [r[i] for r in dsc_list] for i, n in enumerate(names)}}
    return data, metric


def val_hlc_123(input_dir, output_dir=None, GPU=False,
                debug=False, files_filter=None,
                tbx_model=None, seg_model=None, **kwargs):
    column_names = ['Left-Thalamus',    'Right-Thalamus',
                    'Left-Caudate',     'Right-Caudate',
                    'Left-Putamen',     'Right-Putamen',
                    'Left-Pallidum',    'Right-Pallidum',
                    'Left-Hippocampus', 'Right-Hippocampus',
                    'Left-Amygdala',    'Right-Amygdala']
    ffs = resolve_nifti_inputs([join(input_dir, 'raw123', '*.nii.gz')])
    run_model = {}
    if tbx_model: run_model['tbx'] = tbx_model
    if seg_model: run_model['HLC'] = seg_model   # hlc171 uses uppercase 'HLC' key
    run_model = run_model or None

    def compute(f, _tmp):
        result    = tigerbx.hlc(f, _tmp, model=run_model, GPU=GPU, save='h')
        mask_pred = result['hlc'].get_fdata().astype(int)
        mask_gt   = nib.load(f.replace('raw123', 'label123')).get_fdata().astype(int)
        return get_dice12(mask_gt, mask_pred, 'tseg')

    f_list, dsc_list = _run_loop(ffs, compute, debug=debug, files_filter=files_filter)
    _write_csv(output_dir, 'val_hlc_123.csv', column_names, f_list, dsc_list)

    dsc_array       = np.array(dsc_list)
    mean_per_column = dict(zip(column_names, dsc_array.mean(axis=0).tolist()))
    _print_col_means(column_names, mean_per_column)
    data = {'Filename': f_list,
            **{col: dsc_array[:, i].tolist() for i, col in enumerate(column_names)}}
    return data, mean_per_column


def val_reg_60(input_dir, output_dir=None, GPU=False,
               debug=False, files_filter=None, template=None,
               tbx_model=None, seg_model=None, plan='AD', affine_type='numreg', **kwargs):
    """Registration Dice on raw60/label60: each label60 map is carried to the template by the subject's own
    registration (plan, default 'AD' = numreg affine + VFA) and scored against the template tseg (26 structures)."""
    from tigerbx.core.resample import reorder_img  # lazy import - local resample helper
    column_names = [
        'Left-Cerebral WM',       'Right-Cerebral WM',
        'Left-Cerebral Cortex',   'Right-Cerebral Cortex',
        'Left-Lateral Ventricle', 'Right-Lateral Ventricle',
        'Left-Cerebellum WM',     'Right-Cerebellum WM',
        'Left-Cerebellum Cortex', 'Right-Cerebellum Cortex',
        'Left-Thalamus',          'Right-Thalamus',
        'Left-Caudate',           'Right-Caudate',
        'Left-Putamen',           'Right-Putamen',
        'Left-Pallidum',          'Right-Pallidum',
        'Left-Hippocampus',       'Right-Hippocampus',
        'Left-Amygdala',          'Right-Amygdala',
        'Left-VentralDC',         'Right-VentralDC',
        'Brain Stem',             'CSF',
    ]
    ffs = resolve_nifti_inputs([join(input_dir, 'raw60', '*.nii.gz')])
    run_model = {}
    if tbx_model: run_model['tbx'] = tbx_model
    if seg_model: run_model['vfa' if 'D' in plan.upper() else 'reg'] = seg_model
    run_model = run_model or None

    import importlib
    from types import SimpleNamespace
    reg_mod = importlib.import_module('tigerbx.modules.reg')
    mask_gt = reorder_img(lib_reg.get_template_seg(template),
                          resample='nearest').get_fdata().astype(int)

    def compute(f, _tmp):
        args = SimpleNamespace(plan=plan, input=[f], output=_tmp, model=run_model, gpu=GPU, gz=False,
                               template=template, save_displacement=False, affine_type=affine_type, verbose=0)
        _, _, reg_ctx = reg_mod.run_case(args, f)
        moved = reg_mod.apply_warp(nib.load(f.replace('raw60', 'label60')), reg_ctx, interpolation='nearest')
        mask_pred = reorder_img(moved, resample='nearest').get_fdata().astype(int)
        return get_dice26(mask_gt, mask_pred)

    f_list, dsc_list = _run_loop(ffs, compute, debug=debug, files_filter=files_filter)
    _write_csv(output_dir, 'val_reg_60.csv', column_names, f_list, dsc_list)

    dsc_array       = np.array(dsc_list)
    mean_per_column = dict(zip(column_names, dsc_array.mean(axis=0).tolist()))
    _print_col_means(column_names, mean_per_column)
    data = {'Filename': f_list,
            **{col: dsc_array[:, i].tolist() for i, col in enumerate(column_names)}}
    return data, mean_per_column


# ── dataset registry ──────────────────────────────────────────────────────────
#
# 'relative_probe' : glob pattern relative to the candidate input_dir.
#                   Used to detect whether a directory is this dataset type.
# 'funcs'          : list of (display_name, callable) to run on this dataset.
#
# Directory names are NOT hard-coded here — _discover_datasets() scans
# val_dir and all its immediate sub-directories for each probe pattern.

DATASET_REGISTRY = [
    {
        'id':             'strip2',
        'relative_probe': join('*', 'image.nii.gz'),
        'funcs':          [('tbx_strip2', val_tbx_strip2, None)],
    },
    {
        'id':             'strip1',
        'relative_probe': join('*', '*T1w.nii.gz'),
        'funcs':          [('tbx_strip1', val_tbx_strip1, None)],
    },
    {
        'id':             'seg123',
        'relative_probe': join('raw123', '*.nii.gz'),
        'funcs': [
            ('tseg_123', lambda **kw: _val_seg_123('tseg', 't', **kw), 'tseg'),
            ('dgm_123',  lambda **kw: _val_seg_123('dgm',  'd', **kw), 'dgm'),
            ('hlc_123',  val_hlc_123,                                  'hlc'),
        ],
    },
    {
        'id':             'seg18',
        'relative_probe': join('raw18', '*.nii.gz'),
        'funcs':          [('tseg_18', val_tseg_18, 'tseg')],
    },
    {
        'id':             'reg60',
        'relative_probe': join('raw60', '*.nii.gz'),
        'funcs':          [('reg_60', val_reg_60, 'reg')],
    },
]



# ── auto-discovery helpers ────────────────────────────────────────────────────

def _discover_datasets(val_dir):
    """
    Scan *val_dir* and its immediate sub-directories for known dataset patterns.
    Returns ``{ds_id: abs_input_dir}`` for every dataset found.
    Directories are checked in alphabetical order; first match wins.
    """
    val_dir = os.path.abspath(val_dir)
    candidates = [val_dir] + sorted([
        join(val_dir, d) for d in os.listdir(val_dir)
        if os.path.isdir(join(val_dir, d))
    ])
    found = {}
    for ds in DATASET_REGISTRY:
        for candidate in candidates:
            if glob.glob(join(candidate, ds['relative_probe'])):
                found[ds['id']] = candidate
                break
    return found


def _build_lite_list(val_dir, discovered, n=_LITE_N, seed=42):
    """
    Randomly sample *n* files per dataset and save to ``lite_list.json``
    in *val_dir*.  Paths stored relative to *val_dir* for portability.
    Returns ``{ds_id: set(abs_path)}``.
    """
    val_dir = os.path.abspath(val_dir)
    rng  = random.Random(seed)
    lite = {}
    for ds in DATASET_REGISTRY:
        if ds['id'] not in discovered:
            continue
        input_dir = discovered[ds['id']]
        all_files = sorted(glob.glob(join(input_dir, ds['relative_probe'])))
        if not all_files:
            continue
        selected  = sorted(rng.sample(all_files, min(n, len(all_files))))
        lite[ds['id']] = [os.path.relpath(f, val_dir) for f in selected]

    lite_path = join(val_dir, 'lite_list.json')
    with open(lite_path, 'w') as fh:
        json.dump(lite, fh, indent=2)
    print(f'Lite list saved → {lite_path}')
    print(f'  Files per dataset: { {k: len(v) for k, v in lite.items()} }')

    return {k: set(os.path.abspath(join(val_dir, p)) for p in v)
            for k, v in lite.items()}


def _load_lite_list(val_dir):
    """Load ``lite_list.json`` and return ``{ds_id: set(abs_path)}``."""
    val_dir = os.path.abspath(val_dir)
    with open(join(val_dir, 'lite_list.json')) as fh:
        rel = json.load(fh)
    return {k: set(os.path.abspath(join(val_dir, p)) for p in v)
            for k, v in rel.items()}


def _mean_dice(metric):
    """Return a single float mean-Dice from either a float or a per-label dict."""
    if isinstance(metric, dict):
        return float(np.mean(list(metric.values())))
    return float(metric)


# ── internal runner ───────────────────────────────────────────────────────────

def _val_auto(val_dir=None, output_dir=None, model=None, GPU=False,
              full=False, template=None, task=None,
              tbx_model=None, seg_model=None):
    """Auto-discover datasets under val_dir and run all of them."""
    if val_dir is None:
        val_dir = os.getcwd()
    val_dir  = os.path.abspath(val_dir)
    mode_tag = 'full' if full else 'lite'

    model_dict = dict(model or {})
    if tbx_model: model_dict['tbx'] = tbx_model
    tbx_model = model_dict.get('tbx')

    discovered = _discover_datasets(val_dir)
    if not discovered:
        print(f'No recognised datasets found under: {val_dir}')
        print('Expected patterns: */image.nii.gz  |  */*T1w.nii.gz  |  raw123/  |  raw60/')
        return {}
    print(f'Datasets found: {list(discovered.keys())}')

    # lite mode: load or build file list
    lite = None
    if not full:
        lite_path = join(val_dir, 'lite_list.json')
        if os.path.exists(lite_path):
            lite = _load_lite_list(val_dir)
            # check for datasets discovered now but absent from the cached list
            missing = [ds['id'] for ds in DATASET_REGISTRY
                       if ds['id'] in discovered and ds['id'] not in lite]
            if missing:
                print(f'Lite list outdated (missing: {missing}), rebuilding …')
                lite = _build_lite_list(val_dir, discovered)
            else:
                print(f'Lite mode: loaded {lite_path}')
        else:
            print('Lite mode: lite_list.json not found, building …')
            lite = _build_lite_list(val_dir, discovered)

    summary, failed = {}, {}
    for ds in DATASET_REGISTRY:
        if ds['id'] not in discovered:
            continue
        input_dir    = discovered[ds['id']]
        files_filter = lite.get(ds['id']) if lite is not None else None

        for name, func, seg_key in ds['funcs']:
            if task is not None and task not in name:
                continue
            _seg_model = seg_model if seg_model is not None else (
                model_dict.get(seg_key) if seg_key else None)
            print(f'\n{"="*60}')
            print(f'  {name}  [{mode_tag}]')
            print(f'{"="*60}')
            _FAILED.clear()
            try:
                _, metric = func(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    GPU=GPU,
                    files_filter=files_filter,
                    template=template,
                    tbx_model=tbx_model,
                    seg_model=_seg_model,
                )
                summary[name] = metric
            except Exception as exc:
                print(f'  SKIPPED ({type(exc).__name__}: {exc})')
                summary[name] = None
            failed[name] = list(_FAILED)

    # final summary table
    if summary:
        sep = '=' * 60
        print(f'\n{sep}')
        print(f'  Validation Summary  [{mode_tag}]')
        print(sep)
        w = max(len(k) for k in summary)
        for name, metric in summary.items():
            n_fail = len(failed.get(name, []))
            tail = f'  ({n_fail} file(s) failed, excluded)' if n_fail else ''
            if metric is None:
                print(f'  {name:<{w}} : SKIPPED{tail}')
            else:
                print(f'  {name:<{w}} : mean Dice = {_mean_dice(metric):.4f}{tail}')
        print(sep)
        for name, items in failed.items():
            for f, err in items:
                print(f'  [{name}] FAILED {f}: {err}')

    return summary


# ── public entry point ────────────────────────────────────────────────────────

def val(val_dir=None, output_dir=None, model=None, GPU=False,
        full=False, template=None, task=None,
        tbx_model=None, seg_model=None):
    """
    Auto-discover datasets under *val_dir* and run all available validations.

    Parameters
    ----------
    val_dir    : str, optional
        Root directory containing dataset sub-folders.  Defaults to cwd.
        The function scans *val_dir* and all immediate sub-directories for
        recognisable dataset patterns — directory names are not fixed.
    output_dir : str, optional
        Where to write per-validation CSV files.  No CSV saved when None.
    task       : str, optional
        Run only tasks whose name contains this string.
        Valid values: ``'tbx'``, ``'tseg'``, ``'dgm'``,
        ``'hlc'``, ``'reg'``.  Default: run all discovered tasks.
    tbx_model  : str, optional
        Override tBX model filename for all tasks.
    seg_model  : str, optional
        Override the segmentation model for the selected task.
        Most useful when *task* is also specified.
    model      : dict, optional
        Low-level override dict (same format as ``run()``).
        ``tbx_model`` / ``seg_model`` take priority over this dict.
    GPU        : bool
        Use GPU for inference.
    full       : bool
        False (default) → **lite mode**: ≤ ``_LITE_N`` randomly sampled files
        per dataset, list built on first run and cached in ``lite_list.json``.
        True → run on the complete dataset.
    template   : str, optional
        Registration template path (forwarded to val_reg_60 only).

    Returns
    -------
    dict  {validation_name: metric}

    Examples
    --------
    >>> tigerbx.val('/data/val_home')                            # all tasks, lite
    >>> tigerbx.val('/data/val_home', full=True)                 # all tasks, full
    >>> tigerbx.val('/data/val_home', task='tseg',
    ...             seg_model='new_tseg.onnx', GPU=True)         # new TSEG only
    >>> tigerbx.val('/data/val_home', task='tbx',
    ...             tbx_model='new_tbx.onnx', GPU=True)          # new tBX only
    >>> tigerbx.val('/data/val_home', task='tseg',
    ...             tbx_model='new_tbx.onnx',
    ...             seg_model='new_tseg.onnx', GPU=True)         # both overridden
    """
    return _val_auto(val_dir=val_dir, output_dir=output_dir,
                     model=model, GPU=GPU, full=full, template=template,
                     task=task, tbx_model=tbx_model, seg_model=seg_model)


