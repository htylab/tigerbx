from os.path import basename, join, dirname
import numpy as np
import glob
import nibabel as nib
import tigerbx
from tigerbx import lib_tool
from tigerbx import lib_reg
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
            f_list.append(f)
            val = compute_fn(f, _tmp)
            results.append(val)
    return f_list, results


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
        batch = tigerbx.run(argstr, ffs, _tmp, model=model)
        if not isinstance(batch, list):
            batch = [batch]   # single-file path returns dict, not list
        for f, res in zip(ffs, batch):
            f_list.append(f)
            val = compute_metrics_fn(f, res)
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

def _apply_reg_to_seg(result, seg_file, template_nib, pad_width,
                      model_affine_transform, model_transform, reorder_img_fn):
    """Apply affine + dense-warp registration result to a segmentation file."""
    moving_seg_nib  = reorder_img_fn(nib.load(seg_file), resample='nearest')
    moving_seg_data = moving_seg_nib.get_fdata().astype(np.float32)
    moving_seg_data, _ = lib_reg.pad_to_shape(moving_seg_data, (256, 256, 256))
    moving_seg_data, _ = lib_reg.crop_image(moving_seg_data, target_shape=(256, 256, 256))
    moving_seg = np.expand_dims(np.expand_dims(moving_seg_data, axis=0), axis=1)

    init_flow  = result['init_flow'].get_fdata().astype(np.float32)
    affine_mat = np.expand_dims(result['Affine_matrix'].astype(np.float32), axis=0)
    out = lib_tool.predict(model_affine_transform,
                           [moving_seg, init_flow, affine_mat],
                           GPU=None, mode='affine_transform')
    moved_seg = lib_reg.remove_padding(np.squeeze(out[0]), pad_width)

    moved_seg = np.expand_dims(np.expand_dims(moved_seg, axis=0), axis=1)
    warp = np.expand_dims(result['dense_warp'].get_fdata().astype(np.float32), axis=0)
    out  = lib_tool.predict(model_transform, [moved_seg, warp], GPU=None, mode='reg')
    moved_nib = nib.Nifti1Image(np.squeeze(out[0]), template_nib.affine, template_nib.header)
    return reorder_img_fn(moved_nib, resample='nearest').get_fdata().astype(int)


# ── individual validation functions ──────────────────────────────────────────

def val_bet_synstrip(input_dir, output_dir=None, GPU=False,
                     debug=False, files_filter=None,
                     bet_model=None, seg_model=None, **kwargs):
    ffs = sorted(glob.glob(join(input_dir, '*', 'image.nii.gz')))
    run_model = {'bet': bet_model} if bet_model else None
    argstr = ('g' if GPU else '') + 'm'

    tt_list, cat_list = [], []

    def compute_metrics(f, res):
        tt_list.append(basename(dirname(f)).split('_')[1])
        cat_list.append('_'.join(basename(dirname(f)).split('_')[:2]))
        mask_pred = _nib_data(res, 'tbetmask')
        mask_gt   = nib.load(f.replace('image.nii.gz', 'mask.nii.gz')).get_fdata()
        dice = getdice((mask_pred > 0).flatten(), (mask_gt > 0).flatten())
        return dice, res['QC_raw']

    f_list, results = _run_bx_batch(ffs, argstr, GPU, run_model, compute_metrics,
                                    debug=debug, files_filter=files_filter)
    dsc_list   = [r[0] for r in results]
    qcraw_list = [r[1] for r in results]

    _write_csv(output_dir, 'val_bet_synstrip.csv',
               ['type', 'category', 'DICE', 'QC_raw'],
               f_list,
               [[t, c, d, q] for t, c, d, q in zip(tt_list, cat_list, dsc_list, qcraw_list)])

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
            'category': cat_list, 'DICE': dsc_list, 'QC_raw': qcraw_list}
    return data, metric


def val_bet_NFBS(input_dir, output_dir=None, GPU=False,
                 debug=False, files_filter=None,
                 bet_model=None, seg_model=None, **kwargs):
    ffs = sorted(glob.glob(join(input_dir, '*', '*T1w.nii.gz')))
    run_model = {'bet': bet_model} if bet_model else None
    argstr = ('g' if GPU else '') + 'm'

    def compute_metrics(f, res):
        mask_pred = _nib_data(res, 'tbetmask')
        mask_gt   = nib.load(f.replace('T1w.nii.gz', 'T1w_brainmask.nii.gz')).get_fdata()
        dice = getdice((mask_pred > 0).flatten(), (mask_gt > 0).flatten())
        return dice, res['QC_raw']

    f_list, results = _run_bx_batch(ffs, argstr, GPU, run_model, compute_metrics,
                                    debug=debug, files_filter=files_filter)
    dsc_list   = [r[0] for r in results]
    qcraw_list = [r[1] for r in results]

    _write_csv(output_dir, 'val_bet_NFBS.csv', ['DICE', 'QC_raw'], f_list,
               [[d, q] for d, q in zip(dsc_list, qcraw_list)])

    metric = float(np.mean(dsc_list))
    print(f'mean Dice of all data: {metric:.4f}')
    return {'Filename': f_list, 'DICE': dsc_list, 'QC_raw': qcraw_list}, metric


def _val_seg_123(model_str, run_option, input_dir, output_dir=None,
                 GPU=False, debug=False, files_filter=None,
                 bet_model=None, seg_model=None, **kwargs):
    column_names = ['Left-Thalamus',    'Right-Thalamus',
                    'Left-Caudate',     'Right-Caudate',
                    'Left-Putamen',     'Right-Putamen',
                    'Left-Pallidum',    'Right-Pallidum',
                    'Left-Hippocampus', 'Right-Hippocampus',
                    'Left-Amygdala',    'Right-Amygdala']
    ffs = sorted(glob.glob(join(input_dir, 'raw123', '*.nii.gz')))
    run_model = {}
    if bet_model: run_model['bet'] = bet_model
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


def val_hlc_123(input_dir, output_dir=None, GPU=False,
                debug=False, files_filter=None,
                bet_model=None, seg_model=None, **kwargs):
    column_names = ['Left-Thalamus',    'Right-Thalamus',
                    'Left-Caudate',     'Right-Caudate',
                    'Left-Putamen',     'Right-Putamen',
                    'Left-Pallidum',    'Right-Pallidum',
                    'Left-Hippocampus', 'Right-Hippocampus',
                    'Left-Amygdala',    'Right-Amygdala']
    ffs = sorted(glob.glob(join(input_dir, 'raw123', '*.nii.gz')))
    run_model = {}
    if bet_model: run_model['bet'] = bet_model
    if seg_model: run_model['HLC'] = seg_model   # hlc171 uses uppercase 'HLC' key
    run_model = run_model or None

    def compute(f, _tmp):
        result    = tigerbx.hlc(f, _tmp, model=run_model, GPU=GPU, save='h')
        mask_pred = result['hlc'].get_fdata().astype(int)
        mask_gt   = nib.load(f.replace('raw123', 'label123')).get_fdata().astype(int)
        return get_dice12(mask_gt, mask_pred, 'aseg')

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
               bet_model=None, seg_model=None, **kwargs):
    from nilearn.image import reorder_img     # lazy import — nilearn is heavy
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
    ffs = sorted(glob.glob(join(input_dir, 'raw60', '*.nii.gz')))
    run_model = {}
    if bet_model: run_model['bet'] = bet_model
    if seg_model: run_model['reg'] = seg_model
    run_model = run_model or None

    # load models and template once, outside the per-file loop
    model_transform        = lib_tool.get_model('mprage_transform_v002_near.onnx')
    model_affine_transform = lib_tool.get_model('mprage_affinetransform_v002_near.onnx')
    template_nib = reorder_img(lib_reg.get_template(template), resample='continuous')
    _, pad_width = lib_reg.pad_to_shape(template_nib.get_fdata(), (256, 256, 256))
    mask_gt      = reorder_img(lib_reg.get_template_seg(template),
                               resample='nearest').get_fdata().astype(int)

    def compute(f, _tmp):
        result = tigerbx.reg(('g' if GPU else '') + 'r', f, _tmp,
                             model=run_model, template=template)
        mask_pred = _apply_reg_to_seg(
            result, f.replace('raw60', 'label60'),
            template_nib, pad_width,
            model_affine_transform, model_transform, reorder_img)
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
        'id':             'synstrip',
        'relative_probe': join('*', 'image.nii.gz'),
        'funcs':          [('bet_synstrip', val_bet_synstrip, None)],
    },
    {
        'id':             'NFBS',
        'relative_probe': join('*', '*T1w.nii.gz'),
        'funcs':          [('bet_NFBS', val_bet_NFBS, None)],
    },
    {
        'id':             'seg123',
        'relative_probe': join('raw123', '*.nii.gz'),
        'funcs': [
            ('aseg_123', lambda **kw: _val_seg_123('aseg', 'a', **kw), 'aseg'),
            ('dgm_123',  lambda **kw: _val_seg_123('dgm',  'd', **kw), 'dgm'),
            ('syn_123',  lambda **kw: _val_seg_123('syn',  'S', **kw), 'syn'),
            ('hlc_123',  val_hlc_123,                                  'hlc'),
        ],
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
              bet_model=None, seg_model=None):
    """Auto-discover datasets under val_dir and run all of them."""
    if val_dir is None:
        val_dir = os.getcwd()
    val_dir  = os.path.abspath(val_dir)
    mode_tag = 'full' if full else 'lite'

    model_dict = dict(model or {})
    if bet_model: model_dict['bet'] = bet_model
    bet_model = model_dict.get('bet')

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

    summary = {}
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
            try:
                _, metric = func(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    GPU=GPU,
                    files_filter=files_filter,
                    template=template,
                    bet_model=bet_model,
                    seg_model=_seg_model,
                )
                summary[name] = metric
            except Exception as exc:
                print(f'  SKIPPED ({exc})')
                summary[name] = None

    # final summary table
    if summary:
        sep = '=' * 60
        print(f'\n{sep}')
        print(f'  Validation Summary  [{mode_tag}]')
        print(sep)
        w = max(len(k) for k in summary)
        for name, metric in summary.items():
            if metric is None:
                print(f'  {name:<{w}} : SKIPPED')
            else:
                print(f'  {name:<{w}} : mean Dice = {_mean_dice(metric):.4f}')
        print(sep)

    return summary


# ── public entry point ────────────────────────────────────────────────────────

def val(val_dir=None, output_dir=None, model=None, GPU=False,
        full=False, template=None, task=None,
        bet_model=None, seg_model=None):
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
        Valid values: ``'bet'``, ``'aseg'``, ``'dgm'``, ``'syn'``,
        ``'hlc'``, ``'reg'``.  Default: run all discovered tasks.
    bet_model  : str, optional
        Override BET model filename for all tasks.
    seg_model  : str, optional
        Override the segmentation model for the selected task.
        Most useful when *task* is also specified.
    model      : dict, optional
        Low-level override dict (same format as ``run()``).
        ``bet_model`` / ``seg_model`` take priority over this dict.
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
    >>> tigerbx.val('/data/val_home', task='aseg',
    ...             seg_model='new_aseg.onnx', GPU=True)         # new ASEG only
    >>> tigerbx.val('/data/val_home', task='bet',
    ...             bet_model='new_bet.onnx', GPU=True)          # new BET only
    >>> tigerbx.val('/data/val_home', task='aseg',
    ...             bet_model='new_bet.onnx',
    ...             seg_model='new_aseg.onnx', GPU=True)         # both overridden
    """
    return _val_auto(val_dir=val_dir, output_dir=output_dir,
                     model=model, GPU=GPU, full=full, template=template,
                     task=task, bet_model=bet_model, seg_model=seg_model)


# ── QC calibration ────────────────────────────────────────────────────────────

def qc_stat(csv_paths, dice_threshold=0.9):
    """
    Analyse the relationship between QC_raw and Dice from validation CSV files
    and suggest a calibrated QC_raw threshold.

    Reads any CSV(s) produced by ``val_bet_NFBS`` or ``val_bet_synstrip`` that
    contain ``DICE`` and ``QC_raw`` columns, then prints:

    * Distribution statistics for QC_raw and DICE.
    * Pearson correlation between QC_raw and DICE.
    * A suggested ``qc_score`` warning threshold such that ≥ 90 % of cases
      with Dice < *dice_threshold* are flagged, with the lowest possible
      false-alarm rate on good cases.

    Parameters
    ----------
    csv_paths : str or list of str
        Path(s) to validation CSVs.  Glob patterns are accepted
        (e.g. ``'val_out/*.csv'``).
    dice_threshold : float
        Dice below which a case is considered failed.  Default 0.9.

    Returns
    -------
    dict with keys ``dice``, ``qc_raw``, ``suggested_threshold``, ``pearson_r``.

    Examples
    --------
    >>> tigerbx.qc_stat('val_out/val_bet_NFBS.csv')
    >>> tigerbx.qc_stat(['val_out/val_bet_NFBS.csv',
    ...                  'val_out/val_bet_synstrip.csv'])
    """
    import glob as _glob

    if isinstance(csv_paths, str):
        expanded = _glob.glob(csv_paths)
        csv_paths = expanded if expanded else [csv_paths]

    dice_all, qcraw_all = [], []
    for path in csv_paths:
        with open(path, newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if 'DICE' in row and 'QC_raw' in row:
                    dice_all.append(float(row['DICE']))
                    qcraw_all.append(float(row['QC_raw']))

    if not dice_all:
        print('No DICE / QC_raw columns found in the provided CSVs.')
        return {}

    dice  = np.array(dice_all)
    qcraw = np.array(qcraw_all)
    n     = len(dice)
    n_fail = int(np.sum(dice < dice_threshold))
    sep = '─' * 62

    # ── descriptive stats ─────────────────────────────────────────────────────
    print(f'\n{sep}')
    print(f'QC calibration report   (n={n},  fail = Dice < {dice_threshold})')
    print(sep)
    print(f'  QC_raw  mean={np.mean(qcraw):.4f}  std={np.std(qcraw):.4f}  '
          f'min={np.min(qcraw):.4f}  '
          f'p1={np.percentile(qcraw, 1):.4f}  p5={np.percentile(qcraw, 5):.4f}')
    print(f'  DICE    mean={np.mean(dice):.4f}  std={np.std(dice):.4f}  '
          f'min={np.min(dice):.4f}')
    print(f'  Failed (Dice < {dice_threshold}): {n_fail} / {n}  '
          f'({100 * n_fail / n:.1f} %)')

    pearson_r = None
    if n > 2:
        pearson_r = float(np.corrcoef(qcraw, dice)[0, 1])
        print(f'  Pearson r(QC_raw, DICE) = {pearson_r:.4f}')

    # ── threshold search ──────────────────────────────────────────────────────
    # Sweep candidate thresholds; find t with sensitivity ≥ 90 % and minimum
    # false-alarm rate (fraction of good cases incorrectly flagged).
    best_t, best_far = None, 1.0
    fail_mask = dice < dice_threshold
    if n_fail > 0:
        for t in np.sort(np.unique(qcraw)):
            flagged = qcraw < t
            sens    = float(np.sum(flagged & fail_mask)) / n_fail
            if sens >= 0.90:
                far = float(np.sum(flagged & ~fail_mask)) / max(1, n - n_fail)
                if far < best_far:
                    best_far, best_t = far, float(t)

    # ── guidance ──────────────────────────────────────────────────────────────
    print(f'\n{sep}')
    if best_t is not None:
        suggested_score = int(np.clip(best_t * 100, 0, 100))
        print(f'  Suggested threshold : QC_raw = {best_t:.4f}  '
              f'→  qc_score = {suggested_score}')
        print(f'    → captures ≥ 90 % of failed cases  '
              f'(false-alarm rate ≈ {100 * best_far:.1f} %)')
        print(f'')
        print(f'  Recommendation: change the warning threshold in bx.py run_args()')
        print(f'    from:  if qc_score < 50:')
        print(f'    to:    if qc_score < {suggested_score}:')
    else:
        print('  Could not determine a threshold.')
        print('  Possible reasons:')
        print('    • Too few failed cases — model may already be excellent on this data.')
        print('    • QC_raw does not vary enough — try including harder / lower-quality scans.')
    print(f'{sep}\n')

    return {
        'dice':                dice_all,
        'qc_raw':              qcraw_all,
        'suggested_threshold': best_t,
        'pearson_r':           pearson_r,
    }
