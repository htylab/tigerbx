import importlib


def pipeline(name, input=None, output=None, **kwargs):
    """Run an opinionated high-level pipeline by name.

    Example:
        tigerbx.pipeline('vbm', input='T1w.nii.gz', output='out_dir')
    """
    if not name:
        raise ValueError('pipeline name is required')
    key = str(name).strip().lower()

    if key == 'vbm':
        mod = importlib.import_module('tigerbx.pipelines.vbm')
        fn = getattr(mod, 'vbm')
        return fn(input=input, output=output, **kwargs)

    raise ValueError(f'Unknown pipeline: {name!r}. Available: vbm')

