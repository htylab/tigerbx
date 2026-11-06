from tigerbx.modules.bx import infer as bx_run


def bx(input=None, output=None, *, save_mask=False, gpu=False, gz=True, model=None, resume=False,
       verbose=0):
    """Brain extraction (the executable with no option): tBX -> ``_tbx`` (the brain image, input grid), and
    ``_tbxmask`` with ``save_mask=True``. Same as ``tigerbx.run('b')`` / ``run('bm')``. model: {'tbx': ...} override.
    resume: skip inputs whose outputs are already there (continuing a batch; they are left out of the list).
    Returns the result dict (images) for a single input, otherwise a list of file dicts."""
    flags = 'b' + ('m' if save_mask else '') + ('g' if gpu else '') + ('z' if gz else '')
    return bx_run(flags, input, output, model=model, continue_=resume, verbose=verbose)
