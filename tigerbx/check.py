"""tigerbx.check(): the installation at a glance (versions, ONNX Runtime and the GPU, threads, the model cache)."""
import os
import platform
import sys
from os.path import join

from tigerbx import lib_tool
from tigerbx.core import models


def check(gpu=False, download=False, verbose=True):
    """Report what tigerbx runs with and return it as a dict.

    gpu: also build a session on the GPU and report whether it runs there (tigerbx falls back to the CPU with a
    warning). download: fetch the default models that are not cached yet. Every cached default model is checked
    against its release sha256.
    """
    import nibabel
    import numpy
    import scipy
    import tigerbx
    from tigerbx.core import onnx as core_onnx

    r = dict(tigerbx=tigerbx.__version__, python=sys.version.split()[0], platform=platform.platform(),
             numpy=numpy.__version__, scipy=scipy.__version__, nibabel=nibabel.__version__)
    try:
        import onnxruntime as ort
        r['onnxruntime'] = ort.__version__
        r['providers'] = ort.get_available_providers()
    except ImportError:
        r['onnxruntime'] = None
        r['providers'] = []
    r['cpus'] = core_onnx.cpu_count()
    threads = os.environ.get(core_onnx.ORT_INTRA_THREADS_ENV)
    r['ort_threads'] = int(threads) if threads else max(int(r['cpus'] * core_onnx.DEFAULT_ORT_INTRA_THREADS_RATIO), 1)
    r['model_dir'] = lib_tool.get_model_dir()

    defaults = sorted({v for k, v in vars(models).items() if k.isupper() and isinstance(v, str)})
    r['models'] = {}
    for name in defaults:
        f = join(r['model_dir'], name)
        if download and not os.path.exists(f):
            try:
                lib_tool.get_model(name)
            except Exception as exc:              # noqa: BLE001 - report it, check the others
                r['models'][name] = f'download failed: {exc}'
                continue
        if not os.path.exists(f):
            r['models'][name] = 'not downloaded'
        else:
            r['models'][name] = 'ok' if lib_tool.verify_model(f) else 'sha256 mismatch (downloaded again when used)'

    if gpu and r['onnxruntime']:
        import warnings
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                sess = core_onnx.create_session(lib_tool.get_model(models.INSPECT), True)
            r['gpu'] = 'CUDAExecutionProvider' in sess.get_providers()
            r['gpu_note'] = '; '.join(str(w.message) for w in caught)
        except Exception as exc:                  # noqa: BLE001
            r['gpu'], r['gpu_note'] = False, f'{type(exc).__name__}: {exc}'

    if verbose:
        print(f"tigerbx {r['tigerbx']}  (python {r['python']}, {r['platform']})")
        print(f"numpy {r['numpy']}, scipy {r['scipy']}, nibabel {r['nibabel']}")
        if r['onnxruntime']:
            print(f"onnxruntime {r['onnxruntime']}: {', '.join(r['providers'])}")
        else:
            print('onnxruntime: not installed (pip install "tigerbx[cpu]", [cu12] or [cu13])')
        print(f"CPUs {r['cpus']}, ONNX Runtime threads {r['ort_threads']} ({core_onnx.ORT_INTRA_THREADS_ENV})")
        if 'gpu' in r:
            print('GPU: ' + ('yes' if r['gpu'] else 'no, the CPU is used') + (f" ({r['gpu_note']})" if r['gpu_note'] else ''))
        print(f"models: {r['model_dir']}")
        for name, status in r['models'].items():
            print(f'  {name:34s} {status}')
    return r
