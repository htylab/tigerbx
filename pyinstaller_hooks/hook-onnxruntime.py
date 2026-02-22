import os
import onnxruntime
from PyInstaller.utils.hooks import collect_dynamic_libs

# Collect all shared libraries from the installed onnxruntime package
binaries = collect_dynamic_libs('onnxruntime')

# onnxruntime locates providers_shared via os.path.dirname(__file__),
# so it must sit at the bundle root ('.'), not inside the onnxruntime/ subdir.
# We auto-discover it from the installed package to avoid manual DLL maintenance.
_ort_dir = os.path.dirname(onnxruntime.__file__)
for _fname in os.listdir(_ort_dir):
    if 'providers_shared' in _fname:
        binaries.append((os.path.join(_ort_dir, _fname), '.'))
