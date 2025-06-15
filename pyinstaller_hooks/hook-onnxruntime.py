from PyInstaller.utils.hooks import collect_dynamic_libs

# Collect all shared libraries from onnxruntime
binaries = collect_dynamic_libs('onnxruntime')
