import re
from pathlib import Path
from setuptools import setup, find_packages


def _read_version():
    text = Path(__file__).parent.joinpath('tigerbx', '__init__.py').read_text(encoding='utf-8')
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", text, re.MULTILINE)
    if not match:
        raise RuntimeError('Unable to find __version__ in tigerbx/__init__.py')
    return match.group(1)


setup(
     name='tigerbx',
     version=_read_version(),
     license='CC BY-NC-SA 4.0',
     license_files=['LICENSE'],

     packages=find_packages(),
     package_data={
        'tigerbx': ['template/*.nii.gz'],  # include the MNI-space template
     },
     include_package_data=True,

     python_requires='>=3.10',
     install_requires=[
             'numpy>=1.21.6',
             'scipy',
             'nibabel',
             'platformdirs',
             'filelock',
             'tqdm',
             'threadpoolctl>=3.1',   # numreg limits BLAS to one thread (faster for its N x 3 products)
         ],
     extras_require={
         'cpu': ['onnxruntime>=1.18.0'],
         'cu12': ['onnxruntime-gpu>=1.18.0,<1.27'],   # 1.27+ is built for CUDA 13
         'dev': ['pytest', 'pandas'],
     },

)
