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
     license='CC BY-NC 4.0',
     license_files=['LICENSE'],

     packages=find_packages(),
     package_data={
        'tigerbx': ['template/*.nii.gz'],  # include the MNI152 template
     },
     include_package_data=True,

     entry_points={
        'console_scripts': [
            'tiger = tigerbx_cli.tiger:main',
        ]
    },
     python_requires='>=3.10',
     install_requires=[
             'numpy>=1.21.6',
             'scipy',
             'nibabel',
             'optuna',
             'antspyx',
             'platformdirs',
             'filelock',
             'tqdm',
         ],
     extras_require={
         'cpu': ['onnxruntime>=1.18.0'],
         'cu12': ['onnxruntime-gpu>=1.18.0'],
         'dev': ['pytest', 'pandas'],
     },

)
