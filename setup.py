from setuptools import setup, find_packages

setup(
     name='tigerbx',
     version='0.2.2',
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
     python_requires='>=3.8',
     install_requires=[
             'numpy>=1.21.6',
             'nilearn>=0.9.2',
             'optuna',
             'SimpleITK>=2.1.0',
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
