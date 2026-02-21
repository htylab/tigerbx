from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

classifiers = [
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3',
    'License :: Free for non-commercial use',
    "Operating System :: OS Independent"
]

setup(
     name='tigerbx',

     version='0.2.1',
     description='Processing MRI images based on deep-learning',
     long_description=long_description,
     long_description_content_type='text/markdown',
     url='https://github.com/htylab/tigerbx',

     author='AI & Scientific Computing Lab, Taiwan Tech',
     author_email='',
     license='CC BY-NC 4.0',
     license_files=['LICENSE'],
     classifiers=classifiers,

     keywords='MRI brain segmentation',
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
             'numpy>=1.21.6,<2.0',
             'nilearn>=0.9.2',
             'optuna',
             'SimpleITK>=2.1.0',
             'antspyx',
             'platformdirs',
             'filelock',
         ],
     extras_require={
         'cpu': ['onnxruntime>=1.17.0,<1.21.0'],
         'cu12': ['onnxruntime-gpu>=1.17.0,<1.21.0'],
         'dev': ['pytest', 'pandas'],
     },

)
