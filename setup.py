from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

classifiers = [
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3.6',
    'Environment :: GPU :: NVIDIA CUDA :: 10.0',
    'License :: OSI Approved :: MIT License'
]

setup(
     name='tigerseg',
     version='0.0.5',
     description='Package for subcortical brain segmentation',
     long_description_content_type='text/x-rst',
     url='https://github.com/JENNSHIUAN',
     author='JENNSHIUAN',
     author_email='danny092608@gmail.com',
     License='MIT',
     classifiers=classifiers,
     keywords='subcortical brain segmentation',
     package_dir={"": "src"},
     packages=find_packages(where="src"),
     entry_points={
        'console_scripts': [
            'tigerseg = tigerseg.__main__:main'
        ]
    },
     python_requires='>=3.6',
     install_requires=[
             'numpy>=1.16.0',
             'nibabel>=2.5.1',
             'nilearn>=0.6.2',
             'SimpleITK>=2.0.0',
             'tables>=3.6.1',
             'Keras==2.3.1',
             'tensorflow-gpu==1.14.0',
             'h5py==2.9.0'
         ]
)
