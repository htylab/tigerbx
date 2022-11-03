from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

classifiers = [
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3.6',
    'License :: OSI Approved :: MIT License'
]

setup(
     name='tigerseg',
     version='0.1.6',
     description='Package for subcortical brain segmentation',
     long_description_content_type='text/x-rst',
     url='https://github.com/htylab',
     author='JENNSHIUAN',
     author_email='danny092608@gmail.com',
     License='MIT',
     classifiers=classifiers,
     keywords='unet segmentation',
     package_dir={"": "src"},
     packages=find_packages(where="src"),
     entry_points={
        'console_scripts': [
            'tigerseg = tigerseg.console.__main__:main',
            'cine4d = tigerseg.console.__cine4d__:main',
            'aseg = tigerseg.console.__aseg__:main',
            'aseg2 = tigerseg.console.__aseg2__:main',
            'tigerbetpy = tigerseg.console.__tigerbet__:main',
            'vdm = tigerseg.console.__vdm__:main',
        ]
    },
     python_requires='>=3.6',
     install_requires=[
             'numpy>=1.16.0',
             'nilearn>=0.9.1',
             'onnxruntime>1.9.0',
             'simpleitk>=2.0.0',
             'scikit-image',
             'tqdm'

         ]
)
