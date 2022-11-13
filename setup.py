from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

classifiers = [
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3.6',
    'License :: OSI Approved :: MIT License',
    "Operating System :: OS Independent"
]

setup(
     name='tigerseg',
     version='0.1.8',
     description='Processing MRI images based on deep-learning',
     long_description_content_type='text/markdown',
     url='https://github.com/htylab/tigerseg',
     author='Biomedical Imaging Lab, Taiwan Tech',
     author_email='',
     License='MIT',
     classifiers=classifiers,
     keywords='MRI segmentation',
     package_dir={"": "src"},
     packages=find_packages(where="src"),
     entry_points={
        'console_scripts': [
            'cine4d = tigerseg.console.__cine4d__:main',
            'tigerbx = tigerseg.console.tigerbx.tigerbx:main',
            'vdm = tigerseg.console.__vdm__:main',
        ]
    },
     python_requires='>=3.6',
     install_requires=[
             'numpy>=1.16.0',
             'nilearn>=0.9.1',
             'simpleitk>=2.0.0',
         ]
)
