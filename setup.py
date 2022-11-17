from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

classifiers = [
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3.7',
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
     packages=find_packages(),
     entry_points={
        'console_scripts': [
            'tigerhx = tigerseg.tigerhx:main',
            'tigerbx = tigerseg.tigerbx:main',
            'tigervdm = tigerseg.tigervdm:main',
        ]
    },
     python_requires='>=3.7',
     install_requires=[
             'numpy>=1.16.0',
             'nilearn>=0.9.1',
             'simpleitk>=2.0.0',
         ]
)
