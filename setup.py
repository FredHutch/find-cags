"""Setup module for ANN-based linkage clustering"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='ann_linkage_clustering',
    version='0.11',
    description='Linkage clustering via Approximate Nearest Neighbors',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/FredHutch/find-cags',
    author='Samuel Minot',
    author_email='sminot@fredhutch.org',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='science clustering',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        "pandas>=0.20.3",
        "numpy>=1.13.1",
        "scipy>=0.19.1",
        "awscli",
        "boto3>=1.4.7",
        "python-dateutil==2.6.0",
        "fastcluster>=1.1.24",
        "nmslib>=1.7.2",
        "scikit-learn>=0.19.2"
    ],
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/fredhutch/find-cags/issues',
        'Source': 'https://github.com/fredhutch/find-cags/',
    },
    entry_points={}
)