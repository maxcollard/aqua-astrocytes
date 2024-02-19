from setuptools import setup

setup(
    name = 'aqua-astrocytes',
    packages = [
        'aqua',
    ],

    # TODO Pull from source
    version = '0.1',

    description = 'A Python interface for astrocyte quantification and analysis',
    # TODO Add long_description
    # long_description = '...',
    author = 'Max Collard',
    author_email = 'maxwell.collard@ucsf.edu',
    url = 'https://github.com/maxcollard/aqua-py',
    license = 'MIT',
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Programming Language :: Python :: 3.8',
    ],

    # TODO Work through actual minimum required versions
    install_requires = [
        'h5py==3.9.0',
        'numpy==1.24.3',
        'pandas==2.0.3',
        'pyyaml==6.0.1',
        'scikit-learn==1.3.0',
        'scipy==1.10.1',
        'statsmodels==0.14.0',
        'tqdm==4.65.0',
    ],
)

#