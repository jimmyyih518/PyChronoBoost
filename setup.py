from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open('requirements.txt') as f:
    required_pkgs = f.read().splitlines()

setup(
    name='PyChronoBoost',
    version='0.1.0',
    author='Jimmy Zhang',
    author_email='jimmyyih@ualberta.ca',
    description='A Python package for automated time series feature generation and selection',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jimmyyih518/PyChronoBoost',
    packages=find_packages(),
    install_requires=required_pkgs,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.7',
    keywords='time series, feature generation, feature selection',
)
