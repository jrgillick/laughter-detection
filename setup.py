import re
from os import path

from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


def version():
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'laughter_detector/version.py')) as f:
        version_file = f.read()
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
        version = version_match.group(1)

    return version


setup(
    name='laughter-detector',
    version=version(),
    python_requires=">=3.7",
    install_requires=[
        'click', 'librosa', 'torch', 'tgt==1.4.4', 'pyloudnorm==0.1.0', 'praatio==3.8.0',
        'tensorboardX==1.9'
    ],
    description='Laughter detectior',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/charactr-platform/laughter-detection',
    author='Bartłomiej Olechno',
    author_email='barteko@charactr.com',
    keywords=['deep-learning', 'audio', 'classification'],
    licence='private',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={
        "config": ["laughter_detector/config.yaml"],
    },
    packages=find_packages(),
    zip_safe=False
)
