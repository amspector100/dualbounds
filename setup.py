import setuptools
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dualbounds",
    version=get_version('dualbounds/__init__.py'),
    author="Asher Spector",
    author_email="amspector100@gmail.com",
    description="Dual bounds for model-agnostic inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amspector100/dualbounds/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        "numpy>=1.17.4",
        "scipy>=1.12.0",
        "scikit-learn>=1.4.1",
        "pandas",
        "POT",
        "cvxpy",
        "tqdm",
    ],
)