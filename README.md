# Modeling Dynamic Range Compressor using S4

## Using this Repository

This Python project is managed by [Pipenv](https://pipenv.pypa.io/en/latest/) using Python 3.10.

Without configuring specified CUDA version on Nvidia platform, run `pipenv install`.
Otherwise, you might need to manually set up a Python virtual environment
with all dependencies listed in `Pipfile` installed.

All files in `src` folder are not supposed to be executed.
They are library files to be utilized by main scripts and Jupyter Notebook files.

All `*.py` files and `*.ipynb` files are main scripts or Jupyter Notebook files.
They are main scripts supposed to be executed directly.

All main scripts can receive command-line arguments. Run `*.py --help` to see the full list.
Main scripts can automatically handle dataset downloading without any manual setup.
