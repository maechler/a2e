# a2e - a library to auto tune autoencoders

This library is based on [Keras](https://keras.io/), [SMAC](https://github.com/automl/SMAC3/) and [HpBandSter](https://github.com/automl/HpBandSter/). 
It implements AutoML methods that are designed, but not limited to auto tune autoencoder architectures. Examples of its usage can be found in [experiments](./experiments).

## Features

- Supporting [SMAC](https://github.com/automl/SMAC3/) and [HpBandSter](https://github.com/automl/HpBandSter/) optimizers:
    - Bayesian Optimization using Random Forrest
    - Bayesian Optimization using Gaussian Process
    - Hyperband
    - Bayesian Optimization with Hyperband
    - Random 
- Providing various cost functions focused on autoencoders
- Providing Keras models for vanilla, convolutional and recurrent autoencoders
- Providing experiment logging including git repository state, models (including weights and activations plots), predictions, samples, metrics

![Features overview](./doc/readme.png)

## Installation as library

```
pip install git+https://github.com/maechler/a2e@master
pip install -r https://raw.githubusercontent.com/maechler/a2e/master/requirements.txt
```

## Dataset

All experiments are implemented using datasets from [a2e-data](https://github.com/maechler/a2e-data). 
The datasets are automatically downloaded and cached in `~/.a2e/`.

## System Requirements

- Python 3.8
- macOS, Ubuntu 18.04 or Windows
- [SWIG](http://www.swig.org/) (see [SMAC3](https://automl.github.io/SMAC3/master/installation.html))
- [graphviz](https://graphviz.org/)

### macOS installation

```
brew install swig
brew install graphviz
```

### Ubuntu 18.04 installation

Please note that the installation of SMAC will most probably fail on Ubuntu 20.04, use Ubuntu 18.04 instead!

```
sudo apt-get install swig
sudo apt-get install graphviz
```

If you can still not build some dependencies, please make sure you have the correct `python-dev` version installed:

```
sudo apt install python3.8-dev
```

### Windows installation

We recommend to use [WSL 2](https://docs.microsoft.com/en-us/windows/wsl/install-win10) and install Ubuntu 18.04 as you will probably not be able to run the SMAC optimizers on Windows. 
But if you really want to use Windows, you will have to at least install SWIG and graphviz:

- `SWIG`: 
    - Download `swigwin-4.*.*` from http://www.swig.org/download.html
    - Extract the folder and add it to the `PATH` environment variable
- `graphviz`: https://graphviz.org/download/

## Development

### Setup virtualenv

- Mac / Ubuntu
```
pip install virtualenv;
python -m virtualenv env;
source env/bin/activate;
```

- Windows (PowerShell)
```
pip install virtualenv;
python -m virtualenv env;
.\env\Scripts\activate.ps1
```

See https://docs.python.org/3/library/venv.html for further platforms and usages.

### Install dependencies

```
pip install -r requirements.txt;
pip install -e .;
```

### Testing

```
python -m unittest discover tests/*
```

## Documentation

```
pdoc --html --force --output-dir doc a2e;
open doc/a2e/index.html;
```

## Running an experiment

```
python experiments/feed_forward.py;
```

### Run on a specific CUDA GPU

```
python a2e/runner/cuda_runner.py experiments/feed_forward.py --gpu=1
```