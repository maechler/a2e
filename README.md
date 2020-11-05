# a2e - an auto autoencoder for IIoT

## Development

### Setup virtualenv

- MAC
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
