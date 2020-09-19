# a2e - an auto autoencoder for IIoT

## Development

### Setup virtualenv

```
pip install virtualenv;
python -m virtualenv env;
source env/bin/activate;
```

### Install dependencies

```
pip install -r requirements.txt;
pip install -e .;
```

### Documentation

```
pdoc --html --output-dir doc a2e;
open doc/a2e/index.html;
```
