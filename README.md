# product_classification

## Env setup

/!\ This setup was tested on Ubuntu 20.04.2 LTS

Install Anaconda and create a new py38 env.

```commandline
conda create --name product_classification python==3.8
```

Activate the env:

```commandline
conda activate product_classification
```

Install poetry:

```commandline
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

Install the dependencies:

```commandline
poetry install
```