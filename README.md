# product_classification

## Env setup

/!\ This setup was tested on Ubuntu 20.04.2 LTS

Install Anaconda and create a new py38 env.

```shell
conda create --name product_classification python==3.8
```

Activate the env:

```shell
conda activate product_classification
```

Install poetry:

```shell
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

Install the dependencies:

```shell
poetry install
```

Run Jupyterlab

```shell
jupyter-lab
```

## Notebooks

The data exploration and training/evaluation notebooks are in `./notebooks`