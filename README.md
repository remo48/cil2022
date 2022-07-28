# CIL 2022

CIL 2022 course project

## Installation

These instructions assume that Conda is installed. The required python version for this project is Python 3.10. Either create a new conda environment with `make create_environment` or use an existing Python 3.10 environment.

**1. Create new conda environment** 

```sh
cd cil2022
make create_environment
```
**2. Install requirements**

```sh
make requirements
```

### Dataset
The dataset is available through Kaggle. Downloading the dataset from Kaggle requires valid credentials installed on the system. Please follow the instructions to add credentials for the [Kaggle API](https://github.com/Kaggle/kaggle-api).

```sh
make data
```

If the credentials are set correctly, the dataset is downloaded and extracted to the data folder.

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    
## Used Models

We provide checkpoints of the models we used in https://polybox.ethz.ch/index.php/s/WyJ6tidrkkB4mBm

## Authors

- [@Julian Neff](https://github.com/neffjulian)
- [@Colin Arnet](https://github.com/carnet98)
- [@Remo Kellenberger](https://github.com/remo48)
