# Contents of the package

The project is divided into three packages: `algorithms`, `models`, and `processing`.

## Algorithms

This folder contains the Python classes implementing the K-Means, K-Modes and Fuzzy C-Means algorithms.

## Models

In the `models` folder we find all the necessary scripts tu run the different algorithms for each dataset. Each script
is of the from `dataset_algorithm.py`.

In order to run an algorithm for all the datasets you need to access the `models` folder and execute one of the
following scripts:

```bash
cd w1/models
python run_kmeans.py
python run_kmodes.py
python run_optics.py
python run_fuzzy_c_means.py
```
