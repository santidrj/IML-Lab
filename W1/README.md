# Work 1: Clustering excercise

Authors: Augusto Moran Calderon, Santiago del Rey Juárez, and Yazmina Zurita Martel

## Contents of the project

The project is divided into three packages: `processing`, `algorithms`, and `models`.

### Processing

This folder contains the Python scripts used to clean the different datasets used.

In order to clean all the datasets at once you can access the folder and run the `process.py` file as follows:

```bash
cd w1/processing
python process.py
```

In the same way, you can run each script individually.

These scripts will generate their corresponding cleaned datasets and save them as pickle files in
the `datasets/processed` folder.

### Algorithms

This folder contains the Python classes implementing the K-Means, K-Modes and Fuzzy C-Means algorithms.

### Models

In the `models` folder we find all the necessary scripts tu run the different algorithms for each dataset. Each script
is of the form `dataset_algorithm.py`.

In order to run an algorithm for all the datasets you need to access the `models` folder and execute one of the
following scripts:

```bash
cd w1/models
python run_kmeans.py
python run_kmodes.py
python run_optics.py
python run_fuzzy_c_means.py
```

Also, if you want to execute a particular algorithm for a particular dataset you can run any of the scripts in this
folder in the same way as in the above example.

It is worth noting that for the OPTICS and Fuzzy C-Means algorithms the execution time is considerably high due to the
Connect-4 dataset, which is very large.

All these scripts generate several output files such as plots and validation results that are stored in the `figures`
and `validation` folders respectively.
