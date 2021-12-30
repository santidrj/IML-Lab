# Work 3: Instance-Based Learning Algorithms

Authors: Augusto Moran Calderon, Santiago del Rey Juárez, and Yazmina Zurita Martel

## Contents of the project

The project is divided into two packages: `algorithms`, and `models`. And multiple scripts.

In addition, it contains all the available datasets in the `datasets` folder and the obtained results in the `results` folder.

### Algorithms

This folder contains the `IBL` class that implements all the IBL algorithms described in the report and  the auxiliary files required for its implementation.

### Models

In the `models` folder we find the `IBLEval` class that is used to read each dataset fold and execute the different IBL implementations with it.

## Running the algorithms
There are four available Python scripts to run the different implemented algorithms in the `w3` folder.
Please read the comments on the scripts to further configure their executions.

### IB1, IB2 and IB3
To run the IB1, IB2 and IB3 algorithms you can execute the following commands:
```bash
cd w3/
python run_ibl.py
```

### K-IBL
To run the multiple K-IBL configurations you can execute the following commands:
```bash
cd w3/
python run_kibl.py
```

### Selection K-IBL
To run the Selection K-IBL you can execute the following commands:
```bash
cd w3/
python run_select_kibl.py
```

### Statistical analysis
To perform the statistical analysis on a particular dataset you can run the following commands:
```bash
cd w3/
python statistical_analysis.py
```
Note that this script requires that the results of the K-IBL and Selection K-IBL are in their corresponding files in the `results` folder. (They are already provided.)


**Important!!** The execution of these scripts may take several hours, even days depending on the dataset. To avoid having to execute all of them we have saved the results of all the executions in the `results` folder.
