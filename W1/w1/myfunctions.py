import pandas as pd
from scipy.io import arff


def load_arff(path):
    data = arff.loadarff(path)
    return pd.DataFrame(data[0])


def convert_byte_string_to_string(dataframe):
    for col in dataframe:
        if isinstance(dataframe[col][0], bytes):
            print(col, "will be transformed from byte-string to string")
            dataframe[col] = dataframe[col].str.decode("utf8")  # or any other encoding


def get_categorical_features(dataframe):
    features = []
    for column in dataframe.columns:
        if dataframe[column].dtype.kind is 'O':
            features.append(column)
    return features
