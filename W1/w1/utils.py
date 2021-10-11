import pandas as pd
from pandas import DataFrame
from scipy.io import arff
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, normalize


def load_arff(path):
    data = arff.loadarff(path)
    return pd.DataFrame(data[0])


def convert_byte_string_to_string(dataframe: DataFrame):
    for col in dataframe:
        if isinstance(dataframe[col][0], bytes):
            print(col, "will be transformed from byte-string to string")
            dataframe[col] = dataframe[col].str.decode("utf8")  # or any other encoding


def categorical_to_numerical(dataframe: DataFrame):
    df_categorical = dataframe.select_dtypes(include='object')
    enc = OneHotEncoder()
    transformed_features = enc.fit_transform(df_categorical).toarray()
    df_transformed = pd.DataFrame(transformed_features, columns=enc.get_feature_names_out())
    df_numerical = dataframe.select_dtypes(include='number')
    return pd.concat([df_numerical, df_transformed], axis=1)


def normalize_data(dataframe: DataFrame, numerical_columns, scaler):
    df = dataframe.copy()
    scaled_data = scaler.fit_transform(dataframe[numerical_columns])

    if isinstance(scaler, RobustScaler) or isinstance(scaler, StandardScaler):
        normalized = normalize(scaled_data)
        df[numerical_columns] = pd.DataFrame(normalized, columns=numerical_columns)
        return df

    df[numerical_columns] = pd.DataFrame(scaled_data, columns=numerical_columns)
    return df
