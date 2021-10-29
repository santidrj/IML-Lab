import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.io import arff
from sklearn import metrics
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


def validate_kmeans(data, true_labels, pred_labels, k):
    print('\nInternal validation')

    ch_score = metrics.calinski_harabasz_score(data, pred_labels)
    print(f'Calinski-Harabasz score: {ch_score}')

    db_score = metrics.davies_bouldin_score(data, pred_labels)
    print(f'Davies-Bouldin score: {db_score}')

    s_score = metrics.silhouette_score(data, pred_labels)
    print(f'Silhouette score (from -1 to 1): {s_score}')

    print('\nExternal validation')

    hcv_score = metrics.homogeneity_completeness_v_measure(true_labels, pred_labels)
    print(f'Homogeneity, completeness and V-measure: {hcv_score}')

    rand_sc = metrics.rand_score(true_labels, pred_labels)
    print(f'Rand index (form 0 to 1): {rand_sc}')

    adj_rand_sc = metrics.adjusted_rand_score(true_labels, pred_labels)
    print(f'Adjusted Rand index (from -1 to 1): {adj_rand_sc}')

    adj_mutual_info_sc = metrics.adjusted_mutual_info_score(true_labels, pred_labels)
    print(f'Adjusted Mutual Information score (from 0 to 1): {adj_mutual_info_sc}')

    fm_score = metrics.fowlkes_mallows_score(true_labels, pred_labels)
    print(f'Fowlkes-Mallows score (from 0 to 1): {fm_score}')

    contingency_mat = metrics.cluster.contingency_matrix(true_labels, pred_labels)
    fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    plt.clf()

    ax = fig.add_subplot(111)

    ax.set_aspect(1)

    res = sns.heatmap(contingency_mat, annot=True, fmt='.2f', cmap="YlGnBu", vmin=0.0, vmax=100.0)

    plt.title(f'Contingency Matrix for K={k}', fontsize=12)

    # plt.savefig("plot_contingency_table_seaborn_matplotlib_01.png", bbox_inches='tight', dpi=100)

    plt.show()
