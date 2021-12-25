import pandas as pd
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


def print_metrics(data, true_labels, pred_labels, file_path, isOPTICS=False):
    with open(file_path, 'a') as file:

        if isOPTICS:
            aux = data[pred_labels != -1]
            non_clusterized = (1 - len(aux)/len(data))*100
            data = aux
            true_labels = true_labels[pred_labels != -1]
            pred_labels = pred_labels[pred_labels != -1]

            file.write(f'\n{round(non_clusterized, 3)}% of samples are not in a cluster.\n')

        if len(set(pred_labels)) > 1:
            file.write('\nInternal validation')

            ch_score = metrics.calinski_harabasz_score(data, pred_labels)
            file.write(f'\nCalinski-Harabasz score: {ch_score}')

            db_score = metrics.davies_bouldin_score(data, pred_labels)
            file.write(f'\nDavies-Bouldin score: {db_score}')

            # s_score = metrics.silhouette_score(data, pred_labels)
            # print(f'Silhouette score (from -1 to 1): {s_score}')

        file.write('\n\nExternal validation')

        # hcv_score = metrics.homogeneity_completeness_v_measure(true_labels, pred_labels)
        # print(f'Homogeneity, completeness and V-measure (form 0 to 1): {hcv_score}')

        # rand_sc = metrics.rand_score(true_labels, pred_labels)
        # print(f'Rand index (form 0 to 1): {rand_sc}')

        # adj_rand_sc = metrics.adjusted_rand_score(true_labels, pred_labels)
        # print(f'Adjusted Rand index (from -1 to 1): {adj_rand_sc}')

        adj_mutual_info_sc = metrics.adjusted_mutual_info_score(true_labels, pred_labels)
        file.write(f'\nAdjusted Mutual Information score (from 0 to 1): {adj_mutual_info_sc}')

        fm_score = metrics.fowlkes_mallows_score(true_labels, pred_labels)
        file.write(f'\nFowlkes-Mallows score (from 0 to 1): {fm_score}')

        # contingency_mat = metrics.cluster.contingency_matrix(true_labels, pred_labels)
        # fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

        # plt.clf()

        # ax = fig.add_subplot(111)

        # ax.set_aspect(1)

        # res = sns.heatmap(contingency_mat, annot=True, fmt='d', cmap="YlGnBu", vmin=0.0, vmax=contingency_mat.max())

        # plt.title(f'Contingency Matrix for K={k}', fontsize=12)

        # plt.savefig(figure_path, bbox_inches='tight', dpi=100)

        # plt.show()
