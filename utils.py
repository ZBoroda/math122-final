import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)


def split_data(input_file_name,
               output_file_name_train='train_rating.csv',
               output_file_name_test='test_rating.csv'):
    data = pd.read_csv(input_file_name)
    train = data.sample(frac=0.9, random_state=25)
    test = data.drop(train.index)
    train.to_csv(output_file_name_train, index=False)
    test.to_csv(output_file_name_test, index=False)
    return train, test


def PCA(dataframe, index_column,
        spectrum_fraction_threshold=0.9, n_dimensions_to_keep=0, graph_spectrum_fractions=False):
    X = dataframe.to_numpy()
    X_meanzero = (X - np.mean(X, axis=0))  # subtract the mean
    X = (X_meanzero / np.std(X, axis=0))  # divide by the standard deviation
    U, S, Vt = np.linalg.svd(X, full_matrices=True)
    # Compute the fraction of the total spectrum in the first n singular values
    spectrum_fractions = []
    for n in range(S.shape[0]):
        spectrum_fractions.append(sum(S[0:n] ** 2) / sum(S ** 2))
    if graph_spectrum_fractions:
        plt.scatter(np.array(range(len(spectrum_fractions))),
                    np.array(spectrum_fractions), s=10, c='b', marker='o')
    if n_dimensions_to_keep == 0:
        fractions_greater_than_cutoff = np.asarray(spectrum_fractions)
        fractions_greater_than_cutoff[fractions_greater_than_cutoff < spectrum_fraction_threshold] = 1
        n_dimensions = np.argmin(fractions_greater_than_cutoff)
    else:
        n_dimensions = n_dimensions_to_keep
    # Now let us use this top n_dimensions singular vectors to make a matrix
    reduced_data = pd.DataFrame(data=U[:, 0:n_dimensions], index=index_column)
    return reduced_data


def KMeansPCA(dataframe, index_column, index_column_name,
              num_clusters=3, spectrum_fraction_threshold=0.9, plot_clusters=False):
    X = dataframe.to_numpy()
    X_meanzero = (X - np.mean(X, axis=0))  # subtract the mean
    X = (X_meanzero / np.std(X, axis=0))  # divide by the standard deviation
    U, S, Vt = np.linalg.svd(X, full_matrices=True)
    # Reduce the dimensionality of the data to 2 dimensions to visualize it
    user_data_reduced = pd.DataFrame(data=U[:, 0:2], index=index_column)
    # Use k-means to separate the data into 3 clusters
    km = KMeans(n_clusters=num_clusters)
    km.fit(X)
    prediction = km.predict(X)
    clusters_df = pd.DataFrame(data=prediction, index=index_column)
    if plot_clusters:
        colors = {0: 'red', 1: 'green', 2: 'blue'}
        for color in range(3, num_clusters):
            colors[color] = 'yellow'
        user_data_reduced.plot.scatter(x=0, y=1, c=clusters_df[0].map(colors))
        plt.title("Clustered data reduced to 2D")
        plt.xlabel("Principal component 1")
        plt.ylabel("Principal component 2")
        plt.show()
    clusters_df = clusters_df.rename(columns={0: 'Cluster'})
    # separate the user_history df into clusters
    dataframe_clustered = pd.concat([dataframe, clusters_df], axis=1)
    clusters = []
    for cluster in range(num_clusters):
        clusters.append(dataframe_clustered[dataframe_clustered['Cluster'] == cluster])
    num_dimensions_for_each_cluster = []
    # find the optimal number of dimensions (min that keeps at least 90% of the variation) for each cluster separately
    for cluster in clusters:
        X = cluster.loc[:, cluster.columns != 'Cluster'].to_numpy()
        X_meanzero = (X - np.mean(X, axis=0))  # subtract the mean
        X = (X_meanzero / np.std(X, axis=0))  # divide by the standard deviation
        U, S, Vt = np.linalg.svd(X, full_matrices=True)
        fractions = np.zeros((X.shape[1]))
        fractions_greater_than_cutoff = np.zeros((X.shape[1]))
        for num_components in range(1, Vt.shape[0] + 1):
            # Compute the fraction of the total spectrum in the first n singular values
            fractions[num_components - 1] = sum(S[0:num_components] ** 2) / sum(S ** 2)
            fractions_greater_than_cutoff[num_components - 1] = sum(S[0:num_components] ** 2) / sum(S ** 2)
        fractions_greater_than_cutoff[fractions_greater_than_cutoff < spectrum_fraction_threshold] = 1
        num_dimensions_for_each_cluster.append(np.argmin(fractions_greater_than_cutoff))
    # do PCA on each cluster with its optimal number of dimensions
    reduced_cluster_data = []
    for i in range(len(clusters)):
        X = clusters[i].loc[:, clusters[i].columns != 'Cluster'].to_numpy()
        X_meanzero = (X - np.mean(X, axis=0))  # subtract the mean
        X = (X_meanzero / np.std(X, axis=0))  # divide by the standard deviation
        U, S, Vt = np.linalg.svd(X, full_matrices=True)
        reduced_cluster_data.append(pd.DataFrame(data=U[:, 0:num_dimensions_for_each_cluster[i]],
                                                 index=clusters[i].reset_index(inplace=False)[index_column_name]))
    return reduced_cluster_data, dataframe_clustered
