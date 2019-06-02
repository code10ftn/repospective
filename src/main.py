import warnings

import pandas as pd
from sklearn import cluster, metrics, preprocessing
from sklearn.decomposition import PCA

import visualizer

ftr_full_name = 'full_name'
ftr_stars_count = 'stars_count'
ftr_forks_count = 'forks_count'
ftr_contributors_count = 'contributors_count'
ftr_commits_count = 'commits_count'
ftr_days_count = 'days_count'
ftr_is_org = 'is_org'
ftr_readme_path = 'readme_path'
ftr_topics = 'topics'
ftr_readme_topics = 'readme_topics'

remove_columns = [
    # ftr_readme_topics,
    # ftr_is_org,
    ftr_full_name,
    ftr_readme_path,
    ftr_topics]

numeric_columns = [
    ftr_stars_count,
    ftr_forks_count,
    ftr_contributors_count,
    ftr_commits_count,
    ftr_days_count]

CATEGORICAL_COLUMNS = [ftr_readme_topics]

random_state = 360


def encode_data(data):
    encoder = preprocessing.LabelEncoder()
    for category in CATEGORICAL_COLUMNS:
        data[category] = encoder.fit_transform(data[category])

    return data


def preprocess(data):
    data.drop(remove_columns, axis=1, inplace=True)
    data = encode_data(data)

    # Linux has 'infinite' contributors, some sources estimate ~10k
    data.replace('∞', 10000, inplace=True)

    return data


def normalize(data):
    scaler = preprocessing.MinMaxScaler()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # Fit on train and transform both
        train_scaled = scaler.fit_transform(data)

    data = pd.DataFrame(train_scaled, columns=data.columns)

    return data


def silhouette_score(estimator, X):
    labels = estimator.fit_predict(X)
    score = metrics.silhouette_score(X, labels, metric='euclidean')

    return score


def calculate_sse(data):
    sse = {}
    for k in range(2, 15):
        model = cluster.KMeans(n_clusters=k, random_state=random_state).fit(data)
        # Inertia: Sum of distances of samples to their closest cluster center
        sse[k] = model.inertia_
    visualizer.visualize_sse(sse)


def print_silhouette_score(data, labels, model_name):
    score = metrics.silhouette_score(data, labels, metric='euclidean')
    print(f'{model_name} Silhouette Coefficient: {score}')

    return score


def clusterize(data, n_clusters=2):
    calculate_sse(data)

    # Hierarchical
    model = cluster.AgglomerativeClustering(n_clusters=n_clusters)
    calculate_sse(data)

    predictions = model.fit_predict(data)
    print_silhouette_score(data, predictions, 'hierarchical')

    # K-means
    model = cluster.KMeans(n_clusters=n_clusters, random_state=random_state)

    predictions = model.fit_predict(data)
    print_silhouette_score(data, predictions, 'kmeans')

    return predictions


def reduce_dimensionality(data, n_components=3):
    pca = PCA(random_state=random_state, svd_solver='full', whiten=True, n_components=n_components)
    reduced = pca.fit_transform(data)

    return pd.DataFrame(reduced)


def get_statistics(data, clusters):
    data = data.replace('∞', 10000)
    data[ftr_contributors_count] = pd.to_numeric(data[ftr_contributors_count])

    # data = data[numeric_columns + [ftr_is_org]]
    data = data[numeric_columns]
    data = data.assign(clusters=pd.Series(clusters).values)
    stats = data.groupby(['clusters']).mean()

    print(stats.to_string())


def main():
    original_data = pd.read_csv(r'../data/data_with_readme_topics.csv')

    data = preprocess(original_data)

    data = normalize(data)

    # visualizer.save_graphs(data, numeric_columns)

    # Determine clusters
    n_clusters = 4
    print(data.head())
    test_predictions = clusterize(data, n_clusters)

    # Reduce dimensionality for visualizing clusters
    test_reduced = reduce_dimensionality(data)

    # Add column for cluster predictions
    test_reduced['cluster'] = pd.Series(test_predictions)

    visualizer.visualize(test_reduced, n_clusters)

    get_statistics(original_data, test_predictions)


if __name__ == '__main__':
    main()
