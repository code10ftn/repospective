import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Axes3D = Axes3D


def save_graphs(data, columns):
    y_feature = columns[0]
    x_features = columns[1:]

    for x_feature in x_features:
        save_graph(data, x_feature, y_feature)


def save_graph(data, feature1, feature2):
    plt.clf()

    plt.xlabel(feature1)
    plt.ylabel(feature2)

    org_repos = data[data['is_org'] == 1]
    ind_repos = data[data['is_org'] == 0]

    plt.plot(org_repos[feature1], org_repos[feature2],
             'b+', label='organization')
    plt.plot(ind_repos[feature1], ind_repos[feature2],
             'r+', label='individual')

    plt.legend(loc='upper right')
    plt.savefig(fr'..\data\graph\{feature1}')


def visualize(data, n_clusters):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    for i in range(n_clusters):
        cluster = data[data['cluster'] == i]
        ax.scatter(cluster[0], cluster[1], cluster[2],
                   zdir='z', s=20, c=colors[i], depthshade=True)

    plt.show()


def visualize_sse(sse):
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel('Number of cluster')
    plt.ylabel('SSE')
    plt.show()
