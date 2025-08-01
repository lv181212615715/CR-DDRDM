import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from itertools import product
from sklearn.metrics import silhouette_score


def Spectral_clustering_1(X):
    X = np.array(X).T
    best_score = -1
    best_params = {'n_clusters': None, 'gamma': None, 'affinity': None}
    best_clusters = None
    best_centers = None

    # 参数搜索空间
    n_clusters_range = range(2, 51)
    gamma_range = np.logspace(-3, 1, 5)  # 10^-3到10^1
    affinity_options = ['rbf', 'nearest_neighbors']

    for n_clusters, gamma, affinity in product(n_clusters_range, gamma_range, affinity_options):
        try:
            if affinity == 'nearest_neighbors':
                affinity_matrix = kneighbors_graph(X, n_neighbors=10, mode='connectivity')
            else:
                affinity_matrix = rbf_kernel(X, gamma=gamma)

            sc = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed' if affinity == 'rbf' else affinity,
                random_state=42
            ).fit(affinity_matrix)

            labels = sc.labels_
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_params.update({
                        'n_clusters': n_clusters,
                        'gamma': gamma if affinity == 'rbf' else None,
                        'affinity': affinity
                    })
                    # 组织返回格式...
                    clusters = {label: [] for label in set(labels)}
                    for idx, label in enumerate(labels):
                        clusters[label].append(idx)

                    # 计算中心点
                    center_points = []
                    for label, points in clusters.items():
                        geometric_center = np.mean(points, axis=0)
                        center_point = points[np.argmin([np.linalg.norm(point - geometric_center) for point in points])]
                        center_points.append(center_point)

                    # best_clusters = [v for k, v in clusters.items()]
                    best_clusters = [clusters[label] for label in sorted(clusters.keys())]
                    best_centers = center_points
        except:
            continue

    # 返回格式与KMeans保持一致...
    return best_clusters, len(best_clusters), best_centers, best_params