from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import product
import numpy as np


def KMeans_clustering_1(X):
    X = np.array(X).T

    # 针对1000个数据点的优化参数范围
    n_clusters_range = range(2, 51)  # 扩展聚类数范围
    init_methods = ['k-means++']  # 只使用更优的初始化方法
    max_iter_range = [300]  # 固定足够大的迭代次数
    n_init_range = [5, 10]  # 初始中心点尝试次数

    best_score = -1
    best_params = {'n_clusters': None, 'init': None, 'max_iter': None, 'n_init': None}
    best_clusters = None
    best_centers = None

    # 添加进度显示
    total_combinations = len(n_clusters_range) * len(init_methods) * len(max_iter_range) * len(n_init_range)
    current_comb = 0

    for n_clusters, init_method, max_iter, n_init in product(n_clusters_range, init_methods, max_iter_range,
                                                             n_init_range):
        current_comb += 1
        print(f"\rProgress: {current_comb}/{total_combinations} combinations", end="", flush=True)

        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init_method,
            max_iter=max_iter,
            n_init=n_init,
            random_state=42
        ).fit(X)

        labels = kmeans.labels_

        if len(set(labels)) > 1:
            try:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_params.update({
                        'n_clusters': n_clusters,
                        'init': init_method,
                        'max_iter': max_iter,
                        'n_init': n_init
                    })

                    clusters = {label: [] for label in set(labels)}
                    for idx, label in enumerate(labels):
                        clusters[label].append(idx)

                    center_points = []
                    for label in sorted(clusters.keys()):
                        geometric_center = kmeans.cluster_centers_[label]
                        closest_point_idx = clusters[label][np.argmin(
                            [np.linalg.norm(X[idx] - geometric_center) for idx in clusters[label]]
                        )]
                        center_points.append(closest_point_idx)

                    best_clusters = [clusters[label] for label in sorted(clusters.keys())]
                    best_centers = center_points
            except:
                continue

    print(f"\nBest silhouette score: {best_score:.4f}")
    return best_clusters, len(best_clusters), best_centers, best_params