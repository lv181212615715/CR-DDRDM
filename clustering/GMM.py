from itertools import product

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


def GMM_clustering_1(X):
    X = np.array(X).T
    best_score = -1
    best_params = {'n_components': None, 'covariance_type': None}
    best_clusters = None
    best_centers = None

    # 参数搜索空间
    n_components_range = range(2, 51)
    covariance_types = ['spherical', 'tied', 'diag', 'full']

    for n_components, cov_type in product(n_components_range, covariance_types):
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=cov_type,
                random_state=42
            ).fit(X)

            labels = gmm.predict(X)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_params.update({
                        'n_components': n_components,
                        'covariance_type': cov_type
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

    # 返回格式保持一致...
    return best_clusters, len(best_clusters), best_centers, best_params