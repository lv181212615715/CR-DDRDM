# -*- coding: utf-8 -*-
from itertools import product
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
import warnings


def OPTICS_clustering_1(X):
    """
    改进的OPTICS聚类实现，包含以下改进：
    1. 添加了异常处理
    2. 优化了参数搜索
    3. 防止除以零错误
    4. 更稳定的簇提取
    """
    X = np.array(X)
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    best_score = -1
    best_params = {'min_samples': 5, 'xi': 0.05}  # 默认值
    best_clusters = [[]]  # 默认返回一个空簇
    best_centers = [0]  # 默认中心点

    # 参数搜索空间（简化以提高稳定性）
    min_samples_range = range(5, min(20, len(X)), 5)  # 限制最大min_samples
    xi_range = [0.01, 0.05, 0.1]

    # 抑制OPTICS的除以零警告
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        for min_samples, xi in product(min_samples_range, xi_range):
            try:
                # 确保min_samples不超过数据量
                min_samples = min(min_samples, len(X) - 1)

                optics = OPTICS(
                    min_samples=min_samples,
                    xi=xi,
                    metric='euclidean',
                    cluster_method='xi'  # 使用xi方法提取簇
                ).fit(X)

                labels = optics.labels_
                unique_labels = set(labels)

                # 至少需要2个簇才能计算轮廓系数
                if len(unique_labels) > 1 and -1 not in unique_labels:
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_params.update({
                            'min_samples': min_samples,
                            'xi': xi
                        })

                        # 组织簇结构
                        clusters = {}
                        for idx, label in enumerate(labels):
                            if label not in clusters:
                                clusters[label] = []
                            clusters[label].append(idx)

                        # 计算中心点
                        center_points = []
                        valid_clusters = []
                        for label in sorted(clusters.keys()):
                            if label != -1:  # 排除噪声点
                                points = clusters[label]
                                if len(points) > 0:
                                    geometric_center = np.mean(X[points], axis=0)
                                    distances = [np.linalg.norm(X[i] - geometric_center) for i in points]
                                    center_idx = points[np.argmin(distances)]
                                    center_points.append(center_idx)
                                    valid_clusters.append(points)

                        if valid_clusters:  # 确保至少有一个有效簇
                            best_clusters = valid_clusters
                            best_centers = center_points

            except Exception as e:
                continue  # 跳过失败的参数组合

    return best_clusters, len(best_clusters), best_centers, best_params