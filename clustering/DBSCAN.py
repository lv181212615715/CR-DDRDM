import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from collections import defaultdict, Counter
from itertools import product


def DBSCAN_clustering_1(X):
    X = np.array(X).T
    eps_range = np.arange(0.001, 1.001, 0.05).tolist()  # 正确的列表格式
    min_samples_range = np.arange(5, 21, 5).tolist()  # 正确的列表格式
    best_score = -1
    best_params = {'eps': None, 'min_samples': None}
    best_clusters = None
    best_centers = None

    # 遍历所有eps和min_samples的组合
    for eps, min_samples in product(eps_range, min_samples_range):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = dbscan.labels_

        # 重新标记噪声点（将噪声点的标签设置为最近的簇的标签）
        noise_label = -1
        new_labels = np.copy(labels)
        if noise_label in set(labels):
            for i, label in enumerate(labels):
                if label == noise_label:
                    # 计算当前点到所有非噪声点的距离
                    distances = [(np.linalg.norm(X[i] - X[j]), j, lbl) for j, lbl in enumerate(labels) if
                                 lbl != noise_label]
                    # 找到最近的非噪声点及其标签
                    if distances:
                        _, closest_idx, closest_label = min(distances, key=lambda x: x[0])
                        new_labels[i] = closest_label  # 将噪声点的标签更新为最近的非噪声点的标签

        # 计算轮廓系数
        if len(set(new_labels)) > 1:  # 至少有两个簇才能计算轮廓系数
            score = silhouette_score(X, new_labels)
            if score > best_score:
                best_score = score
                best_params['eps'] = eps
                best_params['min_samples'] = min_samples

                # 初始化聚类结果的列表形式
                clusters = {label: [] for label in set(new_labels)}
                for idx, label in enumerate(new_labels):
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

    return best_clusters, len(best_clusters), best_centers, best_params


