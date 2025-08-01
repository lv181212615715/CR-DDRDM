# -*- coding: utf-8 -*-
from itertools import product

import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, OPTICS, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def safe_auto_eps(X, k=5, min_eps=0.1):
    """带保护的eps自动计算"""
    if len(X) <= 1:
        return min_eps

    k = min(k, len(X) - 1)
    knn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = knn.kneighbors(X)
    eps = np.percentile(distances[:, -1], 50)
    return max(eps, min_eps)


def safe_auto_k(X, max_k=10):
    """带保护的k自动计算"""
    if len(X) <= 2:
        return 1

    max_k = min(max_k, len(X) - 1)
    best_k, best_score = 1, -1
    for k in range(1, max_k + 1):
        try:
            labels = KMeans(n_clusters=k).fit_predict(X)
            if len(np.unique(labels)) >= 2:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_k, best_score = k, score
        except:
            continue
    return best_k

def DBSCAN_clustering(X):
    # 数据标准化
    X_scaled = StandardScaler().fit_transform(X)

    # 自适应参数计算（更保守的设置）
    min_samples = max(2, min(5, int(0.05 * len(X))))  # 更小的min_samples比例

    # 更稳健的eps计算
    if len(X) > 1:
        knn = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
        distances, _ = knn.kneighbors(X_scaled)
        eps = np.percentile(distances[:, -1], 50)  # 从75改为50
        eps = max(eps, 0.5)  # 设置最小eps阈值
    else:
        eps = 0.5

    # 执行聚类
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    labels = db.labels_

    # 处理结果
    unique_labels = set(labels) - {-1}
    if not unique_labels:  # 全是噪声点的情况
        return [[]], 0, [], {'eps': eps, 'min_samples': min_samples, 'note': 'all noise'}

    clusters = []
    centers = []
    for k in unique_labels:
        mask = labels == k
        clusters.append(np.where(mask)[0].tolist())
        centers.append(X[mask].mean(axis=0))  # 用原始数据计算中心

    return clusters, len(centers), centers, {'eps': eps, 'min_samples': min_samples}


def KMeans_clustering(X):
    """更健壮的KMeans实现"""
    n_clusters = safe_auto_k(X)
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters).fit(X_scaled)
    labels = km.labels_

    clusters = []
    centers = []
    for k in range(n_clusters):
        mask = labels == k
        clusters.append(np.where(mask)[0].tolist())
        centers.append(X[mask].mean(axis=0))  # 使用原始数据计算中心

    params = {'n_clusters': n_clusters}
    return clusters, n_clusters, centers, params


def process_cluster_results(X, labels):
    """统一处理聚类结果"""
    clusters = []
    centers = []
    for k in set(labels):
        if k == -1:
            continue
        mask = labels == k
        clusters.append(np.where(mask)[0].tolist())
        centers.append(X[mask].mean(axis=0))
    return clusters, centers


# 其他聚类方法也使用类似的保护机制...


def Agglomerative_clustering(X):
    """自适应的层次聚类实现"""
    n_clusters = safe_auto_k(X)
    agg = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    labels = agg.labels_

    clusters = []
    centers = []
    for k in range(n_clusters):
        mask = labels == k
        clusters.append(np.where(mask)[0].tolist())
        centers.append(X[mask].mean(axis=0))

    params = {'auto_n_clusters': n_clusters}
    return clusters, n_clusters, centers, params


def GMM_clustering(X):
    """自适应的GMM实现"""
    n_components = safe_auto_k(X)
    gmm = GaussianMixture(n_components=n_components).fit(X)
    labels = gmm.predict(X)

    clusters = []
    centers = []
    for k in range(n_components):
        mask = labels == k
        clusters.append(np.where(mask)[0].tolist())
        centers.append(gmm.means_[k])

    params = {'auto_n_components': n_components}
    return clusters, n_components, centers, params


def OPTICS_clustering(X):
    """自适应的OPTICS实现"""
    min_samples = max(2, int(0.01 * len(X)))
    optics = OPTICS(min_samples=min_samples, xi=0.05).fit(X)
    labels = optics.labels_
    unique_labels = set(labels) - {-1}
    n_clust = len(unique_labels)

    clusters = []
    centers = []
    for k in unique_labels:
        mask = labels == k
        clusters.append(np.where(mask)[0].tolist())
        centers.append(X[mask].mean(axis=0))

    params = {'auto_min_samples': min_samples, 'xi': 0.05}
    return clusters, n_clust, centers, params


def Spectral_clustering(X):
    """自适应的谱聚类实现"""
    n_clusters = safe_auto_k(X)
    sc = SpectralClustering(n_clusters=n_clusters,
                            affinity='nearest_neighbors',
                            n_neighbors=min(10, len(X) - 1)).fit(X)
    labels = sc.labels_

    clusters = []
    centers = []
    for k in range(n_clusters):
        mask = labels == k
        clusters.append(np.where(mask)[0].tolist())
        centers.append(X[mask].mean(axis=0))

    params = {'auto_n_clusters': n_clusters, 'n_neighbors': 'adaptive'}
    return clusters, n_clusters, centers, params