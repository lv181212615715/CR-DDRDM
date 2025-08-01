import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import diags
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph


# 定义用于肘部法则的函数
def plot_elbow_curve(X, max_k):
    sse = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        # 注意：这里我们直接对原始数据X使用K-means来计算SSE
        # 这与CLSC算法分开，仅用于评估肘部法则
        sse.append(calculate_sse(X, k))

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, 'bx-')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Sum of squared errors (SSE)')
    plt.title('Elbow Method For Optimal k (Direct K-means)')
    plt.grid(True)
    # plt.show()


# 定义SSE计算函数
def calculate_sse(X, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    return np.sum(np.min(kmeans.transform(X)**2, axis=1))


# 构造余弦相似度矩阵
def cosine_similarity_matrix(X):
    return cosine_similarity(X)


# 归一化拉普拉斯矩阵
def normalized_laplacian(similarity_matrix):
    D = np.diag(np.sum(similarity_matrix, axis=1))
    L = D - similarity_matrix
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(similarity_matrix, axis=1)))
    return np.dot(np.dot(D_inv_sqrt, L), D_inv_sqrt)


# Lanczos迭代法求特征值和特征向量
def lanczos_decomposition(L, k):
    eigvals, eigvecs = eigsh(L, k=k, which='SM')
    return eigvecs


# CLSC算法
def CLSC(X, k):
    similarity_matrix = cosine_similarity_matrix(X)
    L = normalized_laplacian(similarity_matrix)
    eigvecs = lanczos_decomposition(L, k)

    # 使用K-means++进行聚类
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0).fit(eigvecs)
    labels = kmeans.labels_
    # 初始化聚类结果的列表形式，包括噪声点（-1标签）的处理
    clusters = {label: [] for label in set(labels)}
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    # final_groups = [clusters[label] for label in sorted(clusters.keys())]
    result = [clusters[label] for label in sorted(clusters.keys())]

    # 确保每个组至少有两个对象，并将单独对象分配到最相近的组中
    final_groups = []
    single_objects = []

    for group in result:
        if len(group) >= 2:
            final_groups.append(group)
        else:
            single_objects.extend(group)

    for obj in single_objects:
        best_group = None
        best_correlation = -1
        for group in final_groups:
            avg_correlation = similarity_matrix[obj, group].mean()
            if avg_correlation > best_correlation:
                best_correlation = avg_correlation
                best_group = group
        if best_group is not None:
            best_group.append(obj)

    return final_groups, labels, similarity_matrix


