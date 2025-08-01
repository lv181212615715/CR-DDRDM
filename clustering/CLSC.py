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


def plot_elbow_curve(X, max_k):
    """Plot elbow curve to help determine optimal number of clusters.

    Args:
        X: Input data matrix
        max_k: Maximum number of clusters to evaluate
    """
    sse = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        # Note: Using K-means directly on original data X to compute SSE
        # This is separate from CLSC algorithm, only for elbow method evaluation
        sse.append(calculate_sse(X, k))

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, 'bx-')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Sum of squared errors (SSE)')
    plt.title('Elbow Method For Optimal k (Direct K-means)')
    plt.grid(True)
    # plt.show()


def calculate_sse(X, k):
    """Calculate sum of squared errors for K-means clustering.

    Args:
        X: Input data matrix
        k: Number of clusters

    Returns:
        Sum of squared errors
    """
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    return np.sum(np.min(kmeans.transform(X) ** 2, axis=1))


def cosine_similarity_matrix(X):
    """Compute pairwise cosine similarity matrix.

    Args:
        X: Input data matrix

    Returns:
        Cosine similarity matrix
    """
    return cosine_similarity(X)


def normalized_laplacian(similarity_matrix):
    """Compute normalized Laplacian matrix.

    Args:
        similarity_matrix: Input similarity matrix

    Returns:
        Normalized Laplacian matrix
    """
    D = np.diag(np.sum(similarity_matrix, axis=1))
    L = D - similarity_matrix
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(similarity_matrix, axis=1)))
    return np.dot(np.dot(D_inv_sqrt, L), D_inv_sqrt)


def lanczos_decomposition(L, k):
    """Compute k smallest eigenvalues and eigenvectors using Lanczos algorithm.

    Args:
        L: Input matrix
        k: Number of eigenvalues/eigenvectors to compute

    Returns:
        Eigenvectors corresponding to k smallest eigenvalues
    """
    eigvals, eigvecs = eigsh(L, k=k, which='SM')
    return eigvecs


def CLSC(X, k):
    """Cosine Similarity-based Spectral Clustering (CLSC) algorithm.

    Args:
        X: Input data matrix
        k: Number of clusters

    Returns:
        final_groups: List of clusters (each cluster contains object indices)
        labels: Cluster labels for each object
        similarity_matrix: Computed cosine similarity matrix
    """
    # Step 1: Compute cosine similarity matrix
    similarity_matrix = cosine_similarity_matrix(X)

    # Step 2: Compute normalized Laplacian
    L = normalized_laplacian(similarity_matrix)

    # Step 3: Perform Lanczos decomposition to get eigenvectors
    eigvecs = lanczos_decomposition(L, k)

    # Step 4: Cluster eigenvectors using K-means++
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0).fit(eigvecs)
    labels = kmeans.labels_

    # Initialize cluster dictionary (handles noise points with label -1)
    clusters = {label: [] for label in set(labels)}
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    # Convert to list of clusters
    result = [clusters[label] for label in sorted(clusters.keys())]

    # Step 5: Post-processing to ensure each cluster has at least 2 objects
    final_groups = []
    single_objects = []

    # Separate valid clusters from single-object clusters
    for group in result:
        if len(group) >= 2:
            final_groups.append(group)
        else:
            single_objects.extend(group)

    # Reassign single objects to most similar clusters
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