import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from itertools import product
from sklearn.metrics import silhouette_score


def DBSCAN_clustering_1(X):
    """Optimized DBSCAN clustering with proper silhouette scoring for core points only.

    Args:
        X: Input data matrix (features x samples)

    Returns:
        best_clusters: List of clusters (each containing object indices)
        num_clusters: Number of clusters found
        best_centers: Representative center points for each cluster
        best_params: Dictionary of optimal parameters (eps, min_samples)
    """
    # Transpose input matrix (features x samples -> samples x features)
    X_data = np.array(X).T

    # Parameter search ranges (adjusted based on data characteristics)
    eps_range = np.arange(0.1, 1.1, 0.1).tolist()  # Slightly larger step for efficiency
    min_samples_range = [5, 10, 15, 20]

    # Initialize tracking variables
    best_score = -1
    best_params = {'eps': None, 'min_samples': None}
    best_labels = None
    best_clusters = None
    best_centers = None

    # Grid search over parameter combinations
    for eps, min_samples in product(eps_range, min_samples_range):
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_data)

        # Identify core points (non-noise)
        core_mask = labels != -1
        core_labels = labels[core_mask]

        # Skip if insufficient core points or clusters for silhouette score
        if np.sum(core_mask) < 2 or len(np.unique(core_labels)) < 2:
            continue

        # Calculate silhouette score using core points only
        score = silhouette_score(X_data[core_mask], core_labels)

        # Update best results if current score is better
        if score > best_score:
            best_score = score
            best_params['eps'] = eps
            best_params['min_samples'] = min_samples
            best_labels = labels.copy()  # Store original labels for later processing

    # If no valid clustering found, use default parameters
    if best_labels is None:
        best_params = {'eps': 0.5, 'min_samples': 10}
        dbscan = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'])
        best_labels = dbscan.fit_predict(X_data)

    # Step 2: Noise point reassignment (post-parameter-selection)
    noise_mask = best_labels == -1
    core_mask = ~noise_mask

    if np.any(noise_mask):
        # Build KDTree for efficient nearest neighbor search
        core_points = X_data[core_mask]
        core_labels = best_labels[core_mask]
        kdtree = KDTree(core_points)

        # Find nearest core point for each noise point
        noise_points = X_data[noise_mask]
        distances, indices = kdtree.query(noise_points, k=1)

        # Assign noise points to nearest core point's cluster
        for i, (noise_idx, core_idx) in enumerate(zip(np.where(noise_mask)[0], indices.flatten())):
            # Find original core point index in full dataset
            original_core_idx = np.where(core_mask)[0][core_idx]
            best_labels[noise_idx] = best_labels[original_core_idx]

    # Organize final clusters
    unique_labels = np.unique(best_labels[best_labels != -1])
    clusters = {label: [] for label in unique_labels}

    for idx, label in enumerate(best_labels):
        clusters[label].append(idx)

    # Calculate representative centers (geometric medoid)
    center_points = []
    for label in sorted(clusters.keys()):
        cluster_points = X_data[clusters[label]]

        if len(cluster_points) > 0:
            # Calculate geometric center
            geometric_center = np.mean(cluster_points, axis=0)

            # Find point closest to geometric center (medoid)
            distances = np.linalg.norm(cluster_points - geometric_center, axis=1)
            medoid_idx = np.argmin(distances)
            center_points.append(cluster_points[medoid_idx])
        else:
            center_points.append(None)

    # Prepare final cluster list
    final_clusters = [clusters[label] for label in sorted(clusters.keys())]

    return final_clusters, len(final_clusters), center_points, best_params