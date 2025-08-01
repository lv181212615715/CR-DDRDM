import numpy as np
from sklearn.cluster import DBSCAN
from itertools import product

from sklearn.metrics import silhouette_score


def DBSCAN_clustering_1(X):
    """Optimized DBSCAN clustering with automatic parameter selection.

    Args:
        X: Input data matrix (features x samples)

    Returns:
        best_clusters: List of clusters (each containing object indices)
        num_clusters: Number of clusters found
        best_centers: Representative center points for each cluster
        best_params: Dictionary of optimal parameters (eps, min_samples)
    """
    # Transpose input matrix (features x samples -> samples x features)
    X = np.array(X).T

    # Parameter search ranges
    eps_range = np.arange(0.001, 1.001, 0.05).tolist()
    min_samples_range = np.arange(5, 21, 5).tolist()

    # Initialize tracking variables
    best_score = -1
    best_params = {'eps': None, 'min_samples': None}
    best_clusters = None
    best_centers = None

    # Grid search over parameter combinations
    for eps, min_samples in product(eps_range, min_samples_range):
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = dbscan.labels_

        # Reassign noise points to nearest cluster
        noise_label = -1
        new_labels = np.copy(labels)
        if noise_label in set(labels):
            for i, label in enumerate(labels):
                if label == noise_label:
                    # Calculate distances to all non-noise points
                    distances = [(np.linalg.norm(X[i] - X[j]), j, lbl)
                                 for j, lbl in enumerate(labels) if lbl != noise_label]
                    # Find nearest non-noise point
                    if distances:
                        _, closest_idx, closest_label = min(distances, key=lambda x: x[0])
                        new_labels[i] = closest_label  # Assign to nearest cluster

        # Evaluate clustering quality
        if len(set(new_labels)) > 1:  # Need at least 2 clusters for silhouette score
            score = silhouette_score(X, new_labels)

            # Update best results if current score is better
            if score > best_score:
                best_score = score
                best_params['eps'] = eps
                best_params['min_samples'] = min_samples

                # Organize clusters by label
                clusters = {label: [] for label in set(new_labels)}
                for idx, label in enumerate(new_labels):
                    clusters[label].append(idx)

                # Calculate representative centers
                center_points = []
                for label, points in clusters.items():
                    geometric_center = np.mean(points, axis=0)
                    center_point = points[np.argmin([np.linalg.norm(point - geometric_center)
                                                     for point in points])]
                    center_points.append(center_point)

                # Sort clusters by label and store results
                best_clusters = [clusters[label] for label in sorted(clusters.keys())]
                best_centers = center_points

    return best_clusters, len(best_clusters), best_centers, best_params