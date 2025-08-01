import numpy as np
import pandas as pd

import clustering.Kmeans as km
import clustering.DBSCAN as db
import TOPSIS as ts
import Ranking_aggregation.Copeland as cl
import Ranking_aggregation.Borda as bd
import Ranking_aggregation.REV as rev
import Ranking_aggregation.MC as mc
import Ranking_aggregation as cc

import random

from clustering.CLSC import CLSC


def rank_column(column):
    """Rank column values in descending order (higher values get better ranks)"""
    ranks = column.rank(method='min', ascending=False).astype(int)
    return ranks


def build_PCM_d(ranking):
    """Build Pairwise Comparison Matrix for ranking distance calculation"""
    n = ranking.shape[0]
    ranking_matrix = ranking.values.reshape(n, 1)
    preference_matrix = np.abs(ranking_matrix - ranking_matrix.T)
    np.fill_diagonal(preference_matrix, 0)  # Set diagonal to 0
    return preference_matrix


def calculate_d(ranking_1, ranking_2):
    """Calculate normalized distance between two rankings"""
    ranking_matrix_1 = build_PCM_d(ranking_1)
    ranking_matrix_2 = build_PCM_d(ranking_2)

    # Compare only upper triangular part
    diff = np.triu(ranking_matrix_1, 1) - np.triu(ranking_matrix_2, 1)
    d = np.sum(np.abs(diff))

    n = diff.shape[0]
    total_count = (n * (n - 1)) // 2

    return d / (total_count * 2 * (n - 1))


def calculate_cp_k(data_k, cluster_center_k, m_k):
    """Calculate cluster compactness for cluster k"""
    ranked_df = data_k.apply(rank_column, axis=0)
    ranked_center = cluster_center_k.rank(method='min', ascending=False).astype(int)
    cp_k = sum(1 - calculate_d(ranked_df.iloc[:, a], ranked_center) for a in range(m_k))
    return cp_k / m_k


def calculate_cl_k(data_k, sorted_cluster_k, m_k):
    """Calculate consensus level for cluster k"""
    ranked_df = data_k.apply(rank_column, axis=0)
    sorted_cluster_k = pd.DataFrame(sorted_cluster_k, index=ranked_df.index).drop(labels=0, axis=1)
    cl_k = [1 - calculate_d(ranked_df.iloc[:, a], sorted_cluster_k) for a in range(m_k)]
    return cl_k


def inner_feedback_2(data, clusters, num, core_data, params, eps_SCL, eps_GCL, eps_LL, eps_HL, eps_LH):
    """Perform consensus-based clustering with feedback mechanism"""
    print(f"Adjustment parameters: {eps_LL}, {eps_HL}, {eps_LH}")
    objective_function_value = 0

    # Print clustering parameters and centers
    print(f"Clustering parameters: {params}, Number of clusters: {num}")
    print("Cluster centers:", core_data)

    # Calculate weights based on cluster compactness and size
    cp = []  # Compactness values
    m = []  # Cluster sizes
    w = []  # Weights
    ro = []  # Density values

    for k, cluster_k in enumerate(clusters):
        data_k = data.iloc[:, cluster_k]
        m_k = len(cluster_k)
        cp_k = calculate_cp_k(data_k, data.iloc[:, core_data[k]], m_k)
        cp.append(float(cp_k))
        m.append(m_k)

    print("Cluster compactness:", cp)

    # Calculate density and weights
    for k in range(num):
        ro_k = (cp[k] / sum(cp)) * (m[k] / sum(m))
        ro.append(ro_k)
    w = [ro_k / sum(ro) for ro_k in ro]
    print("Cluster weights:", w)

    eps_GCL_LL = eps_GCL
    feedback = 'recluster'
    sita = 0.5  # Adjustment parameter

    # Feedback loop for consensus improvement
    while feedback == 'recluster':
        # Perform TOPSIS sorting for each cluster
        sorted_clusters = [ts.topsis_sort(data.iloc[:, cluster_k]) for cluster_k in clusters]

        # Aggregate rankings using Copeland method
        consensus_data = cl.copeland_rank_aggregation(sorted_clusters, w)
        consensus_data = pd.DataFrame(consensus_data, index=data.index).drop(labels=0, axis=1)

        # Calculate consensus metrics
        CL = [calculate_cl_k(data.iloc[:, cluster_k], sorted_clusters[k], len(cluster_k))
              for k, cluster_k in enumerate(clusters)]
        SCL = [float(sum(CL_k) / len(cluster_k)) for CL_k, cluster_k in zip(CL, clusters)]
        ICL = [1 - float(calculate_d(
            pd.DataFrame(sorted_clusters[k], index=data.index).drop(0, axis=1),
            consensus_data)) for k in range(len(clusters))]
        GCL = np.dot(ICL, w)

        print("Intra-cluster consensus (SCL):", SCL)
        print("Inter-cluster consensus (ICL):", ICL)
        print("Global consensus (GCL):", GCL)

        if GCL < eps_GCL:
            # Need to adjust rankings - convert data to ranks
            ranked_data = data.apply(rank_column, axis=0).astype(float)

            for k, cluster_k in enumerate(clusters):
                ranked_data_k = ranked_data.iloc[:, cluster_k]
                sorted_clusters_k = pd.DataFrame(sorted_clusters[k], index=ranked_data.index).drop(0, axis=1)
                mod = 0

                if SCL[k] < eps_SCL and ICL[k] < eps_GCL:  # Low-Low case
                    print(f"Cluster {k}: Low-Low adjustment")
                    for a, CL_a in enumerate(CL[k]):
                        if CL_a < eps_SCL:
                            mod_a = ((1 - eps_LL) * ranked_data_k.iloc[:, a] +
                                     eps_LL * ((1 - sita) * sorted_clusters_k.iloc[:, 0] +
                                               sita * consensus_data.iloc[:, 0]))
                            mod += sum(mod_a - ranked_data_k.iloc[:, a])
                            ranked_data_k.iloc[:, a] = mod_a
                    objective_function_value += w[k] * eps_LL * mod

                elif SCL[k] >= eps_SCL and ICL[k] < eps_GCL:  # High-Low case
                    print(f"Cluster {k}: High-Low adjustment")
                    for a in range(len(CL[k])):
                        mod_a = ((1 - eps_HL) * ranked_data_k.iloc[:, a] +
                                 eps_HL * consensus_data.iloc[:, 0])
                        mod += sum(mod_a - ranked_data_k.iloc[:, a])
                        ranked_data_k.iloc[:, a] = mod_a
                    objective_function_value += w[k] * eps_HL * mod

                elif SCL[k] < eps_SCL and ICL[k] >= eps_GCL:  # Low-High case
                    print(f"Cluster {k}: Low-High adjustment")
                    for a, CL_a in enumerate(CL[k]):
                        if CL_a < eps_SCL:
                            mod_a = ((1 - eps_LH) * ranked_data_k.iloc[:, a] +
                            eps_LH * sorted_clusters_k.iloc[:, 0])
                            mod += sum(mod_a - ranked_data_k.iloc[:, a])
                            ranked_data_k.iloc[:, a] = mod_a
                    objective_function_value += w[k] * eps_LH * mod

                # Update the ranked data
                ranked_data.iloc[:, cluster_k] = ranked_data_k

            # Update data for next iteration (invert ranks since TOPSIS expects higher=better)
            data = -ranked_data
            print(f"Adjusted GCL threshold: {eps_GCL_LL}")

        else:
            # Consensus reached - return results
            feedback = 'proceed'
            print("Optimal intra-cluster consensus (SCL):", SCL)
            print("Optimal inter-cluster consensus (ICL):", ICL)
            print("Optimal global consensus (GCL):", GCL)
            print("Final objective function value:", objective_function_value)

            return objective_function_value