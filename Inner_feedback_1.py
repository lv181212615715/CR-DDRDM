import numpy as np
import pandas as pd
import clustering.DBSCAN as db
import TOPSIS as ts
import Ranking_aggregation.Copeland as cl

def rank_column(column):
    """Calculate rank for each column (higher values get better ranks)"""
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
    """Calculate distance between two rankings"""
    ranking_matrix_1 = build_PCM_d(ranking_1)
    ranking_matrix_2 = build_PCM_d(ranking_2)

    # Vectorized calculation of upper triangular differences
    diff_triu = np.triu(ranking_matrix_1, 1) - np.triu(ranking_matrix_2, 1)
    d = np.sum(np.abs(diff_triu))
    n = diff_triu.shape[0]
    total_count = (n * (n - 1)) // 2
    return d / (total_count * 2 * (n - 1))


def calculate_cp_k(data_k, cluster_center_k, m_k):
    """Calculate cluster compactness (cp_k) for cluster k"""
    ranked_df = data_k.apply(rank_column, axis=0)
    ranked_center = cluster_center_k.rank(method='min', ascending=False).astype(int)
    cp_k = sum(1 - calculate_d(ranked_df.iloc[:, a], ranked_center) for a in range(m_k))
    return cp_k / m_k


def calculate_cl_k(data_k, sorted_cluster_k, m_k):
    """Calculate consensus level (cl_k) for cluster k"""
    ranked_df = data_k.apply(rank_column, axis=0)
    sorted_cluster_k = pd.DataFrame(sorted_cluster_k, index=ranked_df.index).drop(labels=0, axis=1)
    cl_k = [1 - calculate_d(ranked_df.iloc[:, a], sorted_cluster_k) for a in range(m_k)]
    return cl_k


def inner_feedback_1(data, eps_SCL, eps_GCL, eps_LL, eps_HL, eps_LH, i):
    """Main inner feedback function with consensus-based clustering adjustment"""
    objective_function_value = 0

    # Perform DBSCAN clustering
    clusters, num, core_data, params = db.DBSCAN_clustering_1(data)
    print(params, num)
    print("Cluster centers:", core_data)

    # Calculate weights (w) based on compactness (cp) and size (m)
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

    # Calculate density (ro) and weights (w)
    for k in range(num):
        ro_k = (cp[k] / sum(cp)) * (m[k] / sum(m))
        ro.append(ro_k)
    w = [ro_k / sum(ro) for ro_k in ro]
    print("Cluster weights:", w)

    feedback = 'recluster'

    # Feedback loop for reclustering
    while feedback == 'recluster':
        # Perform TOPSIS sorting for each cluster
        sorted_clusters = [ts.topsis_sort(data.iloc[:, cluster_k]) for cluster_k in clusters]

        # Consensus ranking aggregation (using Copeland method)
        consensus_data = cl.copeland_rank_aggregation(sorted_clusters, w)
        consensus_data = pd.DataFrame(consensus_data, index=data.index).drop(labels=0, axis=1)

        # Calculate consensus levels
        CL = [calculate_cl_k(data.iloc[:, cluster_k], sorted_clusters[k], len(cluster_k))
              for k, cluster_k in enumerate(clusters)]
        SCL = [float(sum(CL_k) / len(cluster_k)) for CL_k, cluster_k in zip(CL, clusters)]
        ICL = [1 - float(calculate_d(pd.DataFrame(sorted_clusters[k], index=data.index).drop(0, axis=1),
                                     consensus_data)) for k in range(len(clusters))]
        GCL = np.dot(ICL, w)

        print("Intra-cluster consensus:", SCL)
        print("Inter-cluster consensus:", ICL)
        print("Global consensus:", GCL)

        # Feedback mechanism
        if GCL < eps_GCL:
            feedback = 'recluster'
            ranked_data = data.apply(rank_column, axis=0).astype(float)
            sita = 0.5  # Adjustment parameter

            for k, cluster_k in enumerate(clusters):
                ranked_data_k = ranked_data.iloc[:, cluster_k]
                sorted_clusters_k = pd.DataFrame(sorted_clusters[k], index=ranked_data.index).drop(0, axis=1)
                mod = 0

                if SCL[k] < eps_SCL and ICL[k] < eps_GCL:  # Low-Low case
                    print(k, ':Low-Low')
                    for a, CL_a in enumerate(CL[k]):
                        if CL_a < eps_SCL:
                            mod_a = ((1 - eps_LL) * ranked_data_k.iloc[:, a] +
                                     eps_LL * ((1 - sita) * sorted_clusters_k.iloc[:, 0] +
                                               sita * consensus_data.iloc[:, 0]))
                            mod += sum(mod_a - ranked_data_k.iloc[:, a])
                            ranked_data_k.iloc[:, a] = mod_a
                    objective_function_value += w[k] * eps_LL * mod

                elif SCL[k] >= eps_SCL and ICL[k] < eps_GCL:  # High-Low case
                    print(k, ':High-Low')
                    for a in range(len(CL[k])):
                        mod_a = ((1 - eps_HL) * ranked_data_k.iloc[:, a] +
                                 eps_HL * consensus_data.iloc[:, 0])
                        mod += sum(mod_a - ranked_data_k.iloc[:, a])
                        ranked_data_k.iloc[:, a] = mod_a
                    objective_function_value += w[k] * eps_HL * mod

                elif SCL[k] < eps_SCL and ICL[k] >= eps_GCL:  # Low-High case
                    print(k, ':Low-High')
                    for a, CL_a in enumerate(CL[k]):
                        if CL_a < eps_SCL:
                            mod_a = ((1 - eps_LH) * ranked_data_k.iloc[:, a] +
                            eps_LH * sorted_clusters_k.iloc[:, 0])
                            mod += sum(mod_a - ranked_data_k.iloc[:, a])
                            ranked_data_k.iloc[:, a] = mod_a
                    objective_function_value += w[k] * eps_LH * mod

                # Update the ranked data
                ranked_data.iloc[:, cluster_k] = ranked_data_k

            # Update the data for next iteration
            data = -ranked_data

        else:
            # Consensus reached - return results
            print("Optimal intra-cluster consensus:", SCL)
            print("Optimal inter-cluster consensus:", ICL)
            print("Optimal global consensus:", GCL)
            print("Objective function value:", objective_function_value)

            return params, clusters, sorted_clusters, consensus_data, GCL