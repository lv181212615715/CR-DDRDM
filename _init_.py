import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import PSO
import Inner_feedback_1 as If1
import splicing_integration as si
import test as t
import clustering.DBSCAN as db

import time

from clustering.CLSC import CLSC

start_total_time = time.time()

# Load data (2D list where each row represents an object and each column represents an evaluation)
data = pd.read_csv(r'data\filtered_data0.csv').iloc[:, :-2]

# DBSCAN-TOPSIS-Copeland feedback
# Output records
best_objects_clusters = None  # Optimal object clustering result
best_params = None  # Optimal clustering parameters
best_evaluation_clusters = None  # Optimal evaluation clustering result
best_sorted_clusters = None  # Optimal intra-cluster sorting result
best_consensus_data = None  # Optimal consensus aggregation result
best_consensus_degrees = None  # Optimal consensus degree
best_sorted_data = None  # Optimal sorting result

# Storage
top_rates = []
fin_rates = []

# Object clustering
objects_clusters, labels, similarity_matrix = CLSC(data, 30)
print(objects_clusters, len(objects_clusters))

# Inner feedback output records
sorted_objects_clusters = []
params = []
evaluation_clusters = []
sorted_clusters = []
consensus_datas = []
consensus_degrees = []

# Perform inner feedback for each object cluster
for i, cluster in enumerate(objects_clusters):
    start_cluster_time = time.time()

    eps_SCL, eps_GCL = 0.9, 0.9
    clusters1, num1, core_data1, params1 = db.DBSCAN_clustering_1(data.iloc[cluster, :])
    eps_LL, eps_HL, eps_LH = PSO.pso(data.iloc[cluster, :], eps_SCL, eps_GCL, clusters1, num1, core_data1, params1)

    # Inner feedback
    param, eval_cluster, sorted_cluster, consensus_data, consensus_degree = If1.inner_feedback_1(
        data.iloc[cluster, :], eps_SCL, eps_GCL, eps_LL, eps_HL, eps_LH, i)

    # Record output
    params.append(param)
    evaluation_clusters.append(eval_cluster)
    sorted_clusters.append(sorted_cluster)
    consensus_datas.append(consensus_data)
    consensus_degrees.append(consensus_degree)

    # Record end time and calculate duration for each loop
    end_cluster_time = time.time()
    cluster_duration = end_cluster_time - start_cluster_time
    print(f"Cluster {i + 1} processing time: {cluster_duration:.4f} seconds")

    sorted_objects_clusters.append(consensus_data)  # Sorting result for each object cluster
    print(f'object_cluster{i} evaluation data clustering result:', eval_cluster)
    print(f'object_cluster{i} TOPSIS sorting result for each evaluation cluster:', sorted_cluster)
    print(f'object_cluster{i} consensus sorting result:', consensus_data)

# Splicing and integration
sorted_data, merged_size = si.final_integration(data, sorted_objects_clusters, similarity_matrix)

# Store for plotting
top_rate, fin_rate = t.test1(sorted_data)
top_rates.append(top_rate)
fin_rates.append(fin_rate)

best_objects_clusters = objects_clusters
best_params = params
best_evaluation_clusters = evaluation_clusters
best_sorted_clusters = sorted_clusters
best_consensus_data = consensus_datas
best_consensus_degrees = consensus_degrees
best_sorted_data = sorted_data

best_sorted_data.to_csv(r'results.csv', index=False)

print(best_objects_clusters)
print(best_params)
print(best_evaluation_clusters)
print(best_sorted_clusters)
print(best_consensus_data)
print(best_consensus_degrees)
print(best_sorted_data)

# Record total end time and calculate total duration
end_total_time = time.time()
total_duration = end_total_time - start_total_time
print(f"Total processing time: {total_duration:.4f} seconds")

# Effectiveness testing
t.test1(best_sorted_data)