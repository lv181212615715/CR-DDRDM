import os
from pathlib import Path
import pandas as pd
import numpy as np
import re

from scipy.linalg import eig


def REV_rank_aggregation(sorted_clusters, w):
    preference_matrices = [build_preference_matrix(cluster) for cluster in sorted_clusters]
    # print(preference_matrices)
    MCM = build_MCM_matrix(preference_matrices, w)
    # print(MCM)
    ERV = culculate_ERV(MCM)

    total_scores = ERV

    ranking = []
    i = 0
    for score in total_scores:
        ranking.append((sorted_clusters[0][i][0], float(score)))
        i += 1
    # print(ranking)
    ranking = sorted(ranking, key=lambda x: -x[1])

    # 初始化排名列表
    rankings = []
    current_rank = 1
    previous_score = None

    for idx, score in ranking:
        if score != previous_score:
            # 如果当前得分与上一个得分不同，则分配新的排名
            current_rank = len(rankings) + 1
        rankings.append((idx, current_rank))
        previous_score = score
    # print(rankings)
    final_rankings = sorted(rankings, key=lambda x: x[0])

    return final_rankings


def build_preference_matrix(ranking):
    n = len(ranking)
    preference_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
                if ranking[i][1] < ranking[j][1]:
                    preference_matrix[i, j] = 1
                elif ranking[i][1] > ranking[j][1]:
                    preference_matrix[i, j] = 0
                elif ranking[i][1] == ranking[j][1]:
                    preference_matrix[i, j] = 0.5
    return preference_matrix


def build_MCM_matrix(preference_matrixs, w):
    n = preference_matrixs[0].shape[0]
    m = len(preference_matrixs)
    MCM = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            M = 0
            for k in range(m):
                if preference_matrixs[k][i, j] == 1:
                    M += 1 * w[k]
                elif preference_matrixs[k][i, j] == 0.5:
                    M += 0.5 * w[k]
            MCM[i, j] = M / m

    return MCM

def culculate_ERV(MCM):
    # 计算特征值和特征向量
    [eigenvalues, eigenvectors] = eig(MCM)
    # print(eigenvalues)
    # print(eigenvectors)

    # 找到最大特征值的索引
    max_eigenvalue_index = np.argmax(np.abs(eigenvalues))

    # 获取对应于最大特征值的特征向量
    ERV = abs(eigenvectors[:, max_eigenvalue_index])
    return ERV








