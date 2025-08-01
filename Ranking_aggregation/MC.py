import os
from pathlib import Path
import pandas as pd
import numpy as np


def MC_rank_aggregation(sorted_clusters, w):

        scores = mc4_ranking_aggregation(sorted_clusters, w)
        # print(aggregated_ranking)
        ranking = []
        i = 0
        for score in scores:
            ranking.append((sorted_clusters[0][i][0], float(score)))
            i += 1
        # print(ranking)
        ranking = sorted(ranking, key=lambda x: x[1])

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
def create_pairwise_comparison_matrix(rankings, w):
    # print(rankings)
    n = len(rankings[0])
    comparison_matrix = np.zeros((n, n))
    # print(comparison_matrix)
    k = 0
    for ranking in rankings:
        # print(ranking)
        for i in range(n):
            for j in range(n):
                if ranking[j][1] - ranking[i][1] == 1:
                    comparison_matrix[i][j] += w[k]
        k += 1
    # 归一化每行
    row_sums = comparison_matrix.sum(axis=1, keepdims=True)
    # 替换零值以避免除以零
    row_sums[row_sums == 0] = 1e-10
    P = comparison_matrix / row_sums
    return P

def mc4_ranking_aggregation(rankings, w):
    # 初始化状态向量（均匀分布）
    n = len(rankings[0])
    s = np.ones(n) / n

    # 设置收敛阈值和最大迭代次数
    threshold = 1e-6
    max_iterations = 1000

    P = create_pairwise_comparison_matrix(rankings, w)
    # print(P)
    # 迭代计算平稳分布
    for iteration in range(max_iterations):
        s_new = s @ P
        if np.linalg.norm(s_new - s) < threshold:
            break
        s = s_new
    # print(s)
    # objects = pd.DataFrame(rankings[0]).iloc[:, 0]
    # print(objects)
    # 输出平稳分布
    # final_ranking = list(zip(objects, s))

    return s
