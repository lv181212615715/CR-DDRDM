import os
from pathlib import Path
import pandas as pd
import numpy as np
import re


def Borda_rank_aggregation(sorted_clusters, w):
    B = compute_score_B(sorted_clusters, w)

    total_scores = B

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


# def Borda_rank_aggregation(sorted_clusters):
#     B = compute_score_B(sorted_clusters)
#
#     total_scores = B
#
#     ranking = []
#     i = 0
#     for score in total_scores:
#         ranking.append((sorted_clusters[0][i][0], int(score)))
#         i += 1
#     # print(ranking)
#     ranking = sorted(ranking, key=lambda x: -x[1])
#
#     # 初始化排名列表
#     rankings = []
#     current_rank = 1
#     previous_score = None
#
#     for idx, score in ranking:
#         if score != previous_score:
#             # 如果当前得分与上一个得分不同，则分配新的排名
#             current_rank = len(rankings) + 1
#         rankings.append((idx, current_rank))
#         previous_score = score
#     # print(rankings)
#     final_rankings = sorted(rankings, key=lambda x: x[0])
#
#     return final_rankings


def compute_score_B(rankings, w):
    n = len(rankings[0])
    B = [0] * n
    k = 0
    for ranking_k in rankings:
        for i in range(n):
            B[i] += (n - ranking_k[i][1]) * w[k]
        k += 1
    return B


# def compute_score_B(rankings):
#     n = len(rankings[0])
#     B = [0] * n
#     k = 0
#     for ranking_k in rankings:
#         for i in range(n):
#             B[i] += n - ranking_k[i][1]
#         k += 1
#     return B
