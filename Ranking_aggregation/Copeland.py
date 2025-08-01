import numpy as np


def copeland_rank_aggregation(sorted_clusters, w):
    n = len(sorted_clusters[0])  # 假设所有簇中的对象数量相同
    scores = np.zeros((n, n))

    # 计算每对对象之间的胜负和平局数
    for i in range(n):
        for j in range(n):
            if i != j:
                wins = losses = ties = 0
                k = 0
                for cluster_k in sorted_clusters:
                    # print(cluster)
                    rank_i = rank_j = None
                    for idx, rank in cluster_k:
                        if cluster_k[i][0] == idx:
                            rank_i = rank
                        elif cluster_k[j][0] == idx:
                            rank_j = rank
                    if rank_i is not None and rank_j is not None:
                        # print(rank_i, rank_j)
                        if rank_i < rank_j:
                            wins += w[k]
                        elif rank_i > rank_j:
                            losses += w[k]
                        else:
                            ties += w[k]
                    k += 1
                scores[i, j] = wins - losses + 0 * ties

    # 根据得分矩阵计算每个对象的总得分，并确定排名
    total_scores = np.sum(scores, axis=1)
    # print(total_scores)
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

    # # 返回按得分排序的排名列表
    return final_rankings
