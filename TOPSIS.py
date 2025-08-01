import numpy as np
from scipy.stats import rankdata


def topsis_sort(cluster):
    # print(cluster)
    """
    使用TOPSIS方法对簇中的对象进行排序，并按照原始顺序输出对象及其排名。

    参数:
    cluster (list of list of float): 簇中的对象，每行代表一个对象，每列代表一个属性。

    返回:
    list of tuple: 原始顺序的对象（作为索引）和对应排名的列表。
    """
    # 标准化数据
    norm_data = _normalize(cluster)

    # 确定理想解和负理想解
    ideal_solution = np.max(norm_data, axis=0)
    negative_ideal_solution = np.min(norm_data, axis=0)

    # 计算接近度
    closeness = []
    for row in norm_data:
        dist_to_ideal = np.linalg.norm(row - ideal_solution)
        dist_to_negative_ideal = np.linalg.norm(row - negative_ideal_solution)
        # 添加一个小的常数以避免除以零
        denominator = dist_to_ideal + dist_to_negative_ideal + 1e-10
        closeness.append(-dist_to_negative_ideal / denominator)
    # print(closeness)
    # 使用scipy.stats.rankdata来处理相同接近度的排名
    ranks = rankdata(closeness, method='min')
    # print(ranks)
    # 创建一个字典来映射排序后的索引到原始索引和排名
    sorted_dict = [(cluster.index[i], ranks[i]) for i in range(len(cluster.index))]
    # print(sorted_dict)
    frankings = []
    final_rankings = sorted(sorted_dict, key=lambda x: x[0])
    # print("Rankings sorted by object index:")
    for idx, rank in final_rankings:
        # print(f"Object {idx} is ranked {rank}")
        frankings.append((int(idx), float(rank)))

    return frankings


def _normalize(data):
    """
    标准化数据，使每个属性的值在0到1之间。

    参数:
    data (list of list of float): 待标准化的数据。

    返回:
    numpy.ndarray: 标准化后的数据。
    """
    X = np.array(data)
    col_max = np.max(X, axis=0)
    col_min = np.min(X, axis=0)
    # 避免除以零
    epsilon = 1e-10
    norm_X = (X - col_min) / (col_max - col_min + epsilon)
    return norm_X