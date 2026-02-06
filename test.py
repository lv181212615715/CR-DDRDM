import numpy as np


def test1(data):


    data0 = np.genfromtxt(r'data\data0.csv', delimiter=',', skip_header=1)[:, 0]
    data0 = data0.astype(int)
    data = data.astype(int).iloc[:, 0]
    # print(data0.shape[0])
    # print(data.shape[0])


    top_ns = [16, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    fin_ns = [13, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # print(data0)
    # print(data)

    top_rate = []
    for top_n in top_ns:
        s = 0
        for i in data[:top_n]:
            for j in data0[:top_n]:
                if i == j:
                    s += 1
                    break
        top_rate.append(s / top_n)
    print(f'top_rate:{top_rate}')

    fin_rate = []
    for fin_n in fin_ns:
        s = 0
        for i in data[-fin_n:]:
            for j in data0[-fin_n:]:
                if i == j:
                    s += 1
                    break
        fin_rate.append(s / fin_n)
    print(f'fin_rate:{fin_rate}')

    return top_rate, fin_rate


