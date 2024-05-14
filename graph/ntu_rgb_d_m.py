import copy
import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

edge = inward + outward + self_link

partition_label = [3, 3, 2, 2, 3, 1, 1, 1, 3, 0, 0, 0, 5, 5, 5, 5, 4, 4, 4, 4, 3, 1, 1, 0, 0]


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


# 将人体分为6个部分：左腿、右腿、躯干、左手、右手、头部
class MyGraph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.edge = edge

        self.hop_dis = tools.get_hop_distance(
            self.num_node, self.edge, max_hop=1)
        self.A6 = self.get_adjacency_matrix_A_k(6, tools.get_k_body_parts_ntu(6), labeling_mode)
        self.A6_ones = self.get_adjacency_matrix_A_partly(6, tools.get_k_body_parts_ntu(6), labeling_mode)
        self.A8 = self.get_adjacency_matrix_A_k(8, tools.get_k_body_parts_ntu(8), labeling_mode)
        self.A3 = self.get_adjacency_matrix_A3(labeling_mode)
        self.spd_A = copy.deepcopy(self.A6)

    def get_adjacency_matrix_A_k(self, k, partition_body, labeling_mode=None):
        if labeling_mode is None:
            return self.A6
        if labeling_mode == 'spatial':
            A = np.zeros((25, 25), dtype=np.int32)

            h = {}
            cnt = 6
            for i in range(self.num_node):
                for j in range(self.num_node):
                    indices_i, indices_j = partition_label[i], partition_label[j]
                    if self.hop_dis[j, i] <= 1:
                        if indices_i == indices_j:
                            A[i, j] = A[j, i] = indices_j
                        else:
                            A[i, j] = indices_i
                            A[j, i] = indices_j
                    else:
                        if not h.get(f'{indices_i}-{indices_j}'):
                            h[f'{indices_i}-{indices_j}'] = cnt
                            cnt = cnt + 1
                        A[i, j] = h[f'{indices_i}-{indices_j}']

                        if not h.get(f'{indices_j}-{indices_i}'):
                            h[f'{indices_j}-{indices_i}'] = cnt
                            cnt = cnt + 1
                        A[j, i] = h[f'{indices_j}-{indices_i}']

        else:
            raise ValueError()
        return A

    def get_adjacency_matrix_A_partly(self, k, partition_body, labeling_mode=None):
        if labeling_mode is None:
            return self.A6
        if labeling_mode == 'spatial':
            if k == 6:
                Ak = np.zeros((6, 25, 25), dtype=np.float32)
            elif k == 8:
                Ak = np.zeros((8, 25, 25), dtype=np.float32)
            else:
                raise ValueError()

            for idx, partition in enumerate(partition_body):
                for i in partition:
                    for j in partition:
                        if self.hop_dis[i, j] == 1 or self.hop_dis[i, j] == 0:
                            Ak[idx, i, j] = Ak[idx, j, i] = 1

        else:
            raise ValueError()
        return Ak


    def get_adjacency_matrix_A3(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A3
        if labeling_mode == 'spatial':
            A3 = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A3


if __name__ == '__main__':
    graph = MyGraph()
    A = graph.A6
    for i in range(25):
        for j in range(25):
            print(A[i][j], end=' ')
        print()