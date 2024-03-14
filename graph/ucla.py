import copy
import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 20
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 3), (4, 3), (5, 3), (6, 5), (7, 6),
                    (8, 7), (9, 3), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

edge = inward + outward + self_link


class MyGraph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.edge = edge

        self.hop_dis = tools.get_hop_distance(
            self.num_node, self.edge, max_hop=1)
        self.A6 = self.get_adjacency_matrix_A_k(6, tools.get_k_body_parts_ucla(6), labeling_mode)
        self.A8 = self.get_adjacency_matrix_A_k(8, tools.get_k_body_parts_ucla(8), labeling_mode)
        self.A3 = self.get_adjacency_matrix_A3(labeling_mode)
        self.spd_A = copy.deepcopy(self.A6)

    def get_adjacency_matrix_A_k(self, k, partition_body, labeling_mode=None):
        if labeling_mode is None:
            return self.A6
        if labeling_mode == 'spatial':
            adjacency_matrix = tools.get_spatial_graph_new(num_node, edge)
            if k == 6:
                Ak = np.zeros((6, 20, 20))
            elif k == 8:
                Ak = np.zeros((8, 20, 20))
            else:
                raise ValueError()

            for hop in range(2):
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            part_indices_j = tools.get_part_index(partition_body, j)
                            Ak[part_indices_j, i, j] = adjacency_matrix[i, j]
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


class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
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


if __name__ == '__main__':
    graph = MyGraph()
    print()
