import sys
import numpy as np
import torch

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

left_arm = [10, 11, 12, 24, 25]
right_arm = [6, 7, 8, 22, 23]
head = [3, 4]
body = [1, 2, 5, 9, 21]
left_leg = [17, 18, 19, 20]
right_leg = [13, 14, 15, 16]

partition_body = [left_arm, right_arm, head, body, left_leg, right_leg]
partition_body = [[index - 1 for index in part] for part in partition_body]


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
        # self.inward = inward
        # self.outward = outward
        # self.neighbor = neighbor
        self.partition_body = np.array(partition_body)
        self.hop_dis = tools.get_hop_distance(
            self.num_node, self.edge, max_hop=1)
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            adjacency_matrix = tools.get_spatial_graph_new(num_node, edge)
            A = np.zeros((6, 25, 25))

            for hop in range(2):
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            part_indices = tools.get_part_index(self.partition_body, j)
                            A[part_indices, i, j] += adjacency_matrix[i, j]
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    graph = MyGraph()
    for i in range(25):
        for j in range(25):
            print(graph.A[0, i, j], end=' ')
        print()
