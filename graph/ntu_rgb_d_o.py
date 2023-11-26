import copy
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

left_arm_part = [(24, 12), (25, 12), (12, 11), (11, 10)]
right_arm_part = [(22, 8), (23, 8), (8, 7), (7, 6)]
head_part = [(4, 3)]
body_part = [(1, 2), (2, 21), (21, 9), (21, 5)]
left_leg_part = [(20, 19), (19, 18), (18, 17)]
right_leg_part = [(16, 15), (15, 14), (14, 13)]

left_arm_part = [(i - 1, j - 1) for (i, j) in left_arm_part]
left_arm_part += [(j, i) for (i, j) in left_arm_part] + [(9 - 1, 10 - 1)] + [(i - 1, i - 1) for i in left_arm]
right_arm_part += [(j - 1, i - 1) for (i, j) in right_arm_part] + [(5 - 1, 6 - 1)] + [(i - 1, i - 1) for i in right_arm]
head_part += [(j - 1, i - 1) for (i, j) in head_part] + [(21 - 1, 5 - 1)] + [(i, i) for i in head]
body_part += [(j - 1, i - 1) for (i, j) in body_part] +\
             [(10 - 1, 9 - 1), (6 - 1, 5 - 1), (3 - 1, 21 - 1), (17 - 1, 1 - 1), (13 - 1, 1 - 1)] +\
             [(i - 1, i - 1) for i in body]
left_leg_part += [(j - 1, i - 1) for (i, j) in left_leg_part] + [(1 - 1, 17 - 1)] + [(i - 1, i - 1) for i in left_leg]
right_leg_part += [(j - 1, i - 1) for (i, j) in right_leg_part] + [(1 - 1, 13 - 1)] + [(i - 1, i - 1) for i in right_leg]


# import matplotlib.pyplot as plt
#
# # 提供的数据
# parts = [left_arm_part, right_arm_part, head_part, body_part, left_leg_part, right_leg_part]
#
# # 初始化一个空白的6x6子图矩阵
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#
# # 遍历每个部分并创建热图
# for i, ax in enumerate(axes.flat):
#     part = parts[i]
#     x, y = zip(*part)
#     matrix = [[1 if (j, i) in part else 0 for i in range(22)] for j in range(22)]
#     ax.imshow(matrix, cmap='coolwarm', interpolation='nearest')
#     ax.set_title(f'View {i + 1}')
#     ax.set_xlabel('X Coordinate')
#     ax.set_ylabel('Y Coordinate')
#
# # 调整布局
# plt.tight_layout()
# plt.show()

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

        self.A = self.get_adjacency_matrix_new(labeling_mode)
        self.spd_A = copy.deepcopy(self.A)

    def get_adjacency_matrix_new(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph_new_new(num_node, left_arm_part, right_arm_part, head_part,
                                                body_part, left_leg_part, right_leg_part)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    graph = MyGraph()
    print(graph.A)
