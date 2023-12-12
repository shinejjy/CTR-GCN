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

left_arm = [5, 6, 7, 8]
right_arm = [9, 10, 11, 12]
head = [3, 4]
body = [1, 2]
left_leg = [17, 18, 19, 20]
right_leg = [13, 14, 15, 16]

left_arm_part = [(24, 12), (25, 12), (12, 11), (11, 10)]
right_arm_part = [(22, 8), (23, 8), (8, 7), (7, 6)]
head_part = [(4, 3)]
body_part = [(1, 2), (2, 21), (21, 9), (21, 5)]
left_leg_part = [(20, 19), (19, 18), (18, 17)]
right_leg_part = [(16, 15), (15, 14), (14, 13)]

left_arm_part = [(i - 1, j - 1) for (i, j) in left_arm_part]
left_arm_part += [(j, i) for (i, j) in left_arm_part] + [(9 - 1, 10 - 1)] + [(i - 1, i - 1) for i in left_arm]

right_arm_part = [(i - 1, j - 1) for (i, j) in right_arm_part]
right_arm_part += [(j, i) for (i, j) in right_arm_part] + [(5 - 1, 6 - 1)] + [(i - 1, i - 1) for i in right_arm]

head_part = [(i - 1, j - 1) for (i, j) in head_part]
head_part += [(j, i) for (i, j) in head_part] + [(21 - 1, 5 - 1)] + [(i - 1, i - 1) for i in head]

body_part = [(i - 1, j - 1) for (i, j) in body_part]
body_part += [(j, i) for (i, j) in body_part] +\
             [(10 - 1, 9 - 1), (6 - 1, 5 - 1), (3 - 1, 21 - 1), (17 - 1, 1 - 1), (13 - 1, 1 - 1)] +\
             [(i - 1, i - 1) for i in body]

left_leg_part = [(i - 1, j - 1) for (i, j) in left_leg_part]
left_leg_part += [(j, i) for (i, j) in left_leg_part] + [(1 - 1, 17 - 1)] + [(i - 1, i - 1) for i in left_leg]

right_leg_part = [(i - 1, j - 1) for (i, j) in right_leg_part]
right_leg_part += [(j, i) for (i, j) in right_leg_part] + [(1 - 1, 13 - 1)] + [(i - 1, i - 1) for i in right_leg]


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
