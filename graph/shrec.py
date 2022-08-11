import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools


#ntu kinect
num_node = 22
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(0, 1),
        (1, 2), (2, 3), (3, 4), (4, 5),
        (1, 6), (6, 7), (7, 8), (8, 9),
        (1, 10), (10, 11), (11, 12), (12, 13),
        (1, 14), (14, 15), (15, 16), (16, 17),
        (1, 18), (18, 19), (19, 20), (20, 21)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward



class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.hand_A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A