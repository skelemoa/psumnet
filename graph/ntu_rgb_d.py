import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools


#ntu kinect
num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

hand_num_node = 13
hand_self_link = [(i, i) for i in range(hand_num_node)]
hand_inward_ori_index = [(2,1),(3,2),(4,3),(5,4),(6,5),(7,5),
                    (8,1),(9,8),(10,9),(11,10),(12,11),(13,11)] 

hand_inward = [(i - 1, j - 1) for (i, j) in hand_inward_ori_index]
hand_outward = [(j, i) for (i, j) in hand_inward]
hand_neighbor = hand_inward + hand_outward

leg_num_node = 9
leg_self_link = [(i, i) for i in range(leg_num_node)]
leg_inward_ori_index = [(2,1), (3,2), (4,3), (5,4),
                    (5,1), (6,5), (7,6), (8,7)] 

leg_inward = [(i - 1, j - 1) for (i, j) in leg_inward_ori_index]
leg_outward = [(j, i) for (i, j) in leg_inward]
leg_neighbor = leg_inward + leg_outward



class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.hand_A = self.get_hand_adjacency_matrix(labeling_mode)
        self.leg_A = self.get_leg_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A

    def get_hand_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.hand_A
        if labeling_mode == 'spatial':
            hand_A = tools.get_spatial_graph(hand_num_node, hand_self_link, hand_inward, hand_outward)
        else:
            raise ValueError()
        return hand_A

    def get_leg_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.leg_A
        if labeling_mode == 'spatial':
            leg_A = tools.get_spatial_graph(leg_num_node, leg_self_link, leg_inward, leg_outward)
        else:
            raise ValueError()
        return leg_A