import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools


num_node = 37
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (3, 2), (4, 3), (5, 4), (6, 2), (7, 6),
                     (8, 7), (9, 2), (10, 9), (11, 10), (12, 11), (13, 9),
                     (14, 13), (15, 14), (16, 1), (17, 1), (18, 16), (19, 17),
                     (20, 15), (21, 20), (22, 15), (23, 12), (24, 23), (25, 12),
                     (26,4), 
                     (27, 26), (28, 27), (29, 26), (30, 26), (31, 26),
                     (32,7),
                     (33,32), (34, 32), (35, 32), (36, 32), (37, 32)]


inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


hand_num_node = 48
hand_self_link = [(i, i) for i in range(hand_num_node)]
hand_inward_ori_index = [(2,1),(3,2),(7,3),
                        (8,7),(9,8),(10,9),(11,10),
                        (12,7),(13,12),(14,13),(15,14),
                        (16,7),(17,16),(18,17),(19,18),
                        (20,7),(21,20),(22,21),(23,22),
                        (24,7),(25,24),(26,25),(27,26),
                        (5,4),(6,5),(28,6),
                        (29,28),(30,29),(31,30),(32,31),
                        (33,28),(34,33),(35,34),(36,35),
                        (37,28),(38,37),(39,38),(40,39),
                        (41,28),(42,41),(43,42),(44,43),
                        (45,28),(46,45),(47,46),(48,47),
                        (4,1)] 

hand_inward = [(i - 1, j - 1) for (i, j) in hand_inward_ori_index]
hand_outward = [(j, i) for (i, j) in hand_inward]
hand_neighbor = hand_inward + hand_outward

leg_num_node = 13
leg_self_link = [(i, i) for i in range(leg_num_node)]
leg_inward_ori_index = [(2,1), (3,2), (4,3), (5,4), (6,4), (7,6),
                        (8,1), (9,8), (10,9), (11,10), (12,10), (13, 12)] 

leg_inward = [(i - 1, j - 1) for (i, j) in leg_inward_ori_index]
leg_outward = [(j, i) for (i, j) in leg_inward]
leg_neighbor = leg_inward + leg_outward



class Graph:
    def __init__(self, labeling_mode='spatial', scale=1, **kwargs):
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
