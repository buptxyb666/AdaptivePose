import sys
from .tools import get_spatial_graph

num_node = 7
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(0, 1), (1, 2), (1, 3), (1, 4), (4, 5), (4, 6)]
inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self):
        self.A = self.get_adjacency_matrix()
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self):
        A = get_spatial_graph(num_node, self_link, inward, outward)
        return A

