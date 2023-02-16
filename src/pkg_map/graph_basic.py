import math
import random

import networkx as nx

from typing import List, Dict, Any


class NetGraph(nx.Graph):
    """Interactive interface with networkx library."""
    def __init__(self, node_dict: Dict[Any, tuple], edge_list: List[tuple]):
        """
        Args:
            node_dict: {node_id: (x, y)}
            edge_list: [(node_id1, node_id2), ...]
        """
        super().__init__()
        self._position_key = 'position'
        for node_id in node_dict:
            self.add_node(node_id, **{self._position_key: node_dict[node_id]})
        self.add_edges_from(edge_list)

    def set_distance_weight(self):
        def euclidean_distance(graph: nx.Graph, source, target):
            x1, y1 = graph.nodes[source][self._position_key]
            x2, y2 = graph.nodes[target][self._position_key]
            return math.sqrt((x1-x2)**2 + (y1-y2)**2) 
        for e in self.edges():
            self[e[0]][e[1]]['weight'] = euclidean_distance(self, e[0], e[1])

    def get_node_coord(self, node_id) -> tuple:
        x = self.nodes[node_id][self._position_key][0]
        y = self.nodes[node_id][self._position_key][1]
        return x, y

    def return_given_nodelist(self, graph_node_ids:list) -> List[tuple]:
        return [self.get_node_coord(id) for id in graph_node_ids]

    def return_random_nodelist(self, start_node_id, num_traversed_nodes:int) -> List[tuple]:
        """Return random GeometricGraphNode without repeat nodes
        """
        node_ids = [start_node_id]
        nodelist = [self.get_node_coord(start_node_id)]
        for _ in range(num_traversed_nodes):
            connected_node_ids = list(self.adj[node_ids[-1]])
            connected_node_ids = [x for x in connected_node_ids if x not in node_ids]
            if not connected_node_ids:
                return nodelist
            next_id = random.choice(connected_node_ids) # NOTE: Change this to get desired path pattern
            node_ids.append(next_id)
            nodelist.append(self.get_node_coord(next_id))
        return nodelist



