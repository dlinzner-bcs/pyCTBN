from ctbn.types import States
import itertools
from typing import List, NewType, Optional


class Node:
    def __init__(self, states: States, parents: List['Node']) -> 'Node':
        self._states = states
        self._parents = parents
        self._nid: Optional[int]
        self._nid = None

    @property
    def nid(self):
        return self._nid

    @nid.setter
    def nid(self, node_id):
        self._nid = node_id

    @nid.deleter
    def nid(self):
        self._nid = None

    @property
    def parents(self):
        return self._parents

    @property
    def states(self):
        return self._states

    def all_state_combinations(self):
        if self.parents is None:
            return [None]
        else:
            return itertools.product(*[p.states for p in self.parents])


class Graph:
    def __init__(self, nodes: List[Node]):
        self._nodes = nodes
        for n, n_id in zip(nodes, range(0, len(nodes))):
            n.nid = n_id

    @property
    def nodes(self):
        return self._nodes

    def node_by_id(self, nid) -> 'Node':
        return [n for n in self._nodes if n._nid == nid][0]
