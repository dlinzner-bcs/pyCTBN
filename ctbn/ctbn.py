import numpy as np
import itertools
from typing import List, NewType

from numpy.matrixlib import matrix

State = NewType('State', int)
States = NewType('States', List[State])


class IM:
    def __init__(self, matrix: np.numarray) -> 'IM':
        d = 0
        for row in matrix.T:
            assert np.sum(
                row[:-d]) == -row[d], "Warning! Matrix not a proper IM. Will attempt normalization!"
            row[d] = np.sum(row[:-d])
            d += 1

        self._im = matrix

    @classmethod
    def empty_from_dim(self, dim: int) -> 'IM':
        return IM(matrix=np.zeros((dim, dim)))

    @classmethod
    def random_from_dim(self, dim: int, alpha: float, beta: float) -> 'IM':
        return IM(matrix=np.zeros((dim, dim)))


class Node:
    def __init__(self, states: States, parents: List['Node'], children: List['Node']) -> 'Node':
        self._states = states
        self._parents = parents
        self._children = children
        self._cims = None

    @property
    def cims(self):
        return self._cims

    def generate_empty_cims(self):
        cims = dict()
        for states in itertools.product([p._states for p in self._parents]):
            for state in states:
                dim = int(len(self._states))
                state_map = tuple(state)
                cim = IM.empty_from_dim(dim)
                cims[state_map] = cim
        self._cims = cims

     def generate_random_cims(self,alpha:float,beta:float):
        cims = dict()
        for states in itertools.product([p._states for p in self._parents]):
            for state in states:
                dim = int(len(self._states))
                state_map = tuple(state)
                cim = IM.random_from_dim(dim,alpha,beta)
                cims[state_map] = cim
        self._cims = cims


class Graph:
    def __init__(self, nodes: List[Node]):
        self._nodes = nodes

    @property
    def nodes(self):
        return self._nodes


class CTBN(Graph):
    def __init__(self, nodes, state_spaces, edges):
        super().__init__(nodes, state_spaces, edges)
