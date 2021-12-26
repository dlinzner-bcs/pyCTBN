import numpy as np
import itertools
from typing import List, NewType
from numpy.core.fromnumeric import shape


State = NewType('State', int)
States = NewType('States', List[State])
Transition = NewType('Transition', tuple[States, float])


class IM:
    def __init__(self, matrix: np.numarray) -> 'IM':
        d = 0
        for col in matrix.T:
            if np.sum(
                    np.delete(col, d)) == -col[d]:
                print(
                    "Warning! Matrix not a proper Intensity Matrix. Will attempt normalization!")
            col[d] = -np.sum(np.delete(col, d))
            assert np.sum(
                np.delete(col, d)) == -col[d], "Logic Error! Matrix not a proper Intensity Matrix!"
            d += 1

        self._im = matrix

    @ classmethod
    def empty_from_dim(self, dim: int) -> 'IM':
        return IM(matrix=np.zeros((dim, dim)))

    @ classmethod
    def random_from_dim(self, dim: int, alpha: float, beta: float) -> 'IM':
        return IM(matrix=np.random.gamma(shape=alpha, scale=1/beta, size=(dim, dim)))


class Node:
    def __init__(self, states: States, parents: List['Node'], children: List['Node']) -> 'Node':
        self._states = states
        self._parents = parents
        self._children = children
        self._cims = None

    @ property
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

    def generate_random_cims(self, alpha: float, beta: float):
        cims = dict()
        for states in itertools.product([p._states for p in self._parents]):
            for state in states:
                dim = int(len(self._states))
                state_map = tuple(state)
                cim = IM.random_from_dim(dim, alpha, beta)
                cims[state_map] = cim
        self._cims = cims


class Graph:
    def __init__(self, nodes: List[Node]):
        self._nodes = nodes

    @property
    def nodes(self):
        return self._nodes


class CTBN(Graph):
    def __init__(self, nodes: List[Node], alpha: float, beta: float):
        super().__init__(nodes)
        [n.generate_random_cims(alpha, beta) for n in self._nodes]

    def sample_transition(self, s0: States) -> Transition:
        pass
