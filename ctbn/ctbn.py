import numpy as np
import itertools
from typing import List, NewType, Optional
from numpy.core.fromnumeric import shape


State = NewType('State', int)
States = NewType('States', List[State])


class Transition:
    def __init__(self, s0: State, s1: State, tau: float) -> None:
        self._s_init = s0
        self._s_final = s1
        self._exit_time = tau


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

    def __repr__(self) -> str:
        return np.array2string(self._im)


class Node:
    def __init__(self, state: State, states: States, parents: List['Node'], children: List['Node']) -> 'Node':
        self._states = states
        self._parents = parents
        self._children = children
        self._cims = None
        self._state = state
        self._cim: Optional[IM]
        self._cim = None

    @ property
    def cims(self):
        return self._cims

    @property
    def state(self):
        return self._state

    def generate_empty_cims(self):
        cims = dict()
        if isinstance(self._parents, type(None)):
            dim = int(len(self._states))
            cim = IM.empty_from_dim(dim)
            cims[None] = cim
        else:
            for states in itertools.product([p._states for p in self._parents]):
                for state in states:
                    dim = int(len(self._states))
                    state_map = tuple(state)
                    cim = IM.empty_from_dim(dim)
                    cims[state_map] = cim
        self._cims = cims

    def generate_random_cims(self, alpha: float, beta: float):
        cims = dict()
        if self._parents is None:
            dim = int(len(self._states))
            cim = IM.random_from_dim(dim, alpha, beta)
            cims[None] = cim
        else:
            for states in itertools.product([p._states for p in self._parents]):
                for state in states:
                    dim = int(len(self._states))
                    state_map = tuple(state)
                    cim = IM.random_from_dim(dim, alpha, beta)
                    cims[state_map] = cim
        self._cims = cims

    @property
    def cim(self) -> IM:
        if self._parents is None:
            return self._cims[None]
        else:
            parent_state = tuple([p._states for p in self._parents][0])
            return self._cims[parent_state]

    @property
    def exit_rate(self) -> Optional[float]:
        if self.cim is None:
            return None
        else:
            return -self.cim._im[self._state, self._state]

    @property
    def transition_rates(self) -> Optional[np.numarray]:
        if self.cim is None:
            return None
        else:
            transition_rates = self.cim._im[self._state, :]
            transition_rates[self._state] = 0
            return transition_rates

    def next_state(self):
        transition_rates = self.transition_rates
        if transition_rates is None:
            pass
        else:
            cum_prob = np.cumsum(self.transition_rates) / \
                np.sum(self.transition_rates)
            self._state = State(np.argmax(np.random.uniform() <= cum_prob))

    def exit_time(self):
        return np.random.exponential(self.exit_rate)


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
        [n.set_cim() for n in self._nodes]

    def active_node(self) -> 'Node':
        rates = [0 if np.exit_rate is None else n.exit_rate for n in self._nodes]
        cum_prob = np.cumsum(rates) / \
            np.sum(self.transition_rates)
        return self._nodes[np.argmax(np.random.uniform() <= cum_prob)]

    def transition(self) -> Transition:
        node = self.active_node()
        tau = node.exit_time()
        s_0 = node.state
        node.next_state()
        s_1 = node.state
        return Transition(s_0, s_1, tau)


class Trajectory:
    def __init__(self) -> None:
        self._transitions: List[Transition]
        self._transitions = list()

    def append(self):
        self._transitions.append(Transition)
