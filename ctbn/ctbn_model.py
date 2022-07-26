import logging
import pprint
from copy import copy, deepcopy
from typing import List, Optional
import numpy as np
from ctbn.graph import Graph, Node
from ctbn.types import State, States, Transition

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class IM:
    def __init__(self, matrix: np.numarray) -> 'IM':
        d = 0
        for col in matrix:
            col[d] = -np.sum(np.delete(col, d))
            assert np.sum(
                np.delete(col, d)) == -col[d], "Logic Error! Matrix not a proper Intensity Matrix!"
            d += 1

        self._im = matrix

    @ classmethod
    def empty_from_dim(self, dim: int) -> 'IM':
        return IM(matrix=np.zeros((dim, dim)))

    @ classmethod
    def random_from_dim(self, dim: int, alpha=1.0, beta=1.0) -> 'IM':
        return IM(matrix=np.random.gamma(shape=alpha, scale=1/beta, size=(dim, dim)))

    def __repr__(self) -> str:
        return np.array2string(self._im)

    def set(self, x, y, val) -> None:
        self._im[x, y] = val
        self._im[x, x] = -np.sum(np.delete(self._im[x, :], x))

    @property
    def im(self):
        return self._im


class CTBNNode(Node):
    def __init__(self, state: State, states: States, parents: List['Node'], name: str, alpha=1.0, beta=1.0) -> 'Node':
        super().__init__(states, parents)
        self._state = state
        self._cims = None
        self._cim: Optional[IM]
        self._cim = None
        self._alpha = alpha
        self._beta = beta
        self._name = name

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
            for states in self.all_state_combinations():
                dim = len(self._states)
                cim = IM.empty_from_dim(dim)
                cims[tuple(states)] = cim
        self._cims = cims

    def generate_random_cims(self, alpha: float, beta: float):
        cims = dict()
        if self._parents is None:
            dim = len(self._states)
            cim = IM.random_from_dim(dim, alpha, beta)
            cims[None] = cim
        else:
            for states in self.all_state_combinations():
                dim = len(self._states)
                cim = IM.random_from_dim(dim, alpha, beta)
                cims[tuple(states)] = cim
        self._cims = cims

    @property
    def cim(self) -> IM:
        if self._parents is None:
            return self._cims[None]
        else:
            parent_state = tuple([p.state for p in self._parents])
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
            transition_rates = copy(self.cim._im[self._state, :])
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

    def intervention(self, state: State):
        node = deepcopy(self)
        node._state = state
        for states in node.all_state_combinations():
            node.cims[states]._im = node.cims[states]._im * 0
        return node


class CTBN(Graph):
    def __init__(self, nodes: List[CTBNNode]):
        super().__init__(nodes)

    @ classmethod
    def with_random_cims(self, nodes: List[CTBNNode]):
        [n.generate_random_cims(n._alpha, n._beta) for n in nodes]
        logging.info("Initialiting Random Conditional Intensity Matrices:")
        [logging.debug(pprint.pformat(n.cims)) for n in nodes]
        return CTBN(nodes)

    def print_cims(self) -> None:
        logging.info("Conditional Intensity Matrices:")
        [logging.debug(pprint.pformat(n.cims)) for n in self.nodes]
        return None

    @ property
    def state(self):
        return States([n.state for n in self._nodes])

    def active_node(self) -> 'Node':
        rates = [0 if n.exit_rate is None else n.exit_rate for n in self._nodes]
        cum_prob = np.cumsum(rates) / np.sum(rates)
        return self._nodes[np.argmax(np.random.uniform() <= cum_prob)]

    def exit_time(self):
        rates = [0 if n.exit_rate is None else n.exit_rate for n in self._nodes]
        return np.random.exponential(1/abs(np.sum(rates)))

    def transition(self) -> Transition:
        node = self.active_node()
        tau = self.exit_time()
        s_0 = self.state
        node.next_state()
        s_1 = self.state
        return Transition(node.nid, s_0, s_1, tau)

    def randomize_states(self):
        for n in self.nodes:
            n._state = np.random.choice(n._states)

    def intervention(self, node_id: int, state: State):
        ctbn = deepcopy(self)
        node = ctbn.node_by_id(node_id)
        node.intervention(state)
        return ctbn
