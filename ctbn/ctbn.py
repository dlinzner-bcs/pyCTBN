import numpy as np
import itertools
from typing import List, NewType, Optional
from numpy.core.fromnumeric import shape
from copy import copy
import pprint
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

State = NewType('State', int)
States = NewType('States', tuple[State])


class Transition:
    def __init__(self, node_id: int,  s0: States, s1: States, tau: float) -> None:
        self._node_id = node_id
        self._s_init = s0
        self._s_final = s1
        self._exit_time = tau

    def __repr__(self):
        return "Transition of Node " + repr(self._node_id) + " with Initial State:" + repr(self._s_init) + ";End State:" + repr(self._s_final) + ";Exit Time:" + repr(self._exit_time)


class IM:
    def __init__(self, matrix: np.numarray) -> 'IM':
        d = 0
        for col in matrix:
            if np.sum(
                    np.delete(col, d)) == -col[d]:
                logging.warning(
                    "Matrix not a proper Intensity Matrix. Will attempt normalization!")
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

    def set(self, x, y, val) -> None:
        self._im[x, y] = val
        self._im[x, x] = -np.sum(np.delete(self._im[x, :], x))


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


class CTBNNode(Node):
    def __init__(self, state: State, states: States, parents: List['Node']) -> 'Node':
        super().__init__(states, parents)
        self._state = state
        self._cims = None
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
            for states in itertools.product(*[p._states for p in self._parents]):
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
            for states in itertools.product(*[p._states for p in self._parents]):
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


class CTBN(Graph):
    def __init__(self, nodes: List[CTBNNode]):
        super().__init__(nodes)

    @ classmethod
    def with_random_cims(self, nodes: List[CTBNNode], alpha, beta):
        [n.generate_random_cims(alpha, beta) for n in nodes]
        logging.info("Initialiting Random Conditional Intensity Matrices:")
        [logging.debug(pprint.pformat(n.cims)) for n in nodes]
        return CTBN(nodes)

    @ property
    def state(self):
        return States([n.state for n in self._nodes])

    def active_node(self) -> 'Node':
        rates = [0 if n.exit_rate is None else n.exit_rate for n in self._nodes]
        cum_prob = np.cumsum(rates) / np.sum(rates)
        return self._nodes[np.argmax(np.random.uniform() <= cum_prob)]

    def exit_time(self):
        rates = [0 if n.exit_rate is None else n.exit_rate for n in self._nodes]
        return np.random.exponential(1/np.sum(rates))

    def transition(self) -> Transition:
        node = self.active_node()
        tau = self.exit_time()
        s_0 = self.state
        node.next_state()
        s_1 = self.state
        return Transition(node.nid, s_0, s_1, tau)


class Trajectory:
    def __init__(self) -> None:
        self._transitions: List[Transition]
        self._transitions = list()

    def append(self, transition: Transition):
        self._transitions.append(transition)

    def __repr__(self):
        representation = pprint.pformat(
            self._transitions)
        return representation

    def __iter__(self):
        pass


class CTBNLearnerNode(CTBNNode):
    def __init__(self, state: State, states: States, parents: List['Node'], alpha: float, beta: float) -> 'Node':
        super().__init__(state, states, parents)
        transition_stats = dict()
        exit_time_stats = dict()
        if self._parents is None:
            dim = len(self._states)
            transition_stats[None] = np.zeros((dim, dim))
            exit_time_stats[None] = np.zeros((dim,))
        else:
            for states in itertools.product(*[p._states for p in self._parents]):
                dim = len(self._states)
                transition_stats[tuple(states)] = np.ones((dim, dim))*alpha
                exit_time_stats[tuple(states)] = np.ones((dim,))*beta

        self._transition_stats = transition_stats
        self._exit_time_stats = exit_time_stats
        self._cims = None

    def estimate_cims(self):
        cims = dict()
        for key in self._transition_stats.keys():
            t_stat = self._transition_stats[key]
            e_stat = self._exit_time_stats[key]
            cim = IM.empty_from_dim(len(self._states))
            for s in self._states:
                for s_ in self._states:
                    cim.set(s, s_, t_stat[s, s_]/e_stat[s])
            cims[key] = cim
        self._cims = cims


class CTBNLearner(Graph):
    def __init__(self, nodes: List[CTBNLearnerNode]):
        super().__init__(nodes)

    def update_stats(self, transition: Transition):
        node = self.node_by_id(transition._node_id)
        if node._parents is None:
            p_state = None
        else:
            p_state = tuple([transition._s_init[n.nid]
                             for n in node._parents])
        s0 = transition._s_init[node.nid]
        s1 = transition._s_final[node.nid]
        t_stat = node._transition_stats[p_state]
        t_stat[s0, s1] += 1
        for n in self.nodes:
            if n != node:
                s0 = transition._s_init[n.nid]
                if n._parents is None:
                    p_state = None
                else:
                    p_state = tuple([transition._s_init[n_p.nid]
                                    for n_p in n._parents])
                e_stat = n._exit_time_stats[p_state]
                e_stat[s0] += transition._exit_time
