from ctbn.ctbn_model import State, States, Transition, CTBN, CTBNNode, IM
from ctbn.graph import Node
from ctbn.types import Transition
from ctbn.active_sampler import ActiveSampler, ActiveTransition
from typing import List
import numpy as np
import unittest


class CTBNLearnerNode(CTBNNode):
    def __init__(self, state: State, states: States, parents: List['Node'], alpha=1.0, beta=1.0) -> 'Node':
        super().__init__(state, states, parents)
        transition_stats = dict()
        exit_time_stats = dict()
        if self._parents is None:
            dim = len(self._states)
            transition_stats[None] = np.ones((dim, dim))*alpha
            exit_time_stats[None] = np.ones((dim,))*beta
        else:
            for states in self.all_state_combinations():
                dim = len(self._states)
                transition_stats[tuple(states)] = np.ones((dim, dim))*alpha
                exit_time_stats[tuple(states)] = np.ones((dim,))*beta

        self._transition_stats = transition_stats
        self._exit_time_stats = exit_time_stats
        self._cims = None

    @classmethod
    def from_ctbn_node(self, node: CTBNNode, alpha=1.0, beta=1.0):
        return CTBNLearnerNode(node.state, node.states, node.parents)

    def estimate_cims(self):
        cims = dict()
        for key in self._transition_stats.keys():
            t_stat = self._transition_stats[key]
            e_stat = self._exit_time_stats[key]
            cim = IM.empty_from_dim(len(self._states))
            for s in self._states:
                for s_ in self._states:
                    if s != s_:
                        cim.set(s, s_, t_stat[s, s_]/e_stat[s])
            cims[key] = cim
        self._cims = cims


class CTBNLearner(CTBN):
    def __init__(self, nodes: List[CTBNLearnerNode]):
        super().__init__(nodes)

    def update_stats(self, transiton):
        if type(transiton) is ActiveTransition:
            self.update_stats_active(transiton)
        if type(transiton) is Transition:
            self.update_stats_passive(transiton)
        else:
            pass

    def update_stats_passive(self, transition: Transition):
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
            s0 = transition._s_init[n.nid]
            if n._parents is None:
                p_state = None
            else:
                p_state = tuple([transition._s_init[n_p.nid]
                                for n_p in n._parents])
            e_stat = n._exit_time_stats[p_state]
            e_stat[s0] += transition._exit_time

    def update_stats_active(self, transition: ActiveTransition):
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

        if transition._intervention is None:
            intervened_nodes = []
        else:
            intervened_nodes = [intv[0] for intv in transition._intervention]
        for n in self.nodes:
            if n.nid in intervened_nodes:
                pass
            else:
                s0 = transition._s_init[n.nid]
                if n._parents is None:
                    p_state = None
                else:
                    p_state = tuple([transition._s_init[n_p.nid]
                                    for n_p in n._parents])
                e_stat = n._exit_time_stats[p_state]
                e_stat[s0] += transition._exit_time

    def estimate_cims(self):
        [n.estimate_cims() for n in self.nodes]


class TestLearner(unittest.TestCase):

    def test_update_statistics(self):

        states = States(list([State(0), State(1)]))
        nodes = []
        nodes.append(CTBNLearnerNode(state=State(0), states=states,
                                     parents=None))
        nodes.append(CTBNLearnerNode(state=State(0), states=states,
                                     parents=[nodes[0]]))
        nodes.append(CTBNLearnerNode(state=State(0), states=states,
                                     parents=[nodes[0], nodes[1]]))

        init_states = States([State(0), State(0), State(0)])
        ctbn_learner = CTBNLearner(nodes)

        transition = Transition(2, init_states, States(
            [State(0), State(0), State(1)]), 1.03)

        ctbn_learner.update_stats(transition)

        assert(abs(nodes[0]._exit_time_stats[None][0] - 2.03) < 10**-9)
        assert(abs(nodes[0]._exit_time_stats[None][1] - 1.0) < 10**-9)
        assert(nodes[0]._transition_stats[None][0, 1] == 1.0)
        assert(nodes[0]._transition_stats[None][1, 0] == 1.0)

        curr_state = tuple(States([State(0)]))
        assert(abs(nodes[1]._exit_time_stats[curr_state][0] - 2.03) < 10**-9)
        assert(nodes[1]._transition_stats[curr_state][0, 1] == 1.0)
        assert(nodes[1]._transition_stats[curr_state][1, 0] == 1.0)
        for states in nodes[1].all_state_combinations():
            if states != curr_state:
                assert(
                    abs(nodes[1]._exit_time_stats[states][1] - 1.0) < 10**-9)
                assert(nodes[1]._transition_stats[states][0, 1] == 1.0)
                assert(nodes[1]._transition_stats[states][1, 0] == 1.0)

        curr_state = tuple(States([State(0), State(0)]))
        assert(abs(nodes[2]._exit_time_stats[curr_state][0] - 2.03) < 10**-9)
        assert(nodes[2]._transition_stats[curr_state][0, 1] == 2.0)
        assert(nodes[2]._transition_stats[curr_state][1, 0] == 1.0)
        for states in nodes[2].all_state_combinations():
            if states != curr_state:
                assert(
                    abs(nodes[2]._exit_time_stats[states][1] - 1.0) < 10**-9)
                assert(nodes[2]._transition_stats[states][0, 1] == 1.0)
                assert(nodes[2]._transition_stats[states][1, 0] == 1.0)
