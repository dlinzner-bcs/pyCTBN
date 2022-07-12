from copy import deepcopy
from ctbn.ctbn_model import State, States, Transition, CTBN, CTBNNode, IM
from ctbn.graph import Node
from ctbn.types import Trajectory, Transition, ActiveTransition
from typing import List
from itertools import chain, combinations
from scipy.special import gammaln
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
        #self._cims = None

    @classmethod
    def from_ctbn_node(self, node: CTBNNode):
        return CTBNLearnerNode(node.state, node.states, node.parents,node._alpha,node._beta)

    def reset_stats(self):
        transition_stats = dict()
        exit_time_stats = dict()
        if self._parents is None:
            dim = len(self._states)
            transition_stats[None] = np.zeros((dim, dim))
            exit_time_stats[None] = np.zeros((dim,))
        else:
            for states in self.all_state_combinations():
                dim = len(self._states)
                transition_stats[tuple(states)] = np.zeros((dim, dim))
                exit_time_stats[tuple(states)] = np.zeros((dim,))

        self._transition_stats = transition_stats
        self._exit_time_stats = exit_time_stats

    def update_stats(self, transition: Transition):
        if self.nid == transition._node_id:
            node = self
            if node._parents is None:
                p_state = None
            else:
                p_state = tuple([transition._s_init[n.nid]
                                for n in node._parents])
            s0 = transition._s_init[node.nid]
            s1 = transition._s_final[node.nid]
            t_stat = node._transition_stats[p_state]
            t_stat[s0, s1] += 1
        else:
            s0 = transition._s_init[self.nid]
            if self._parents is None:
                p_state = None
            else:
                p_state = tuple([transition._s_init[n_p.nid]
                                for n_p in self._parents])
            e_stat =  self._exit_time_stats[p_state]
            e_stat[s0] += transition._exit_time

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

    def sample_cims(self):
        cims = dict()
        for key in self._transition_stats.keys():
            t_stat = self._transition_stats[key]
            e_stat = self._exit_time_stats[key]
            cim = IM.empty_from_dim(len(self._states))
            for s in self._states:
                for s_ in self._states:
                    if s != s_:
                        cim.set(s, s_, np.random.gamma(
                            shape=t_stat[s, s_], scale=1/e_stat[s]))
            cims[key] = cim
        self._cims = cims

    def llikelihood(self):
        llh = 0
        for key in self._transition_stats.keys():
            t_stat = self._transition_stats[key]
            e_stat = self._exit_time_stats[key]
            for s in self._states:
                for s_ in self._states:
                    if s != s_:
                        llh += t_stat[s, s_]*np.log(self._cims[key].im[s, s_]) - \
                            e_stat[s]*self._cims[key].im[s, s]
        return llh

    def structure_score(self):
        llh = 0
        for key in self._transition_stats.keys():
            t_stat = self._transition_stats[key]
            e_stat = self._exit_time_stats[key]
            for s in self._states:
                for s_ in self._states:
                    if s != s_:
                        llh += gammaln(t_stat[s, s_]) - \
                            t_stat[s, s_]*np.log(e_stat[s])
        return llh

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

    def average_stats_active(self, transition: ActiveTransition, num_samples):
        node = self.node_by_id(transition._node_id)
        if node._parents is None:
            p_state = None
        else:
            p_state = tuple([transition._s_init[n.nid]
                             for n in node._parents])
        s0 = transition._s_init[node.nid]
        s1 = transition._s_final[node.nid]
        t_stat = node._transition_stats[p_state]
        t_stat[s0, s1] += 1/num_samples

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
                e_stat[s0] += transition._exit_time/num_samples

    def estimate_cims(self):
        [n.estimate_cims() for n in self.nodes]

    def sample_cims(self):
        [n.sample_cims() for n in self.nodes]

    def reset_stats(self):
        [n.reset_stats() for n in self.nodes]

    def llikelihood(self):
        return np.sum([n.llikelihood() for n in self.nodes])

    def estimate_expected_statistics(self, intervention, num_samples):
        simulator = deepcopy(self)
        simulator.reset_stats()
        for k in range(0, num_samples):
            simulator.average_stats_active(
                self.intervention(intervention).transition(), num_samples)
        return simulator

    def score_parents_of_node(self,node: CTBNLearnerNode,data: Trajectory):
        ctbn_learner_ = deepcopy(self)
        node_ = ctbn_learner_.node_by_id(node.nid)
        scores ={}
        for parent_candidate in self.node_powerset(3):
            ctbn_learner_.reset_stats()
            node_.set_parents(parent_candidate)
            node_.generate_random_cims(node_._alpha,node_._beta)
            node_.reset_stats()
            ctbn_learner_ = CTBNLearner(ctbn_learner_.nodes)
            for trans in data:
                node_.update_stats(trans)
                #ctbn_learner_.update_stats(trans)
            ctbn_learner_.estimate_cims()
            scores[tuple([n.nid for n in parent_candidate])]  = node_.structure_score()
        return scores
    
    def node_powerset(self,k):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(self.nodes)
        return chain.from_iterable(combinations(s, r) for r in range(0,k))




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
