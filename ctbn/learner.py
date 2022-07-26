from copy import deepcopy
from itertools import chain, combinations
from typing import List, Optional
import numpy as np
from scipy.special import gammaln
from ctbn.ctbn_model import CTBN, IM, CTBNNode, State, States, Transition
from ctbn.graph import Node
from ctbn.types import ActiveTransition, Trajectory, Transition


class CTBNLearnerNode(CTBNNode):
    def __init__(self, state: State, states: States, parents: List['Node'], alpha=1.0, beta=1.0, name=Optional[str]) -> 'Node':
        super().__init__(state, states, parents, alpha, beta)
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
        if name:
            self._name = name
        else:
            self._name = self._nid
        #self._cims = None

    @classmethod
    def from_ctbn_node(self, node: CTBNNode):
        return CTBNLearnerNode(node.state, node.states, node.parents, node._alpha, node._beta)

    def reset_stats(self):
        transition_stats = dict()
        exit_time_stats = dict()
        if self._parents is None:
            dim = len(self._states)
            transition_stats[None] = np.ones((dim, dim))*self._alpha
            exit_time_stats[None] = np.ones((dim,))*self._beta
        else:
            for states in self.all_state_combinations():
                dim = len(self._states)
                transition_stats[tuple(states)] = np.ones(
                    (dim, dim))*self._alpha
                exit_time_stats[tuple(states)] = np.ones((dim,))*self._beta

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

        s0 = transition._s_init[self.nid]
        if self._parents is None:
            p_state = None
        else:
            p_state = tuple([transition._s_init[n_p.nid]
                            for n_p in self._parents])
        e_stat = self._exit_time_stats[p_state]
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
                            t_stat[s, s_]*np.log(e_stat[s]) -\
                            gammaln(self._alpha) +\
                            self._alpha*np.log(self._beta)
            if llh == np.inf:
                llh = -np.inf
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

    def score_parent_candidate_of_node(self, node: CTBNLearnerNode, data: Trajectory, parent_candidate: List[CTBNLearnerNode]):
        ctbn_learner_ = deepcopy(self)
        node_ = ctbn_learner_.node_by_id(node.nid)
        ctbn_learner_.reset_stats()
        node_.set_parents(parent_candidate)
        node_.generate_random_cims(node_._alpha, node_._beta)
        node_.reset_stats()
        ctbn_learner_ = CTBNLearner(ctbn_learner_.nodes)
        for trans in data:
            node_.update_stats(trans)
            # ctbn_learner_.update_stats(trans)
        node_.estimate_cims()
        return node_.structure_score()

    def score_parents_of_node(self, node: CTBNLearnerNode, data: Trajectory, max_num_parents: int):
        ctbn_learner_ = deepcopy(self)
        node_ = ctbn_learner_.node_by_id(node.nid)
        scores = {}

        def score_parent_of_node(parent_candidate: List[CTBNLearnerNode]):
            return self.score_parent_candidate_of_node(node_, data, parent_candidate)

        for p in self.node_powerset(max_num_parents+1):
            scores[tuple([n._nid for n in p])] = score_parent_of_node(p)

        return scores

    def score_parents_of_node_greedy(self, node: CTBNLearnerNode, data: Trajectory, max_num_parents: int):
        ctbn_learner_ = deepcopy(self)
        node_ = ctbn_learner_.node_by_id(node.nid)
        scores = {}

        def score_parent_of_node(parent_candidate: List[CTBNLearnerNode]):
            return self.score_parent_candidate_of_node(node_, data, parent_candidate)

        parents = list(self.node_powerset(2))
        for i in range(0, len(parents)):
            score = score_parent_of_node(parents[i])
            if np.isnan(score) or np.isinf(abs(score)):
                None
            else:
                scores[i] = score

        ranked_scores = dict(
            sorted(scores.items(), key=lambda x: x[1], reverse=True))
        best_candidates = []
        for p in list(ranked_scores.keys()):
            if len(parents[p]) != 0:
                best_candidates.append(parents[p][0])

        scores = {}
        for p in self.node_list_powerset(max_num_parents+1, best_candidates[0:10]):
            scores[tuple([n._nid for n in p])] = score_parent_of_node(p)

        return scores

    def learn_parents_of_node(self, node: CTBNLearnerNode, data: Trajectory, max_num_parents: int):
        scores = self.score_parents_of_node(node, data, max_num_parents)
        return (max(scores, key=scores.get), scores)

    def learn_parents_of_node_greedy(self, node: CTBNLearnerNode, data: Trajectory, max_num_parents: int):
        scores = self.score_parents_of_node_greedy(node, data, max_num_parents)
        return (max(scores, key=scores.get), scores)

    def node_powerset(self, k):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(self.nodes)
        return chain.from_iterable(combinations(s, r) for r in range(0, k))

    def node_list_powerset(self, k, s):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        return chain.from_iterable(combinations(s, r) for r in range(0, k))

    def learn_k_hop_network_of_node(self, node_origin: CTBNLearnerNode, data: Trajectory, max_num_parents: int, number_of_hops: int):
        node_list = [node_origin]
        k = 0
        sources = []
        targets = []
        while len(node_list) > 0:
            if k <= number_of_hops:
                node = node_list[0]
                node_list.pop(0)
                parents, _ = self.learn_parents_of_node(
                    node, data, max_num_parents)
                parent_nodes = [self.node_by_id(id) for id in parents]
                node.set_parents(parent_nodes)
                node.reset_stats()
                [node_list.append(p) for p in parent_nodes]
                parent_string = "".join(
                    ["["+p._name+"]"+", " for p in parent_nodes])
                print("Parents of [%s] are %s" % (node._name, parent_string))

                [targets.append(node._name)
                 for i in range(0, len(parent_nodes))]
                [sources.append(p._name) for p in parent_nodes]
                k += 1
            else:
                break
        return (sources, targets)

    def learn_k_hop_network_of_node_greedy(self, node_origin: CTBNLearnerNode, data: Trajectory, max_num_parents: int, number_of_hops: int):
        node_list = [node_origin]
        k = 0
        sources = []
        targets = []
        while len(node_list) > 0:
            if k <= number_of_hops:
                node = node_list[0]
                node_list.pop(0)
                parents, _ = self.learn_parents_of_node_greedy(
                    node, data, max_num_parents)
                parent_nodes = [self.node_by_id(id) for id in parents]
                node.set_parents(parent_nodes)
                node.reset_stats()
                [node_list.append(p) for p in parent_nodes]
                parent_string = "".join(
                    ["["+p._name+"]"+", " for p in parent_nodes])
                print("Parents of [%s] are %s" % (node._name, parent_string))

                [targets.append(node._name)
                 for i in range(0, len(parent_nodes))]
                [sources.append(p._name) for p in parent_nodes]
                k += 1
            else:
                break
        return (sources, targets)
