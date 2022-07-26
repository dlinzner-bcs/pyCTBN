from copy import deepcopy
import numpy as np
from scipy.special import digamma
from ctbn.ctbn_model import CTBN
from ctbn.learner import CTBNLearner
from ctbn.types import Transition, State, States, Intervention, ActiveTransition
from typing import NewType
from enum import Enum


class SamplingStrategy(Enum):
    RANDOM = 1
    EIG = 2
    BHC = 3
    VBHC = 4


class ActiveSampler():
    def __init__(self, simulator: CTBN, belief: CTBNLearner, strategy: SamplingStrategy, max_elements=None) -> None:
        self._simulator = simulator
        self._belief = belief
        self._strategy = strategy
        self._max_elements = max_elements

    def sample(self):
        if self._strategy == SamplingStrategy.RANDOM:
            all_combs = list(self._simulator.all_combos(self._max_elements))
            intervention = np.random.choice(all_combs)
            return ActiveTransition.from_transition(self._simulator.intervention(
                intervention=intervention).transition(), intervention=intervention)
        if self._strategy == SamplingStrategy.EIG:
            intervention = self.eig_aquisition()
            return ActiveTransition.from_transition(self._simulator.intervention(
                intervention=intervention).transition(), intervention=intervention)
        if self._strategy == SamplingStrategy.BHC:
            intervention = self.bhc_aquisition()
            print(intervention)
            return ActiveTransition.from_transition(self._simulator.intervention(
                intervention=intervention).transition(), intervention=intervention)
        else:
            pass

    def eig_aquisition(self):
        all_combs = self._simulator.all_combos(self._max_elements)
        eig_vals = dict()
        for comb in all_combs:
            eig = 0
            for _ in range(0, 3):
                prior = deepcopy(self._belief)
                prior.sample_cims()
                trans = prior.intervention(comb).transition()
                posterior = deepcopy(prior)
                posterior.update_stats(trans)
                post_llh = posterior.llikelihood()
                post_llhs = 0
                for m in range(0, 3):
                    prior.sample_cims()
                    posterior = deepcopy(prior)
                    posterior.update_stats(trans)
                    post_llhs += np.exp(posterior.llikelihood())
                eig += post_llh-np.log(np.sum(post_llhs))
            eig_vals[comb] = eig/100
        return max(eig_vals,  key=eig_vals.get)

    def bhc_job(self, args):
        intervention = args[0]
        num_samples = args[1]
        num_samples_stats = args[2]
        bhc = 0
        for _ in range(0, num_samples):
            self._belief.sample_cims()
            bhc += self.bhc(intervention, num_samples_stats)/num_samples
        return [intervention, bhc]

    def bhc(self, intervention, num_samples):
        estimate = self._belief.estimate_expected_statistics(
            intervention=intervention, num_samples=num_samples)
        bhc_k = 0
        for n in estimate.nodes:
            for states in n.all_state_combinations():
                for s in n._states:
                    for s_ in n._states:
                        if s != s_:
                            belief_node = self._belief.node_by_id(
                                n.nid)
                            cim = belief_node.cims[states].im
                            bhc_k += n._transition_stats[states][s, s_]*(
                                np.log(
                                    cim[s, s_])
                                - digamma(belief_node._transition_stats[states][s, s_])
                                + np.log(belief_node._exit_time_stats[states][s])
                            ) - n._exit_time_stats[states][s]*(cim[s, s_]-belief_node._transition_stats[states][s, s_]/belief_node._exit_time_stats[states][s])
        return bhc_k
