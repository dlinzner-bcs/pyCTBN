import numpy as np
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
        else:
            pass
