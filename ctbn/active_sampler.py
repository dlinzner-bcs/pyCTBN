import numpy as np
from ctbn.ctbn_model import CTBN
from ctbn.types import Transition, State, States, Intervention
from typing import NewType
from enum import Enum


class ActiveTransition(Transition):
    def __init__(self, node_id: int, s0: States, s1: States, tau: float, intervention: Intervention) -> None:
        super().__init__(node_id, s0, s1, tau)
        self._intervention = intervention

    @classmethod
    def from_transition(self, transition, intervention):
        return ActiveTransition(transition._node_id, transition._s_init,
                                transition._s_final, transition._exit_time, intervention=intervention)


class SamplingStrategy(Enum):
    RANDOM = 1
    EIG = 2
    BHC = 3
    VBHC = 4


class ActiveSampler():
    def __init__(self, simulator: CTBN, strategy: SamplingStrategy, max_elements=None) -> None:
        self._simulator = simulator
        self._strategy = strategy
        self._max_elements = max_elements

    def sample(self):
        if self._strategy == SamplingStrategy.RANDOM:
            all_combs = self._simulator.all_combos(self._max_elements)
            intervention = np.random.choice(list(all_combs))
            return ActiveTransition.from_transition(self._simulator.intervention(
                intervention=intervention).transition(), intervention=intervention)
        else:
            pass
