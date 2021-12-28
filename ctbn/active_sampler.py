from ctbn.learner import CTBNLearner
from ctbn.types import Transition, State, States
from typing import NewType
from enum import Enum

NewType('Intervention', tuple[tuple[int, State]])


class ActiveTransition(Transition):
    def __init__(self, node_id: int, s0: States, s1: States, tau: float, intervention='Intervention') -> None:
        super().__init__(node_id, s0, s1, tau)
        self._intervention = intervention


class SamplingStrategy(Enum):
    RANDOM = 1
    EIG = 2
    BHC = 3
    VBHC = 4


class ActiveSampler():
    def __init__(self, simulator: CTBNLearner, stragtegy: SamplingStrategy) -> None:
        self._simulator = simulator
        self._strategy = stragtegy
