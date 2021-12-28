
from typing import List, NewType, Optional
import pprint


State = NewType('State', int)
States = NewType('States', tuple[State])
Intervention = NewType('Intervention', tuple[tuple[int, State]])


class Transition:
    def __init__(self, node_id: int,  s0: States, s1: States, tau: float) -> None:
        self._node_id = node_id
        self._s_init = s0
        self._s_final = s1
        self._exit_time = tau

    def __repr__(self):
        return "Transition of Node " + repr(self._node_id) + " with Initial State:" + repr(self._s_init) + ";End State:" + repr(self._s_final) + ";Exit Time:" + repr(self._exit_time)


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
