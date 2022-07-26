import pytest
from ctbn.learner import CTBNLearner, CTBNLearnerNode
from ctbn.types import State, States, Transition


def test_update_statistics():

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
