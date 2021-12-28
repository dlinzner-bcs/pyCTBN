import unittest
from ctbn.types import Transition, Trajectory, States, State
from ctbn.ctbn import CTBNNode, CTBN
from ctbn.learner import CTBNLearner, CTBNLearnerNode
from ctbn.plots import LearningCurve, PlotType, LearningPlotter
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # unittest.main()

    states = States(list([State(0), State(1), State(2)]))
    node_A = CTBNNode(state=State(0), states=states,
                      parents=None)
    node_B = CTBNNode(state=State(0), states=states,
                      parents=list([node_A]))
    node_C = CTBNNode(state=State(1), states=states,
                      parents=list([node_A, node_B]))
    node_D = CTBNLearnerNode(state=State(1), states=states,
                             parents=list([node_A, node_B]), alpha=1.0, beta=1.0)
    ctbn = CTBN.with_random_cims([node_A, node_B, node_C, node_D], 1.0, 1.0)

    traj = Trajectory()
    for m in range(0, 100):
        ctbn.randomize_states()
        for k in range(0, 100):
            traj.append(ctbn.transition())

    print(traj)

    node_A_ = CTBNLearnerNode.from_ctbn_node(node_A)
    node_B_ = CTBNLearnerNode.from_ctbn_node(node_B)
    node_C_ = CTBNLearnerNode.from_ctbn_node(node_C)
    node_D_ = CTBNLearnerNode.from_ctbn_node(node_D)
    ctbn_learner = CTBNLearner([node_A_, node_B_, node_C_, node_D_])

    learning_curve = LearningCurve(ctbn, ctbn_learner, PlotType.PARAMS)
    c = 0
    for trans in traj._transitions:
        ctbn_learner.update_stats(trans)
        if c % 10 == 0:
            ctbn_learner.estimate_cims()
            learning_curve.add_point()
        c += 1

    plotter = LearningPlotter()
    plotter.plot(learning_curve)
