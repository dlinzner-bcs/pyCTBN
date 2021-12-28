import unittest
from ctbn.ctbn import CTBNNode, CTBNLearner, CTBNLearnerNode, CTBN, State, States, Trajectory, TestLearner
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
    for k in range(0, 10000):
        traj.append(ctbn.transition())

    print(traj)

    node_A_ = CTBNLearnerNode(state=State(0), states=states,
                              parents=None, alpha=1.0, beta=1.0)
    node_B_ = CTBNLearnerNode(state=State(0), states=states,
                              parents=list([node_A]), alpha=1.0, beta=1.0)
    node_C_ = CTBNLearnerNode(state=State(1), states=states,
                              parents=list([node_A, node_B]), alpha=1.0, beta=1.0)
    node_D_ = CTBNLearnerNode(state=State(1), states=states,
                              parents=list([node_A, node_B]), alpha=1.0, beta=1.0)
    ctbn_learner = CTBNLearner([node_A_, node_B_, node_C_, node_D_])

    for trans in traj._transitions:
        ctbn_learner.update_stats(trans)
    node_B_.estimate_cims()
    2
