import unittest
import numpy as np
from ctbn.active_sampler import ActiveSampler, SamplingStrategy
from ctbn.types import Transition, Trajectory, States, State
from ctbn.ctbn_model import CTBNNode, CTBN
from ctbn.learner import CTBNLearner, CTBNLearnerNode
from ctbn.plots import LearningCurve, PlotType, LearningPlotter
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # unittest.main()

    states = States(list([State(0), State(1)]))
    node_A = CTBNNode(state=State(0), states=states,
                      parents=None, alpha=0.25, beta=1.0)
    node_B = CTBNNode(state=State(0), states=states,
                      parents=list([node_A]), alpha=4.0, beta=1.0)
    node_C = CTBNNode(state=State(1), states=states,
                      parents=list([node_A, node_B]), alpha=0.25, beta=1.0)
    node_D = CTBNNode(state=State(1), states=states,
                      parents=list([node_A, node_B]), alpha=4.0, beta=1.0)
    ctbn = CTBN([node_A, node_B, node_C, node_D])
    print(ctbn.amalgamation())
    print(ctbn.state2global())
    print(ctbn.all_combos(1))

    curves_list = dict()

    curves = []
    for m in range(0, 10):
        traj = Trajectory()
       # ctbn.randomize_states()
        for k in range(0, 100):
            traj.append(ctbn.transition())
        node_A_ = CTBNLearnerNode.from_ctbn_node(node_A)
        node_B_ = CTBNLearnerNode.from_ctbn_node(node_B)
        node_C_ = CTBNLearnerNode.from_ctbn_node(node_C)
        node_D_ = CTBNLearnerNode.from_ctbn_node(node_D)
        ctbn_learner = CTBNLearner([node_A_, node_B_, node_C_, node_D_])

        learning_curve = LearningCurve(ctbn, ctbn_learner, PlotType.PARAMS)
        c = 0
        for trans in traj._transitions:
            ctbn_learner.update_stats(trans)
            if c % 1 == 0:
                ctbn_learner.estimate_cims()
                learning_curve.add_point()
            c += 1
        curves.append(learning_curve)
    curves_list['passive'] = curves

    curves = []
    for m in range(0, 10):
        traj = Trajectory()
        # ctbn.randomize_states()

        node_A_ = CTBNLearnerNode.from_ctbn_node(node_A)
        node_B_ = CTBNLearnerNode.from_ctbn_node(node_B)
        node_C_ = CTBNLearnerNode.from_ctbn_node(node_C)
        node_D_ = CTBNLearnerNode.from_ctbn_node(node_D)
        ctbn_learner = CTBNLearner([node_A_, node_B_, node_C_, node_D_])
        active_sampler = ActiveSampler(
            simulator=ctbn, belief=ctbn_learner, strategy=SamplingStrategy.BHC, max_elements=1)
        for k in range(0, 100):
            print(k)
            trans = active_sampler.sample()
            traj.append(trans)
            ctbn_learner.update_stats(trans)

        node_A_ = CTBNLearnerNode.from_ctbn_node(node_A)
        node_B_ = CTBNLearnerNode.from_ctbn_node(node_B)
        node_C_ = CTBNLearnerNode.from_ctbn_node(node_C)
        node_D_ = CTBNLearnerNode.from_ctbn_node(node_D)
        ctbn_learner = CTBNLearner([node_A_, node_B_, node_C_, node_D_])
        learning_curve = LearningCurve(ctbn, ctbn_learner, PlotType.PARAMS)
        c = 0
        for trans in traj._transitions:
            ctbn_learner.update_stats(trans)
            if c % 1 == 0:
                ctbn_learner.estimate_cims()
                learning_curve.add_point()
            c += 1
        curves.append(learning_curve)
    curves_list['bhc'] = curves

    curves = []
    for m in range(0, 10):
        traj = Trajectory()
        # ctbn.randomize_states()

        node_A_ = CTBNLearnerNode.from_ctbn_node(node_A)
        node_B_ = CTBNLearnerNode.from_ctbn_node(node_B)
        node_C_ = CTBNLearnerNode.from_ctbn_node(node_C)
        node_D_ = CTBNLearnerNode.from_ctbn_node(node_D)
        ctbn_learner = CTBNLearner([node_A_, node_B_, node_C_, node_D_])
        active_sampler = ActiveSampler(
            simulator=ctbn, belief=ctbn_learner, strategy=SamplingStrategy.RANDOM, max_elements=1)
        for k in range(0, 100):
            trans = active_sampler.sample()
            traj.append(trans)

        learning_curve = LearningCurve(ctbn, ctbn_learner, PlotType.PARAMS)
        c = 0
        for trans in traj._transitions:
            ctbn_learner.update_stats(trans)
            if c % 1 == 0:
                ctbn_learner.estimate_cims()
                learning_curve.add_point()
            c += 1
        curves.append(learning_curve)
    curves_list['random'] = curves

    plotter = LearningPlotter()
    plotter.plot(curves_list)
