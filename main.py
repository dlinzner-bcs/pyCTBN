from ctbn.ctbn import CTBNNode, CTBN, State, States, Trajectory
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    states = States(list([State(0), State(1), State(2)]))
    node_A = CTBNNode(state=State(0), states=states,
                      parents=None, children=None)
    node_B = CTBNNode(state=State(0), states=states,
                      parents=None, children=None)
    node_C = CTBNNode(state=State(1), states=states,
                      parents=list([node_A, node_B]), children=None)
    ctbn = CTBN.with_random_cims([node_A, node_B, node_C], 1.0, 1.0)

    traj = Trajectory()
    for k in range(0, 10):
        traj.append(ctbn.transition())

    print(traj)
