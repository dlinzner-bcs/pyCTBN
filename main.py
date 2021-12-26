from ctbn.ctbn import Node, CTBN, State, States

if __name__ == "__main__":
    states = States(list([State(0), State(1)]))
    node_A = Node(states, parents=None, children=None)
    node_B = Node(states, parents=list([node_A]), children=None)
    node_B.generate_random_cims(1.0, 1.0)
    2
