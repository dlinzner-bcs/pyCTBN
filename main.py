from ctbn.ctbn import Node, CTBN, State, States

if __name__ == "__main__":
    states = States(list([State(0), State(1), State(2)]))
    node_A = Node(states, parents=None, children=None)
    node_B = Node(states, parents=list([node_A]), children=None)
    node_B.generate_empty_cims()
    2
