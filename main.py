from ctbn.ctbn import Node, CTBN, State, States

if __name__ == "__main__":
    states = States(list([State(0), State(1), State(3)]))
    node_A = Node(state=State(0), states=states, parents=None, children=None)
    node_B = Node(state=State(1), states=states,
                  parents=list([node_A]), children=None)
    node_A.generate_random_cims(1.0, 1.0)
    node_B.generate_random_cims(1.0, 1.0)
    print(node_B.cim)
    print(node_B.exit_rate)
    print(node_A.cim)
    print(node_A.exit_rate)
    print(node_B.transition_rates)
    print(node_B.transition_rates)
    for k in range(0, 10):
        print(node_B.transition_rates)
        node_B.next_state()
        print(node_B._state)
