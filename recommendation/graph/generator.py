import networkx as nx


def generate_sample_binary_bipartite_graph():
    g = nx.Graph()

    g.add_node(1, bipartite=0)
    g.add_node(2, bipartite=0)
    g.add_node(3, bipartite=0)
    g.add_node(4, bipartite=0)
    g.add_node(5, bipartite=0)
    g.add_node(6, bipartite=0)

    g.add_node('A', bipartite=1)
    g.add_node('B', bipartite=1)
    g.add_node('C', bipartite=1)
    g.add_node('D', bipartite=1)
    g.add_node('E', bipartite=1)
    g.add_node('F', bipartite=1)

    g.add_edge('A', 1, weight=1)
    g.add_edge('A', 2, weight=1)
    g.add_edge('B', 1, weight=1)
    g.add_edge('B', 2, weight=1)
    g.add_edge('B', 3, weight=1)
    g.add_edge('B', 4, weight=1)
    g.add_edge('B', 5, weight=1)
    g.add_edge('C', 2, weight=1)
    g.add_edge('D', 3, weight=1)
    g.add_edge('E', 4, weight=1)
    g.add_edge('E', 5, weight=1)
    g.add_edge('E', 6, weight=1)
    g.add_edge('F', 6, weight=1)

    return g


def generate_sample_weighted_bipartite_graph():
    g = nx.Graph()

    g.add_node(1, bipartite=0)
    g.add_node(2, bipartite=0)
    g.add_node(3, bipartite=0)
    g.add_node(4, bipartite=0)
    g.add_node(5, bipartite=0)
    g.add_node(6, bipartite=0)

    g.add_node('A', bipartite=1)
    g.add_node('B', bipartite=1)
    g.add_node('C', bipartite=1)
    g.add_node('D', bipartite=1)
    g.add_node('E', bipartite=1)
    g.add_node('F', bipartite=1)

    g.add_edge('A', 1, weight=4)
    g.add_edge('A', 2, weight=2)
    g.add_edge('B', 1, weight=2)
    g.add_edge('B', 2, weight=1)
    g.add_edge('B', 3, weight=4)
    g.add_edge('B', 4, weight=2)
    g.add_edge('B', 5, weight=3)
    g.add_edge('C', 2, weight=5)
    g.add_edge('D', 3, weight=6)
    g.add_edge('E', 4, weight=4)
    g.add_edge('E', 5, weight=2)
    g.add_edge('E', 6, weight=1)
    g.add_edge('F', 6, weight=1)

    return g

