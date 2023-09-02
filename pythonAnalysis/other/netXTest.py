import networkx as nx
import matplotlib.pyplot as plt


G = nx.DiGraph()

# 2 -> 2,3 and 1,2 -> 1,2,3 -> 1,2,3,4 -> 1,2,3,4,5

G.add_node((2,))
G.add_node((2,3))
G.add_node((1,2))
G.add_node((1,2,3))
G.add_node((1,2,3,4))
G.add_node((1,2,3,4,5))

G.add_edge((2,),(2,3))
G.add_edge((2,), (1,2))
G.add_edge((1,2), (1,2,3))
G.add_edge((2,3), (1,2,3))
G.add_edge((1,2,3), (1,2,3,4))
G.add_edge((1,2,3,4), (1,2,3,4,5))

print(G.nodes())
print(G.edges())
print(G)

# traversal

print("BFS Edges")
search = nx.bfs_edges(G, (2,))
print(list(search))

print("BFS Layers")
search = nx.bfs_layers(G, (2,))
print(list(search))

print("BFS Tree")
search = nx.bfs_tree(G, (2,))
print(list(search))

print("BFS Predecessors")
search = nx.bfs_predecessors(G, (2,))
print(list(search))

print("BFS Successors")
search = nx.bfs_successors(G, (2,))
print(list(search))

# nx.draw(G, with_labels=True)
# plt.show()



# deltas graph
print("Deltas Graph")

def delta_graph(G):
    # create a new graph with the same nodes as G
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes)
    H.add_edges_from(G.edges)

    # default delta value is the node itself
    nx.set_node_attributes(H, {node: set(node) for node in G.nodes}, 'delta')

    for u, v in G.edges:
        H.nodes[v]['delta'] -= set(u)
    
    return H

deltas = delta_graph(G)
print(deltas.nodes(data=True))

nx.draw(deltas, with_labels=True, labels=nx.get_node_attributes(deltas, 'delta'))
plt.show()