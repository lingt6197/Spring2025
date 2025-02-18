import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools # helps increase efficiency
import time

# seed = 1
n_lower = 0; n_upper = 1
num = np.unique(np.logspace(n_lower, n_upper, num = 10, dtype = int))
prob = np.arange(0.1, 1, 0.1)
num_graphs = 50
# print(num)
# g = nx.erdos_renyi_graph(10, 0.5, seed = None)

# print(g.nodes)
# print(g.edges)

# nx.draw(g, with_labels=True)
# plt.show()

def generate_graph_matrix(n, p, num_graphs, seed = None):
    # num_graphs: number of random graphs generated per parameter pair (n,p)
    graphs = [nx.erdos_renyi_graph(n, p, seed) for _ in range(num_graphs)]
    adj_matrix = [nx.to_numpy_array(g) for g in graphs]

    return adj_matrix


def generate_adjacency_matrix(m1, m2):
    g1 = nx.from_numpy_array(m1)
    g2 = nx.from_numpy_array(m2)
    matrix = nx.graph_edit_distance(g1, g2, timeout=1)
    
    return matrix


def compute_distances(adj_matrix):
    graph_pairs = itertools.combinations(adj_matrix, 2)
    dist = [generate_adjacency_matrix(g1,g2) for g1,g2 in graph_pairs]
    
    return dist


def main():
    # generate parameter grid
    distribution_grid = {}

    for n in num:
        for p in prob:
            print(f"Processing n={n}, p={p:.2f}")
            matrix = generate_graph_matrix(n, p, num_graphs)
            dist = compute_distances(matrix)
            distribution_grid[(n,p)] = dist

    fig, axes = plt.subplots(len(num), len(prob), figsize=(15, 15), sharex=True, sharey=True)
    fig.suptitle("Graph Edit Distance Distributions for Generated Erdos-Reyni Random Graphs")

    for i, n in enumerate(num):
        for j, p in enumerate(prob):
            axis = axes[i,j]
            axis.hist(distribution_grid[(n,p)])
            axis.set_title(f"n={n}, p={p:.1f}")
            axis.set_xlabel("Graph Edit Distance")
            axis.set_ylabel("Frequency")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

if __name__ == "__main__":
    main()

# startTime = time.time()

# endTime = time.time()
# runTime = endTime - startTime
# print(f"Run time: {runTime} seconds")

