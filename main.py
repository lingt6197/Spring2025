import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

def adjacency_matrix(g):
    """
    Graph g is assumed to have no selfloops

    Input: networkx Graph
    Output: numpy array
    """
    adj = nx.adjacency_matrix(g)
    numpy_adj = np.matrix(adj.toarray()) 
    # print(numpy_adj)
    return numpy_adj


def degree_matrix(g):
    """
    Graph g is assumed to have no selfloops
    """
    degrees = np.array([deg for _, deg in g.degree()])
    D = np.diag(degrees) 
    degrees = np.where(degrees == 0, 1, degrees) # replace all 0's with 1's for the sake of division, replace back to 0 after
    D_sqrt_inv = np.diag(1 / np.sqrt(degrees))
    return D, D_sqrt_inv


def normalized_Laplacian(D, A, D_sqrt_inv):
    """
    Graph Laplcian L(G) = D - A
    D = degree matrix
    A = adjacency matrix of undirected graph G

    Input: D, A - both numpy arrays
    Output: normalized Laplacian matrix
    """
    L = D - A  # graph Laplacian
    norm_L = D_sqrt_inv @ L @ D_sqrt_inv  # normalized Laplacian
    return norm_L


def euclidean_distance(a1, a2, l1, l2):
    """
    Compute Euclidean distance between adjacency and Laplacian matrices
    """
    adj_euc = np.linalg.norm(a1 - a2)
    lap_euc = np.linalg.norm(l1 - l2)
    return adj_euc, lap_euc


def frobenius_distance(a1, a2, l1, l2):
    """
    Compute Frobenius distance between adjacency and Laplacian matrices
    """
    adj_frob = np.linalg.norm(a1 - a2, 'fro') # Frobenius norm
    lap_frob = np.linalg.norm(l1 - l2, 'fro')
    return adj_frob, lap_frob


def spectral_distance(a1, a2, l1, l2):
    """
    e1, e2: eigenvalues for corresponding matrix
    """
    adj_e1 = np.linalg.eig(a1)[0]
    adj_e1 = adj_e1[np.argsort(adj_e1)] # sorts from largest to smallest eigenvalue
    adj_e2 = np.linalg.eig(a2)[0]
    adj_e2 = adj_e2[np.argsort(adj_e2)]

    lap_e1 = np.linalg.eig(l1)[0]
    lap_e1 = lap_e1[np.argsort(lap_e1)]
    lap_e2 = np.linalg.eig(l2)[0]
    lap_e2 = lap_e2[np.argsort(lap_e2)]

    if len(adj_e1) == len(adj_e2) and len(lap_e1) == len(lap_e2):
        adj_spec = np.sum(np.abs(adj_e1 - adj_e2))
        lap_spec = np.sum(np.abs(lap_e1 - lap_e2))
    
    return adj_spec, lap_spec

def plot_grid(values, measurement):
    n_lower = 0; n_upper = 2
    num = np.unique(np.logspace(n_lower, n_upper, num = n_upper+1, base=10, dtype = int))
    prob = np.arange(0.1, 1, 0.1)
    N = 200

    display_rows = [n for n in num if n != 1]

    fig, axes = plt.subplots(len(display_rows), len(prob), figsize=(20, 6), sharex=True, sharey=True)
    fig.suptitle(f"{measurement} Distributions for Erdos-Reyni Random Graphs", fontsize=26)

    for i, n in enumerate(display_rows):
        for j, p in enumerate(prob):
            axis = axes[i,j]

            min_xval = min(values[(n,p)])
            max_xval = max(values[(n,p)])
            axis.set_xlim(min_xval,max_xval)
            # axis.set_ylim(0, (N/2))

            axis.hist(values[(n,p)])
            axis.set_title(f"n={n}, p={p:.1f}")

    fig.text(0.5, 0.04, measurement, ha='center', va='center', fontsize=20)
    fig.text(0.04, 0.5, 'Frequency', ha='center', va='center', rotation='vertical', fontsize=20)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, left=0.1, right=0.9)
    plt.show()


def main():
    """
    Generate parameter grid where each index is a parameter pair (n,p)
    For each parameter pair (n,p), N graphs were generated
    Plot distribution of each (n,p) pair for N graphs

    num: number of nodes (incremented by log10 starting from 1)
    prob: probability of edge creation
    N: number of resamples (graphs generated per pair)
    """
    n_lower = 0; n_upper = 2
    num = np.unique(np.logspace(n_lower, n_upper, num = n_upper+1, base=10, dtype = int))
    prob = np.arange(0.1, 1, 0.1)
    N = 200

    # generate parameter grids
    adj_euc_grid = {}
    lap_euc_grid = {}
    adj_frob_grid = {}
    lap_frob_grid = {}
    adj_spec_grid = {}
    lap_spec_grid = {}

    for n in num:
        for p in prob:
            print(f"Processing n={n}, p={p:.2f}")
            adj_euc_vals = []
            lap_euc_vals = []
            adj_frob_vals = []
            lap_frob_vals = []
            adj_spec_vals = []
            lap_spec_vals = []

            for _ in range(N):
                g1 = nx.erdos_renyi_graph(n,p)
                A1 = adjacency_matrix(g1)
                D1, D1_sqrt_inv = degree_matrix(g1)
                norm_L1 = normalized_Laplacian(D1, A1, D1_sqrt_inv)

                g2 = nx.erdos_renyi_graph(n,p)
                A2 = adjacency_matrix(g2)
                D2, D2_sqrt_inv = degree_matrix(g2)
                norm_L2 = normalized_Laplacian(D2, A2, D2_sqrt_inv)

                adj_euc, lap_euc = euclidean_distance(A1, A2, norm_L1, norm_L2)
                adj_frob, lap_frob = frobenius_distance(A1, A2, norm_L1, norm_L2)
                adj_spec, lap_spec = spectral_distance(A1, A2, norm_L1, norm_L2)

                adj_euc_vals.append(adj_euc)
                lap_euc_vals.append(lap_euc)

                adj_frob_vals.append(adj_frob)
                lap_frob_vals.append(lap_frob)

                adj_spec_vals.append(adj_spec)
                lap_spec_vals.append(lap_spec)

            adj_euc_grid[(n,p)] = adj_euc_vals
            lap_euc_grid[(n,p)] = lap_euc_vals

            adj_frob_grid[(n,p)] = adj_frob_vals
            lap_frob_grid[(n,p)] = lap_frob_vals

            adj_spec_grid[(n,p)] = adj_spec_vals
            lap_spec_grid[(n,p)] = lap_spec_vals

    print("Plotting distance distributions...")
    print("")
    print("Type 'adjacency/laplacian euclidean', 'frobenius', or 'spectral' to plot corresponding distance distribution. Otherwise, type 'break'.")
    print("")
    command = input()
    responses = {"adjacency euclidean": plot_grid(adj_euc_grid, "Adjacency Euclidean Distance"),
                "adjacency frobenius": plot_grid(adj_frob_grid, "Adjacency Frobenius Distance"),
                "adjacency spectral": plot_grid(adj_spec_grid, "Adjacency Spectral Distance"),
                "laplacian euclidean": plot_grid(lap_euc_grid, "Laplacian Euclidean Distance"),
                "laplacian frobenius": plot_grid(lap_frob_grid, "Laplacian Frobenius Distance"),
                "laplacian spectral": plot_grid(lap_spec_grid, "Laplacian Spectral Distance")}
    
    while command != "break":
        print(responses[command])
        command = input("Continue?")


if __name__ == "__main__":
    main()

# startTime = time.time()

# endTime = time.time()
# runTime = endTime - startTime
# print(f"Run time: {runTime} seconds")

