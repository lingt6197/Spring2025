import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial.distance import squareform, pdist


def erdos_Reyni(n,p):
    """
    Generate Erdos-Reyni graphs manually by generating edges using binomial distribution,
    then directly calculating the adjacency matrix using edges.

    From there, the degree matrix can be found for Laplacian matrix calculation.

    Outputs: adjacency, degree, and square inverse of degree matrices
    """
    edges = np.random.binomial(n*(n-1)/2, p) # n choose 2
    adj = squareform(edges)
    d = np.sum(adj, axis=0) # in this case, axis=1 is equivalent
    D_sqrt_inv = np.linalg.pinv(np.diag(np.sqrt(d)))

    return adj, d, D_sqrt_inv


def randomGeometric(n,d, dist_type="uniform"):
    """
    To generate a random geometric graph (undirected), we need:
     - To generate a sample point cloud
     - Some function that computes distance between points
     - Some function that computes the weights
    """
    if dist_type=="uniform":
        pt_cloud = np.random.uniform(low=0.0, high=1.0, size=(n,d))
    elif dist_type=="normal":
        pt_cloud = np.random.standard_normal(size=(n,d))
    else:
        raise ValueError(f"Unrecognized distribution type {dist_type}")

    # the following lines create a (weighted) random geometric graph from a point cloud
    dist_mtx = pdist(pt_cloud, metric="euclidean") ### make a choice of metric; investigate at least 'euclidean' and 'seuclidean', but feel free to play with others 
    # adj = some_function of 'dist_mtx' (can just be 'adj=dist_mtx', but consider the implications of 
    # choosing other linking functions, like 'adj=1-dist_mtx**2' or 'adj_ij = 1/(dist_mtx_ij)'; which choices make sense for any 
    # distribution, and which do not?)
    dist_mtx = squareform(dist_mtx)

    # weight function: inverse distance
    # 
    # 
    # Reasoning: weight decreases with distance, assumes that things that are close to one another 
    #            are more alike than those that are farther apart.
    adj = 1/dist_mtx
    d = np.sum(adj, axis=0) # in this case, axis=1 is equivalent
    D_sqrt_inv = np.linalg.pinv(np.diag(np.sqrt(d)))
    return adj, d, D_sqrt_inv


def laplacian(adj, d, D_sqrt_inv):
    """
    Graph Laplcian given as: L(G) = D - A
    D = degree matrix
    A = adjacency matrix of undirected graph G

    Input: D, A - both numpy arrays
    Output: normalized Laplacian matrix
    """
    L = d - adj  # graph Laplacian
    return D_sqrt_inv @ L @ D_sqrt_inv  # normalized Laplacian


def euclidean_distance(m1, m2):
    """
    Compute Euclidean distance between adjacency and Laplacian matrices
    """
    return np.linalg.norm(m1 - m2)


def spectral_distance(m1, m2):
    """
    e1, e2: eigenvalues for corresponding matrix
    """
    e1 = np.linalg.eigvalsh(m1)
    e1.sort() # sorts from largest to smallest eigenvalue
    e2 = np.linalg.eigvalsh(m2)
    e2.sort()
    return np.sum(np.abs(e1 - e2))


def plot_grid(values, measurement):
    n_lower = 0; n_upper = 2
    num = np.unique(np.logspace(n_lower, n_upper, num = n_upper+1, base=10, dtype = int))
    prob = np.arange(0.1, 1, 0.1)

    display_rows = [n for n in num if n != 1]

    fig, axes = plt.subplots(len(display_rows), len(prob), figsize=(20, 6), sharex=True, sharey=True)
    fig.suptitle(f"{measurement} Distributions for Erdos-Reyni Random Graphs", x=0.5, y=1.05, fontsize=20)
    
    for i, n in enumerate(display_rows):
        for j, p in enumerate(prob):
            axis = axes[i,j]
            axis.hist(values[(n,p)])
            axis.set_title(f"n={n}, p={p:.1f}")

    fig.text(0.5, -0.05, measurement, ha='center', va='center', fontsize=15)
    fig.text(0.05, 0.5, 'Frequency', ha='center', va='center', rotation='vertical', fontsize=15)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, left=0.1, right=0.9)
    plt.show()


def main(graph_type):
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
    dim = [2,3,4,6,8,10] # arbitrarily chosen
    N = 100

    # generate parameter grids
    adj_euc_grid = {}
    lap_euc_grid = {}
    adj_spec_grid = {}
    lap_spec_grid = {}

    if graph_type == "erdos-reyni":
        for n in num:
            for p in prob:
                print(f"Processing n={n}, p={p:.2f}")
                adj_euc_vals = []
                lap_euc_vals = []
                adj_spec_vals = []
                lap_spec_vals = []

                for _ in range(N):
                    A1, D1, D1_sqrt_inv = erdos_Reyni(n,p)
                    A2, D2, D2_sqrt_inv = erdos_Reyni(n,p)
                    norm_L1 = laplacian(D1, A1, D1_sqrt_inv)
                    norm_L2 = laplacian(D2, A2, D2_sqrt_inv)

                    adj_euc_vals.append(euclidean_distance(A1, A2))
                    lap_euc_vals.append(euclidean_distance(norm_L1, norm_L2))
                
                    adj_spec_vals.append(spectral_distance(A1, A2))
                    lap_spec_vals.append(spectral_distance(norm_L1, norm_L2))

                adj_euc_grid[(n,p)] = adj_euc_vals
                lap_euc_grid[(n,p)] = lap_euc_vals

                adj_spec_grid[(n,p)] = adj_spec_vals
                lap_spec_grid[(n,p)] = lap_spec_vals

    elif graph_type == "random geometric":
        for n in num:
            for d in dim:
                print(f"Processing n={n}, d={d}")
                adj_euc_vals = []
                lap_euc_vals = []
                adj_spec_vals = []
                lap_spec_vals = []

                for _ in range(N):
                    A1, D1, D1_sqrt_inv = randomGeometric(n,d,dist_type="uniform")
                    A2, D2, D2_sqrt_inv = randomGeometric(n,d,dist_type="uniform")
                    norm_L1 = laplacian(D1, A1, D1_sqrt_inv)
                    norm_L2 = laplacian(D2, A2, D2_sqrt_inv)

                    adj_euc_vals.append(euclidean_distance(A1, A2))
                    lap_euc_vals.append(euclidean_distance(norm_L1, norm_L2))
                
                    adj_spec_vals.append(spectral_distance(A1, A2))
                    lap_spec_vals.append(spectral_distance(norm_L1, norm_L2))

                adj_euc_grid[(n,d)] = adj_euc_vals
                lap_euc_grid[(n,d)] = lap_euc_vals

                adj_spec_grid[(n,d)] = adj_spec_vals
                lap_spec_grid[(n,d)] = lap_spec_vals



    # print("Plotting distance distributions...")
    # print("")
    # print("Type 'adjacency/laplacian euclidean', 'frobenius', or 'spectral' to plot corresponding distance distribution. Otherwise, type 'break'.")
    # print("")
    # command = input()
    # responses = {"adjacency euclidean": plot_grid(adj_euc_grid, "Adjacency Euclidean Distance"),
    #             "adjacency frobenius": plot_grid(adj_frob_grid, "Adjacency Frobenius Distance"),
    #             "adjacency spectral": plot_grid(adj_spec_grid, "Adjacency Spectral Distance"),
    #             "laplacian euclidean": plot_grid(lap_euc_grid, "Laplacian Euclidean Distance"),
    #             "laplacian frobenius": plot_grid(lap_frob_grid, "Laplacian Frobenius Distance"),
    #             "laplacian spectral": plot_grid(lap_spec_grid, "Laplacian Spectral Distance")}
    
    # while command != "break":
    #     print(responses[command])
    #     command = input("Continue?")

    plot_grid(adj_euc_grid, "Adjacency Euclidean Distance")
    plot_grid(adj_spec_grid, "Adjacency Spectral Distance")
    plot_grid(lap_euc_grid, "Laplacian Euclidean Distance")
    plot_grid(lap_spec_grid, "Laplacian Spectral Distance")


if __name__ == "__main__":
    main(graph_type="erdos-reyni")
    # test()
