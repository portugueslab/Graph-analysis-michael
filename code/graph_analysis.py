import os
# import glob
import numpy as np
# import pylab as pl
# import scipy.io as sio
# for_Jyotika.m
from copy import copy, deepcopy
# import pickle
# import matplotlib.cm as cm
# import pdb
# import h5py

import pandas as pd
# import bct
# from collections import Counter 

import sys


# Plotting libs
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.cm as cm
#
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
# graph libs
import networkx as nx

#
import warnings
warnings.simplefilter('always')  # Continually warn if problem

# Define destination variables
sys.path.append("./common/")
# import analyze as anal # Where is this module?
data_dir = "./data/"
data_target_dir = "./data/"
fig_target_dir = "./Figure2/"


# Draw network graphs
import numpy as np
import sys, os
# import snap
import community
import networkx as nx
# import igraph as ig



def plot_connectivity_matrix(correlation_matrix, ax=None, title='Correlation matrix'):
    if ax is None:
        ax = plt.gca()
    lim = np.abs(correlation_matrix).max()
    ax.set_title('{}'.format(title))
    np.fill_diagonal(correlation_matrix, 0)
    im = ax.imshow(correlation_matrix, aspect='auto',cmap='inferno_r',vmin=-lim, vmax=lim)
    ax.tick_params(labelsize=10)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=.7)
    cbar.ax.set_ylabel("Connectivity Strength", rotation=90,
                     labelpad= 20, va="bottom")
    ax.set(xlabel="Nodes", ylabel="Nodes")


def parcellate_zebrafish_brain(ROI_ALL, z_res, nantest, **kwargs):
    """
    Inputs:
     - ROI_ALL: All the roi coordinates
     - 
    Returns
     - XYZ: Coordinates transformed to fit 
     - LR_Centers: left and right centroid from the spatial clustering
     - 
    
    """
    XYZ = ROI_ALL

    # Transform XYZ coordinates
    XYZ,xx,yy,zz =  _transform_xyz_coord(XYZ, z_res, nantest)

    # Find the LR centers
    LR_centers, rxx, ryy, rzz, lxx, lyy, lzz = _cluster_left_right_centroids(XYZ, xx, hf = 205, sep = 60)

    # Find the nearest neighbours
    indices = _get_nearest_neurons_cluster(LR_centers, XYZ)

    # Combine all the scattering points into tuple
    coord_tuple = (xx,yy,zz,rxx,ryy,rzz,lxx,lyy,lzz)

    return XYZ, LR_centers, indices, coord_tuple

def _transform_xyz_coord(XYZ, z_res, nantest):
    # Correct rows by swapping using advanced indices (1,2,0)
    XYZ[:,[0, 1]] = XYZ[:,[1, 0]]
    XYZ[:,[2, 1]] = XYZ[:,[1, 2]]

    # Squeeze and remove nan values
    XYZ = np.squeeze(XYZ[np.argwhere(~np.isnan(nantest)),:])
    xx =  np.squeeze(XYZ[:,0])
    yy =  np.squeeze(XYZ[:,1])
    zz = np.squeeze(XYZ[:,2])

    # Tranform coordinates to planes fit to scanning
    XYZ[:,0] *= 0.6
    XYZ[:,1] *= 0.6
    XYZ[:,2] *= z_res
    XYZ = np.squeeze(XYZ[np.argwhere(~np.isnan(nantest)),:])
    xx =  np.squeeze(XYZ[:,0])
    yy =  np.squeeze(XYZ[:,1])
    zz = np.squeeze(XYZ[:,2]) 
    return XYZ, xx,yy,zz

def _cluster_left_right_centroids(XYZ, xx, hf = 205, sep = 60):
    # Extract L 
    L=np.argwhere(xx>hf)
    XYZL = np.squeeze(XYZ[L,:])
    lxx =  np.squeeze(XYZL[:,0])
    lyy =  np.squeeze(XYZL[:,1])
    lzz = np.squeeze(XYZL[:,2])


    # Find centers
    nClus = 100
    clusterer = KMeans(n_clusters=nClus, random_state=0)
    cluster_labels = clusterer.fit_predict(XYZL)
    centers = clusterer.cluster_centers_
    L_centers = centers

    lxx =  np.squeeze(L_centers[:,0])
    lyy =  np.squeeze(L_centers[:,1])
    lzz = np.squeeze(L_centers[:,2])
    rxx =  max(lxx) -  lxx + hf/2 - sep
    ryy = lyy
    rzz = lzz
    R_centers = np.hstack((rxx.reshape(nClus,1),ryy.reshape(nClus,1),rzz.reshape(nClus,1)))
    LR_centers = np.concatenate((L_centers, R_centers))
    return LR_centers, rxx, ryy, rzz, lxx, lyy, lzz

def _get_nearest_neurons_cluster(LR_centers, XYZ):
    """
    Return:
     - indices: A list of indices for for which each neuron belongs to a specific cluster
    """
    X = np.array(LR_centers)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(XYZ)
    return indices

def plot_3d_projection(XYZ, ax=None, dot_size=100, transparency=0.5):
    if ax is None:
        ax = plt.gca()
    names = ['X-axis','Y-axis','Z-axis']
    for label, name in enumerate(names):
        ax.text3D(
            XYZ[:, 0].mean(),
            XYZ[:, 1].mean(),
            XYZ[:, 2].mean() + 2,
            name,
            horizontalalignment="center",
            bbox=dict(alpha=0.2, edgecolor="w", facecolor="w"),
        )
    ax.scatter(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], 
                edgecolor="k", s=dot_size, 
                cmap='viridis', alpha=transparency)
    ax.view_init(20, -50)
    ax.set_xlabel(names[0], fontsize=12)
    ax.set_ylabel(names[1], fontsize=12)
    ax.set_zlabel(names[2], fontsize=12)
    ax.set_title("Distribution of neurons", fontsize=12)

def plot_spatial_node_clustering(coord_tuple, indices, axs,  nClus: int = 200):
    """
    Display spatial map of clustering results
    """
    xx, yy , zz, rxx, ryy, rzz, lxx, lyy, lzz = coord_tuple 

    axs[0].scatter(lxx, lyy, lw=0, s=15, alpha=1,  color=(0.3,) * 3)
    axs[0].scatter(rxx, ryy, lw=0, s=15, alpha=1,  color=(0.3,) * 3)


    axs[1].scatter(lyy, lzz, lw=0, s=15, alpha=1,  color=(0.3,) * 3)
    axs[1].scatter(ryy, rzz, lw=0, s=15, alpha=1, color=(0.3,) * 3)

    for c in range(2*nClus):
        sub_ind = np.argwhere(indices==c) 
        
        axs[0].scatter(xx[sub_ind], yy[sub_ind], lw=0, s=5, alpha=0.15)
        axs[1].scatter(yy[sub_ind], zz[sub_ind], lw=0, s=5, alpha=0.15)

    for i in range(2):
        # axs[i].set_title(k)
        axs[i].axis("equal")
        axs[i].axis("off")

    coord_tuple = (xx,yy,zz,rxx,ryy,rzz,lxx,lyy,lzz)
    plt.show() 

def plot_corr_vs_dist(spatial_dist_mat, correlation_matrix, ax=None, title=None):
    if ax is None:
        ax = plt.gca()
    # Double verify the spacing is infact correct
    # Generate color array for intra/inter connections
    color_arr = np.asarray(spatial_dist_mat, dtype=str)
    color_arr[0:100,0:100] = 1
    color_arr[100:200,100:200] = 1
    color_arr[0:100,100:200] = 0
    color_arr[100:200,0:100] = 0
    scatter_corr = ax.scatter(spatial_dist_mat,correlation_matrix,c=color_arr)
    ax.set_xlabel('Distance between nodes')
    ax.set_ylabel('Correlation between nodes')
    ax.set_title('Distance vs correlation for {}'.format(title))
    ax.legend(*scatter_corr.legend_elements() ,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.axhline(y=0, xmin=0, xmax= spatial_dist_mat.max())
    classes=['Intra-hemisphere','Inter-hemisphere']
    ax.legend(handles=scatter_corr.legend_elements()[0], labels=classes)

    # Plot the fit
    z = np.polyfit(spatial_dist_mat.ravel(), correlation_matrix.ravel(), 1)
    p = np.poly1d(z)
    ax.plot(spatial_dist_mat.ravel(),p(spatial_dist_mat.ravel()),"r-")

#------------------------------------ Generate graphs --------------------------------------- #

# def generate_snap_unweighted_graph(A, directed=False, save=None):
#     N, _ = A.shape
#     nids = np.arange(N)
    
#     graph = None
#     if directed:
#         graph = snap.TNGraph.New()
#     else:
#         A = np.triu(A)
#         graph = snap.TUNGraph.New()
    
#     for nid in nids: graph.AddNode(int(nid))
#     srcs,dsts = np.where(A == 1)
#     for (src, dst) in list(zip(srcs,dsts)):
#         if src == dst: continue
#         graph.AddEdge(int(src), int(dst))  
#     if save is not None:
#         try:
#             snap.SaveEdgeList(graph, save)
#         except:
#             print('Error, could not save to %s' % save)

#     return graph

def generate_nx_graph(A, directed=False, weighted=False):
    graph = None
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()
        A = np.triu(A)
    graph.add_nodes_from(range(A.shape[0]))
    
    srcs,dsts = np.nonzero(A)
    for (src, dst) in list(zip(srcs,dsts)):
        if src == dst: continue
        if weighted:
            graph.add_edge(src, dst, weight=A[src, dst])
        else:
            graph.add_edge(src, dst)       
    return graph
  

def generate_igraph(A, directed=False):
    if not directed:
        A = np.triu(A)
    sources, targets = A.nonzero()
    weights = A[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(A.shape[0])
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except BaseException:
        print('base exception excepted..')
        pass
    return g, weights


def draw_network_graph(graph_fish, coord_pos, cof_d, node_options = {}, edge_options = {}, draw_edge: bool = False, ax=None, title=''):
    '''
    '''
    if ax is None:
        ax = plt.gca()

    # get edge weights labels
    labels = nx.get_edge_attributes(graph_fish, 'weight')

    # Draw selected coefficient
    d = dict(graph_fish.degree)
    
    # Color nodes according to their hemisphere
    # color_map = []
    # for node in graph_fish:
    #     if node < 100:
    #         color_map.append('blue')
    #     else: 
    #         color_map.append('green')

    nx.draw_networkx_nodes(graph_fish, node_size=[cof_d[k] for k in cof_d], pos=coord_pos, ax=ax) # draw nodes and edges
    if draw_edge:
        for edge in graph_fish.edges(data='weight'):
            nx.draw_networkx_edges(graph_fish, coord_pos, edgelist=[edge], width=edge[2])
    ax.set_title(title)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

def efficiency(G, n1, n2):
    """
    Calculate the efficiency of a pair of nodes in a graph. 
    Efficiency of a pair of nodes is defined as the inverse of the shortest path each
    node.
    
    Args:
        G : graph
            An undirected graph to compute the average local efficiency of
        n1, n2: node
            Nodes in the graph G
    Returns:
        float
            efficiency between the node u and v"""

    if G.is_directed() is True:
        warnings.warn("Graph shouldn't be directed", Warning)

    if nx.has_path(G, n1, n2):
        efficiency_param = 1. / nx.shortest_path_length(G, n1, n2)
        return efficiency_param
    else:
        return 0.

def global_efficiency(G):
    """Calculate global efficiency for a graph. Global efficiency is the
    average efficiency of all nodes in the graph.
    Parameters
    ----------
    G : networkX graph
    Returns
    -------
    float : avg global efficiency of the graph
    """

    if G is None:
        return 0.
    if G.is_directed() is True:
        warnings.warn("Graph shouldn't be directed", Warning)

    n_nodes = G.number_of_nodes()
    if n_nodes < 2:
        return 0
    den = np.float(n_nodes * (n_nodes - 1))

    return sum(efficiency(G, n1, n2) for n1, n2 in permutations(G, 2)) / den

def local_efficiency(G):
    """Calculate average local efficiency for a graph. Local efficiency is the
    average efficiency of all neighbors of the given node.
    Parameters
    ----------
    G : networkX graph
    Returns
    -------
    float : avg local efficiency of the graph
    """

    if G is None:
        return 0
    if G.is_directed() is True:
        warnings.warn("Graph shouldn't be directed", Warning)

    return sum([global_efficiency(G.subgraph(G[v])) for v in G]) / G.order()

def swapped_cost_distr(A, D, n_trials=500, percent_change=True):
    """Calculate how much the wiring distance cost changes for a random node
    swap.
    Args:
        A: adjacency matrix
        D: distance matrix
        n_trials: how many times to make a random swap (starting from the
            original configuration)
        percent_change: set to True to return percent changes in cost
    Returns:
        vector of cost changes for random swaps"""
    # Calculate true cost
    true_cost = wiring_distance_cost(A, D)
    # Allocate space for storing amount by which cost changes
    cost_changes = np.zeros((n_trials,), dtype=float)

    # Perform random swaps
    for trial in range(n_trials):
        print('trial %d' % trial)
        # Randomly select two nodes
        idx0, idx1 = np.random.permutation(D.shape[0])[:2]
        # Create new distance matrix
        D_swapped = aux.swap_nodes(D, idx0, idx1)
        # Calculate cost change for swapped-node graph
        cost_changes[trial] = wiring_distance_cost(A, D_swapped) - true_cost

    if percent_change:
        cost_changes /= true_cost
        cost_changes *= 100

    return cost_changes


def edge_count(G):
    """Helper for number of edges"""
    if G is None:
        return 0
    else:
        return G.number_of_edges()


def density(G):
    """Helper for connection density"""
    if G is None:
        return 0
    else:
        return nx.density(G)

def get_node_centrality(graph, gtype='nx'):
    nids, deg_centr = [], []
    if gtype == 'igraph':
        for NI in graph.Nodes():
            centr = ig.GetDegreeCentr(graph, NI.GetId())
            nids.append(NI.GetId())
            deg_centr.append(centr)
    elif gtype == 'nx':
        # Optino 1:
        # nnodes = graph.number_of_nodes()
        # output = graph.degree(range(nnodes), weight='weight')
        # for (nid, con) in output:
        #     nids.append(nid)
        #     deg_centr.append(con) 

        # Compute the degree centrality
        deg_dict = nx.degree_centrality(graph)
        for k in np.sort(list(deg_dict.keys())):
            nids.append(k)
            deg_centr.append(deg_dict[k])
        
    return np.asarray(nids, dtype='uint32'), np.asarray(deg_centr, dtype='float32')


def get_eigenvector_centrality(graph, gtype='nx'):
    nids, ev_centr = [], []
    if gtype == 'snap':
        NIdEigenH = snap.TIntFltH()
        snap.GetEigenVectorCentr(graph, NIdEigenH)
        for item in NIdEigenH:
            nids.append(item)
            ev_centr.append(NIdEigenH[item])
    elif gtype == 'nx':
        centrality = nx.eigenvector_centrality(graph, weight='weight')
        nids, ev_centr = [], []
        for node in np.sort(list(centrality.keys())):
            nids.append(node)
            ev_centr.append(centrality[node])
            
    return np.asarray(nids, dtype='uint32'), np.asarray(ev_centr, dtype='float32')


def get_katz_centrality(A, b=1.0, normalized=False):
    eigs = np.linalg.eigvals(A)
    largest_eig = max(eigs.real())
    print(largest_eig)
    alpha = 1./largest_eig - 1e-4
    n = A.shape[0]
    b = np.ones((n,1))*float(b)
    centrality = np.linalg.solve(np.eye(n,n) - (alpha * A) , b)
    if normalized:
        norm = np.sign(sum(centrality) * np.linalg.norm(centrality))
    else:
        norm = 1.0
    return centrality / norm
    

def get_betweenness_centrality(graph,k=None):
    nids, bw_centr = [], []
    centrality = nx.betweenness_centrality(graph,k=k)
    for node in np.sort(list(centrality.keys())):
        nids.append(node)
        bw_centr.append(centrality[node])
    return np.asarray(nids, dtype='uint32'), np.asarray(bw_centr, dtype='float32')

def get_clustering_coefficient(graph):
    nids, ccs = [], []
    cc_output = nx.clustering(graph, weight='weight')
    for nid in np.sort(list(cc_output.keys())):
        nids.append(nid)
        ccs.append(cc_output[nid])
    return np.asarray(nids, dtype='uint32'), np.asarray(ccs, dtype='float32')

          
def get_transitivity(graph, gtype='nx'):
    t = None
    if gtype == 'nx':
        t = nx.transitivity(graph)
    elif gtype == 'snap':
        t = 1.
    return np.float32(t)


def get_outgoing_degrees(J, threshold, pos=True):
    from futils import jthreshold
    if threshold is not None:
        J = jthreshold(J, threshold, binarized=True, pos=pos, above=True)
    outgoing_degrees = []
    for i in range(J.shape[0]):
        outgoing = J[:,i]
        outgoing_degrees.append(np.sum(outgoing))
    outgoing_degrees = np.asarray(outgoing_degrees, dtype='uint32')   
    return outgoing_degrees


def extract_outgoing_hubs(J, jthreshold, hub_percentile, N, idxs=None):
    outgoing_degrees = get_outgoing_degrees(J, jthreshold)
    cutoff = np.quantile(outgoing_degrees, hub_percentile)
    outgoing_hub_idxs = np.where(outgoing_degrees > cutoff)[0]
    if idxs is None:
        valid_nids = outgoing_hub_idxs
    else:
        valid_nids = []
        for nid in outgoing_hub_idxs:
            if nid not in idxs: 
                valid_nids.append(nid)
    
    return valid_nids, outgoing_degrees

def louvain_clustering(graph):
    part = community.best_partition(graph, weight='weight')
    modularity = community.modularity(part, graph)
    return part, modularity
        
              
def leiden_clustering(adjacency, res=1.0, directed=False, part=None):
    import leidenalg
    g, weights= generate_igraph(adjacency,directed=directed)
    if part is None:
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res)
        part = part.membership
    modularity = g.modularity(part, weights=weights)
    return part, modularity 


def swap_nodes(D, idx0, idx1):
    """Return distance matrix after randomly swapping a pair of nodes.
       Returns:
           swapped distance matrix, indices of swapped nodes"""
    D_swapped = D.copy()

    # Swap rows & columns of distance matrix
    D_swapped[[idx0, idx1], :] = D_swapped[[idx1, idx0], :]
    D_swapped[:, [idx0, idx1]] = D_swapped[:, [idx1, idx0]]

    return D_swapped


def lesion_graph_randomly(graph, prop):
    """
    Randomly remove vertices from a graph according to probability of removal.
    Args:
        graph: NetworkX graph to be lesioned.
        prop: Occupation probability (probability of node being pruned)
    Returns:
        G: NetworkX graph
        A: Adjacency matrix for graph
    """
    G = deepcopy(graph)
    if prop == 0.:
        return G, nx.adjacency_matrix(G)

    # Error checking
    assert prop >= 0. and prop <= 1.0, 'prop must be 0.0 <= prop <= 1.0'
    assert graph.order > 0, 'Graph is empty'

    # Get list of nodes and probabilty (uniform random) of executing each one
    node_list = graph.nodes()
    execute_prob = np.random.random((len(node_list),))

    # Identify random nodes to cut
    cut_nodes = [node_list[i] for i in range(len(node_list))
                 if execute_prob[i] <= prop]

    G.remove_nodes_from(cut_nodes)

    if G.order() > 0:
        return G, nx.adjacency_matrix(G)
    else:
        #print 'Graph completely lesioned.'
        return G, None


def lesion_graph_degree(graph, num_lesions):
    """
    Remove vertices from a graph according to degree.
    Args:
        graph: NetworkX graph to be lesioned.
        num_lesions: Number of top degree nodes to remove.
    Returns:
        G: NetworkX graph
        A: Adjacency matrix for graph
    """
    # Error checking
    G = deepcopy(graph)
    if num_lesions == 0:
        return G, nx.adjacency_matrix(G)

    assert num_lesions >= 0 and num_lesions < graph.order, 'Attempting to\
        remove too many/few nodes'

    for l in range(num_lesions):
        # Identify node to cut
        node_i, node_d = max(G.degree().items(),
                             key=lambda degree: degree[1])
        G.remove_node(node_i)

    if G.order() > 0:
        return G, nx.adjacency_matrix(G)
    else:
        #print 'Graph completely lesioned.'
        return None, None


def lesion_graph_degree_thresh(graph, threshold):
    """
    Remove vertices from a graph with degree greater than or equal to
    threshold.
    Parameters:
    -----------
        graph: NetworkX graph to be lesioned.
        threshold: Degree above which to remove nodes.
    Returns:
    --------
        G: NetworkX graph
        A: Adjacency matrix for graph
    """
    # Error checking
    G = deepcopy(graph)

    assert threshold >= 0, " In percolation, `threshold` must be >= 0."
    # Check if lesioning is necessary for threshold
    if threshold > max(G.degree().values()):
        return G, nx.adjacency_matrix(G)

    # Identify all node indices >= threshold
    node_inds = np.where(np.asarray(G.degree().values()) >= threshold)[0]
    # Eliminate these nodes
    G.remove_nodes_from(node_inds)

    if G.order() > 0:
        return G, nx.adjacency_matrix(G)
    else:
        #print 'Graph completely lesioned.'
        return None, None

    
def plot_higher_order(input_filepath, spatial_coords, view, idxs=None):
    import matplotlib.pyplot as plt
    
    nids1, nids2 = [], []
    f = open(input_filepath, 'r')
    line = f.readline().strip('\n').split('\t')
    for l in line: nids1.append(int(l))
    line = f.readline().strip('\n').split('\t')
    for l in line: nids2.append(int(l))
        
    if idxs is not None:
        idxs = np.asarray(idxs, dtype='uint32')
        nids1 = idxs[nids1]
        nids2 = idxs[nids2]
    
    fig = plt.figure(figsize=(18,8))
    ax = plt.axes(projection='3d')
    ax.scatter(*spatial_coords[nids1,:].T, color='r')
    ax.scatter(*spatial_coords[nids2,:].T, color='b')
    ax.scatter(*spatial_coords.T, alpha=0.1, color='k')
    ax.view_init(*view)
    
    
def filter_triplets(triplets, idxsA, idxsB, idxsC=None):
    filtered_triplets = []
    for (x,y,z) in triplets:
        if x not in idxsA: continue
        if idxsC is None:
            if y in idxsB and z in idxsB:
                if not ((x,y,z) in filtered_triplets or (x,z,y) in filtered_triplets):
                    filtered_triplets.append([x,y,z])
        else:
            if (y in idxsB and z in idxsC) or (y in idxsC and z in idxsB):
                if not( (x,y,z) in filtered_triplets or (x,z,y) in filtered_triplets):
                       filtered_triplets.append([x,y,z])
                       
    return np.asarray(filtered_triplets, dtype='uint32')

def get_contagious_edges(triplets):
    motif_edges = []    
    for (x,y,z) in triplets:
        if [x,y] not in motif_edges:
            motif_edges.append([x,y])
        if [x,z] not in motif_edges: 
            motif_edges.append([x,z])
    return motif_edges

def get_bifan_edges(triplets):
    
    bifan_edges = []    
    valid_tails = []
    for _,y,z in triplets:
        if [y,z] not in valid_tails and [z,y] not in valid_tails: valid_tails.append([y,z])
    valid_tails = np.asarray(valid_tails, dtype='uint32')
    
    for (i,(x,y,z)) in enumerate(triplets):
        locs = np.where((valid_tails == (y,z))|(valid_tails==(z,y)))[0]
        for (x2,y2,z2) in triplets[locs]:
            if x == x2: continue
            if (y == y2 and z == z2) or (y == z2 and z == y2):
                if [x,y] not in bifan_edges:
                    bifan_edges.append([x,y])
                if [x,y2] not in bifan_edges:
                    bifan_edges.append([x,y2])
                if [x2,y] not in bifan_edges:
                    bifan_edges.append([x2,y])
                if [x2,y2] not in bifan_edges:
                    bifan_edges.append([x2,y2])
    return bifan_edges
    
def extract_motifs(J, motif='send'):
    triplets = []   
    srcs, dsts = J.nonzero()
    if motif =='send':
        i = 0
        for (src, dst) in list(zip(srcs, dsts)):
            locs = np.where(srcs == src)[0]
            tsrcs, tdsts = srcs[locs], dsts[locs]
            for (src2, dst2) in list(zip(tsrcs, tdsts)):
                if dst == dst2: continue
                if J[dst,dst2] or J[dst2,dst] or J[dst,src] or J[dst2,src]: continue
                triplets.append((src,dst,dst2))
            if i % 5000 == 0:
                print(i,len(srcs))
            i += 1
            
    ## do 'out'
    elif motif == 'receive':
        i = 0
        for (src, dst) in list(zip(srcs, dsts)):
            locs = np.where(dsts == dst)[0]
            tsrcs, tdsts = srcs[locs], dsts[locs]
            for (src2, dst2) in list(zip(tsrcs, tdsts)):
                if src == src2: continue
                if J[dst,src] or J[dst, src2] or J[src,src2] or J[src2,src]: continue
                triplets.append((dst, src, src2))
            if i % 5000 == 0:
                print(i, len(srcs))
            i += 1
    
    elif motif == 'recurrent':
        i = 0
        for (src, dst) in list(zip(srcs, dsts)):
            if J[dst, src]: continue

            locs = np.where(srcs == dst)[0]
            tsrcs, tdsts = srcs[locs], dsts[locs]
            for (src2, dst2) in list(zip(tsrcs, tdsts)):
                if src == src2: continue
                if J[dst2, src2]: continue
                locs2 = np.where(srcs == dst2)[0]
                ttsrcs, ttdsts = srcs[locs2], dsts[locs2]
                for (src3, dst3) in list(zip(ttsrcs, ttdsts)):
                    if dst3 != src: continue
                    if J[dst3,src3]: continue
                    triplets.append((src,dst,dst2))
            if i % 5000 == 0: 
                print(i, len(srcs))
            i += 1
                    
    else:
        print('motif argument not recognized')
    return triplets


def plot_motif_statistics(baseline, presz):
    import matplotlib.pyplot as plt
    
    motif_jump = presz - baseline
    bins = np.linspace(0, 10000, 20)
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].plot(np.sort(baseline)[:], color='k')
    ax[0].plot(np.sort(presz)[:], color='r')
    ax[0].set_yscale('log')
    ax[1].hist([baseline, presz], color=['k', 'r'], bins=bins, rwidth=0.65)
    ax[1].set_yscale('log')
    plt.show()

    fig, ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].plot(np.sort(motif_jump))
    ax[1].hist(motif_jump, color='k', rwidth=0.65)
    ax[1].set_yscale('log')
    plt.show()

    sizes = []
    colors = []
    for jump in motif_jump:
        ajump = abs(jump)
        if ajump < 10: sizes.append(0.5)
        elif ajump >= 10 and ajump < 100: sizes.append(1.0)
        elif ajump >= 100 and ajump < 1000: sizes.append(5.0)
        else: sizes.append(20.0)
        if jump > 0: colors.append('r')
        else: colors.append('k')

    plt.figure(figsize=(12,8))
    plt.scatter(baseline, presz, color=colors, s=sizes, alpha=1.0)
    plt.plot([i for i in range(15000)], [i for i in range(15000)], color='k', linestyle='--')
    plt.xscale('log'); plt.yscale('log')
    plt.title(np.corrcoef(baseline, presz)[0][1] ** 2)
    plt.show()
    

def nid_motif_participation(triplets,N, loc='full'):
    participation = [0 for _ in range(N)]
    for (x,y,z) in triplets:
        if loc == 'head':
            participation[x] += 1
        elif loc == 'tail':
            participation[y] += 1
            participation[z] += 1
        else:
            participation[x] += 1
            participation[y] += 1
            participation[z] += 1
            
    return np.asarray(participation, dtype='uint32')


def clustering(A):
    r"""Compute the clustering coefficient for nodes.
    For unweighted graphs, the clustering of a node :math:`u`
    is the fraction of possible triangles through that node that exist,
    .. math::
      c_u = \frac{2 T(u)}{deg(u)(deg(u)-1)},
    where :math:`T(u)` is the number of triangles through node :math:`u` and
    :math:`deg(u)` is the degree of :math:`u`.
    For weighted graphs, there are several ways to define clustering [1]_.
    the one used here is defined
    as the geometric average of the subgraph edge weights [2]_,
    .. math::
       c_u = \frac{1}{deg(u)(deg(u)-1))}
             \sum_{vw} (\hat{w}_{uv} \hat{w}_{uw} \hat{w}_{vw})^{1/3}.
    The edge weights :math:`\hat{w}_{uv}` are normalized by the maximum weight
    in the network :math:`\hat{w}_{uv} = w_{uv}/\max(w)`.
    The value of :math:`c_u` is assigned to 0 if :math:`deg(u) < 2`.
    For directed graphs, the clustering is similarly defined as the fraction
    of all possible directed triangles or geometric average of the subgraph
    edge weights for unweighted and weighted directed graph respectively [3]_.
    .. math::
       c_u = \frac{1}{deg^{tot}(u)(deg^{tot}(u)-1) - 2deg^{\leftrightarrow}(u)}
             T(u),
    where :math:`T(u)` is the number of directed triangles through node
    :math:`u`, :math:`deg^{tot}(u)` is the sum of in degree and out degree of
    :math:`u` and :math:`deg^{\leftrightarrow}(u)` is the reciprocal degree of
    :math:`u`.
    Parameters
    ----------
    G : graph
    nodes : container of nodes, optional (default=all nodes in G)
       Compute clustering for nodes in this container.
    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.
    Returns
    -------
    out : float, or dictionary
       Clustering coefficient at specified nodes
    Examples
    --------
    >>> G=nx.complete_graph(5)
    >>> print(nx.clustering(G,0))
    1.0
    >>> print(nx.clustering(G))
    {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    Notes
    -----
    Self loops are ignored.
    References
    ----------
    .. [1] Generalizations of the clustering coefficient to weighted
       complex networks by J. Saramäki, M. Kivelä, J.-P. Onnela,
       K. Kaski, and J. Kertész, Physical Review E, 75 027105 (2007).
       http://jponnela.com/web_documents/a9.pdf
    .. [2] Intensity and coherence of motifs in weighted complex
       networks by J. P. Onnela, J. Saramäki, J. Kertész, and K. Kaski,
       Physical Review E, 71(6), 065103 (2005).
    .. [3] Clustering in complex directed networks by G. Fagiolo,
       Physical Review E, 76(2), 026107 (2007).
    """
  

    td_iter = _triangles_and_degree_iter(A)
    clusterc = {v: 0 if t == 0 else t / (d * (d - 1)) for
                    v, d, t, _ in td_iter}
    return clusterc



def _weighted_triangles_and_degree_iter(A):
    """ Return an iterator of (node, degree, weighted_triangles).
    Used for weighted clustering.
    """

    max_weight = np.max(A)
    def wt(u, v):
        return A[u][v] / max_weight

    for i in range(A.shape[0]):
        nbhrs = list(range(A.shape[0]))
        inbrs = set(nbrs) - {i}
        weighted_triangles = 0
        seen = set()
        for j in inbrs:
            seen.add(j)
            # This prevents double counting.
            jnbrs = set(G[j]) - seen
            # Only compute the edge weight once, before the inner inner
            # loop.
            wij = wt(i, j)
            weighted_triangles += sum((wij * wt(j, k) * wt(k, i)) ** (1 / 3)
                                      for k in inbrs & jnbrs)
        yield (i, len(inbrs), 2 * weighted_triangles)