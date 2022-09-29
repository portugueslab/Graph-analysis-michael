import os
from turtle import color
import numpy as np
import pickle

# from numba import jit
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from collections import OrderedDict
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.cluster.vq import kmeans2
from scipy.cluster.vq import whiten
from scipy.spatial.distance import cdist, pdist
import scipy.cluster.hierarchy as sch
from scipy.stats import spearmanr
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances, euclidean_distances
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools 

# import tsne_adapted
# import pca_basic
# import similarity_utils as simu

# import tsne_adapted
# import pca_basic
# import similarity_utils as simu


def fancy_dendrogram(*args, **kwargs):
    """Visulize dendrogram distance in plot"""
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)', fontsize=16)
        plt.xlabel('sample index or (cluster size)', fontsize=16)
        plt.ylabel('distance', fontsize=16)
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

# Auxilary function for painting

class functional_clustering(object):
    """
    """
    def __init__(self, data, threshold=0.8):
        self.X = data
        self.n_neuron = np.shape(self.X)[0]
        self.Lin = []
        self.threshold=threshold

    def normalize(self):
        # 1. normalization (var=1 in all dims)
        self.X_n = StandardScaler().fit_transform(self.X)

    def pca(self, norm=True):
        if norm:
            self.Z, self.k = pca_basic.pca_basic(self.X_n, threshold=self.threshold)
        else:
            self.Z, self.k = pca_basic.pca_basic(self.X, threshold=self.threshold)

    def probability_matrix(self, k, data=[], iter_n = 1000, boot_n = 100):
        """Different initial value.
        TODO: add subsampling
        """
        if  len(data)==0:
            data = self.Z.T

        n = self.n_neuron
        L = np.zeros((n,boot_n))
        for boot in range(boot_n):
            #if boot%10==0:
                #print(boot)
            # method1: random sub sample (99% of data) with random initiation
            #indices = np.sort(np.random.choice(range(n), size=round(n*0.99), replace=False))
            #clusters = kmeans2(Z_n[indices,:], k, iter=iter_n, thresh=5e-6,minit='random')
            # method2: random initiation
            clusters = kmeans2(np.nan_to_num(data), k, iter=iter_n, thresh=5e-6, minit='points')
            L[:,boot]=clusters[1]+3

        matrix = np.zeros((n,n))
        dist=[]
        for boot in range(boot_n):
            tmp = np.atleast_2d(L[:,boot]).T*(1/(np.atleast_2d(L[:,boot]))) 
            tmp[tmp!=1]=0
            matrix+=tmp
            matrix_previous=matrix-tmp
            #plt.figure()
            #plt.imshow(matrix/float(boot)-matrix_previous/float(boot-1))
            # compute distance between two adjacent similarity matrix
            dist.append(np.linalg.norm(matrix/float(boot)-matrix_previous/float(boot-1)))
        self.dist=np.array(dist)
        self.matrix=matrix/float(boot_n)
        return self.matrix

    def linkage(self, input=[]):
        """1. generate the linkage matrix
        input=[]: run hierarchical clustering on the coclustering matrix
        input!=[]: run hierarchical clusttering on the input
        """
        if len(input)==0:
            self.Lin = sch.linkage(self.matrix, 'ward')
        else:
            self.Lin = sch.linkage(input, 'ward')

        # 'ward' is one of the methods that can be used to calculate the distance between newly formed clusters. 
        # 'ward' causes linkage() to use the Ward variance minimization algorithm.

        # 2. compares the actual pairwise distances of all your samples to those implied by the hierarchical clustering. 
        #c, coph_dists = sch.cophenet(Lin, pdist(matrix))
        #print(c)
    # prepare a dictionary to color a dendrogram by clusters.
    
    def _paint_dendrogram(self, Z, cl):
        n = len(Z) + 1
        # n = len(Z)
        # first, assign a color (in matplotlib string code) to each leaf by the cluster to which it belongs
        cl_colors = {i: to_hex(cm.tab20.colors[cl[i] - 1]) for i in range(n)}
        # linkage matrix Z is (n-1) by 4.
        #     at the i-th iteration, clusters with indices Z[i, 0] and Z[i, 1] are combined to form cluster n+i.
        #     a cluster with an index less than n corresponds to one of the n original observations.
        #     rows are ordered by increasing distance.
        # a dendrogram has 2n-1 elements.
        #     the first n elements are the original leaves, which don't seem to be colored since they are points on the axis.
        #     the last n-1 elements are the links berween clusters, whose colors can be assigned individually by "link_color_func=lambda k: colors[k]"
        for i, original_cls in enumerate(Z[:, :2].astype(int)):
            # check the original colors of the clusters that are combined in the row
            color1, color2 = (cl_colors[c] for c in original_cls)
            # the i-th iteration of Z will be the (n+i)-th element drawn in the dendrogram
            if color1 == color2:  # if the color of two clusters are the same, the new cluster is given the same color
                cl_colors[i + n] = color1
            else:  # otherwise black
                cl_colors[i + n] = 'k'
        return cl_colors

    def plot_matrix(self, figname=[], input=[], D=[], vmax=1, vmin=0, cmap='Purples'):
        if len(D)==0:
            D = self.matrix
        else:
            D = D
        # Compute and plot dendrogram.
        fig = plt.figure()
        axdendro = fig.add_axes([0.09,0.1,0.2,0.8])
        if len(self.Lin)>0:
            Y = self.Lin
        else:
            if len(input)==0:
                Y = sch.linkage(self.matrix, method='ward')
            else:
                Y = sch.linkage(input, method='ward')

        # Paint the dendogram accordingly
        cl_colors = self._paint_dendrogram(Y, self.clusters)
        Z = sch.dendrogram(
                Y,
                no_labels=True,
                orientation='left',
                distance_sort='descending',
                link_color_func=lambda k: cl_colors[k]
        )
        axdendro.set_xticks([])
        axdendro.set_yticks([])


        # Plot distance matrix.
        axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
        index = Z['leaves']
        D = D[index,:]
        D = D[:,index]
        self.D_ordered = D
        im = axmatrix.matshow(D, origin='lower', cmap=cmap, vmax=vmax, vmin=vmin)
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])

        # Plot colorbar.
        axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
        plt.colorbar(im, cax=axcolor)

        if len(figname)>0:
            plt.savefig(figname)

        return fig 
        

    def plot_dendrogram(self, max_d):
        # 5. Selecting a Distance Cut-Off aka Determining the Number of Clusters
        # set cut-off to 50
        max_d = max_d # max_d as in max_distance
        plt.figure(figsize=(16,10))
        fancy_dendrogram(
            self.Lin,
            truncate_mode='lastp',
            p=100,
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,
            annotate_above=10,
            max_d=max_d,  # plot a horizontal cut-off line
        )
        #plt.savefig('/Users/xiaoxuanj/work/work_allen/DynamicBrain/figures/sessionB_ns_sg_matrix_cluster_dendrogram.pdf')
    
    def predict_cluster(self, **args):
        """
        predict number of clusters based on distance threshold or number of k.
        clusters start from 1
        """

        # 7. Retrieve the Clusters
        # 7.1 knowing max_d from dendrogram
        if 'max_d' in args.keys():
            self.clusters = sch.fcluster(self.Lin, args['max_d'], criterion='distance')
            np.unique(self.clusters)
        elif 'k' in args.keys():
            # knowing K
            self.clusters = sch.fcluster(self.Lin, args['k'], criterion='maxclust')
        return self.clusters

    def plot_clusters(self,Y):
        plt.figure(figsize=(12,8))
        plt.subplot(221)
        cmap = plt.cm.get_cmap('RdYlBu')
        plt.scatter(Y[:,0], Y[:,1], 20, c=cmap(0.9))
        plt.title('tSNE with 2D waveform',fontsize=16)

        # label could be RS/FS or depth
        labels = self.clusters-1 #clusters #mini.labels_ #ypos_all #new_type_all #ypos_all #RF_cluster[1]
        n_cluster = len(np.unique(labels))
        plt.subplot(222)
        cmap = plt.cm.get_cmap('spectral')
        #a = plt.scatter(Y[np.where(waveform_class=='fs')[0],0], Y[np.where(waveform_class=='fs')[0],1], 20, c=cmap(0.2))
        #b = plt.scatter(Y[np.where(waveform_class=='rs')[0],0], Y[np.where(waveform_class=='rs')[0],1], 20, c=cmap(0.4))
        #plt.legend((a,b), ('FS', 'RS'))
        for i in range(n_cluster):
            plt.plot(Y[np.where(labels==i)[0],0], Y[np.where(labels==i)[0],1], 20, c=cmap(1./n_cluster*i), label='group'+str(i))
        plt.legend(loc='upper left', numpoints=1, ncol=2, fontsize=10, bbox_to_anchor=(0, 0))

        #plt.savefig('/Users/xiaoxuanj/work/work_allen/DynamicBrain/figures/sessionB_ns_sg_matrix_cluster_colorplot.pdf')

class analyze_clustering(object):
	def __init__(self, X, n):
		"""X_pca/whitened data with sample*features.
		n upper limit of k values. """
		X_pca=np.array(X)
		self.K=range(1,n)
		self.KM = [KMeans(n_clusters=k).fit(X) for k in self.K]
		self.centroids = [k.cluster_centers_ for k in self.KM] # cluster centroids

		self.D_k = [cdist(X, cent, 'euclidean') for cent in self.centroids]
		cIdx = [np.argmin(D,axis=1) for D in self.D_k]
		dist = [np.min(D,axis=1) for D in self.D_k]
		self.avgWithinSS = [sum(d)/X.shape[0] for d in dist]

		# Total with-in sum of square
		self.wcss = [sum(d**2) for d in dist]
		self.tss = sum(pdist(X)**2)/X.shape[0]
		self.bss = self.tss-self.wcss

        # TODO: Add TSNE Embedded analysis

	def plot_elbow(self, kIdx):
        # TODO: Plot TSNE Embedded analysis

		# elbow curve
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(self.K, self.avgWithinSS, 'b*-')
		ax.plot(self.K[kIdx], self.avgWithinSS[kIdx], marker='o', markersize=12, 
		markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
		plt.grid(True)
		plt.xlabel('Number of clusters')
		plt.ylabel('Average within-cluster sum of squares')
		plt.title('Elbow for KMeans clustering')

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(self.K, self.bss/self.tss*100, 'b*-')
		ax.plot(self.K[kIdx], self.bss[kIdx]/self.tss*100, marker='o', markersize=12, 
		markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
		plt.grid(True)
		plt.xlabel('Number of clusters')
		plt.ylabel('Percentage of variance explained')
		plt.title('Elbow for KMeans clustering')


def run_and_plot_spectral_clustering(cov, LR_centers):
    plt.rcParams["figure.figsize"] = 15,7;count=2
    fig, axs = plt.subplots(2, 5, subplot_kw={'xticks': [], 'yticks': []})
    for i in range(2):
        for j in range(5):   
            model = SpectralBiclustering(n_clusters=count, random_state=0)
            model.fit(cov)
            fit_cov = cov[np.argsort(model.row_labels_)]
            fit_cov = fit_cov[:, np.argsort(model.row_labels_)]#column_labels_  // axs[i, j].imshow(fit_cov, cmap=plt.cm.Blues, vmin=0, vmax=30)
            axs[i, j].scatter(LR_centers[:,0],LR_centers[:,1], s=50,  c=model.row_labels_, cmap='viridis_r')
            count+=1
    plt.tight_layout()
    plt.tick_params(left = False)
    plt.savefig('raw_cluster', dpi = 100)
    plt.show()

def default_colors():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    return itertools.cycle(colors)

def plot_cluster_traces(traces, clust_ids, axes, swim_arr, dt=1.0, sel_ids=None, colors=None, title='', **fig_kwargs):
    """ Plot the mean traces for clusters
    :param traces:
    :param clust_ids:
    :param fig_kw:
    :return:
    """

    sel_ids = sel_ids or np.unique(clust_ids)
    colors = colors or default_colors()

    time = np.arange(traces.shape[1])*dt

    for i_ax, (i_clust, ax, col) in enumerate(zip(sel_ids, axes, colors)):
        sel = clust_ids == i_clust
        mn = np.mean(traces[sel, :], 0)
        sd = np.std(traces[sel, :], 0)
        ax.fill_between(time, mn - sd, mn + sd, color=(0.9, 0.9, 0.9))
        ax.plot(time, mn, color=col, **fig_kwargs)
        sns.despine(ax=ax, trim=True)
        if i_ax < len(axes)-1:
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
        ax.set_title(title)
    axes[-1].plot(time, swim_arr*20, color='black')
    axes[-1].spines['top'].set_visible(True)
    axes[-1].axes.get_yaxis().set_visible(False)

def _get_clusters(X, y):
    # TODO: Add to mutils
    s = np.argsort(y)
    return np.split(X[s], np.unique(y[s], return_index=True)[1][1:])

def plot_cluster_anatomy(clus_labels, LR_centers, XYZ, ax, **kwargs):
    clust_colors = default_colors()
    scatter_clus = []
    xx =  np.squeeze(XYZ[:,0])
    yy =  np.squeeze(XYZ[:,1])
    zz = np.squeeze(XYZ[:,2])
    # sorted_labels = clus_labels[np.argsort(clus_labels)]
    clustered_LR_centers = _get_clusters(LR_centers, clus_labels)

    # plt.rcParams["figure.figsize"] = 15,7;
    # count=2
    ax[0].scatter(xx, yy, lw=0, s=5, alpha=0.08, color=(0.3,) * 3)
    ax[1].scatter(yy, zz, lw=0, s=5, alpha=0.08, color=(0.3,) * 3)
    for clus_color, clus_n in zip(clust_colors,range(np.unique(clus_labels).shape[0])):
        # scatter_clus.append(color)
        # sorted_specific_labels = sorted_labels[sorted_labels==clus_n]
        # LR_centers_clus = LR_centers[np.argsort(clus_labels)[:sorted_specific_labels.shape[0]]]
        LR_centers_clus = clustered_LR_centers[clus_n]
        sc = ax[0].scatter(LR_centers_clus[:,0],LR_centers_clus[:,1], s=50,  color=clus_color,**kwargs)
        ax[1].scatter(LR_centers_clus[:,1],LR_centers_clus[:,2], s=50,  color=clus_color,**kwargs)
    plt.tick_params(left = False)
    return sc


def correlation_for_all_neurons(X):
  """Computes the connectivity matrix for the all neurons using correlations
    Args:
        X: the matrix of activities
    Returns:
        estimated_connectivity (np.ndarray): estimated connectivity for the selected neuron, of shape (n_neurons,)
  """
  n_neurons = len(X)
  S = np.concatenate([X[:, 1:], X[:, :-1]], axis=0)
  R = np.corrcoef(S)[:n_neurons, n_neurons:]
  return R

def plot_connectivity_matrix(A, ax=None):
  """Plot the (weighted) connectivity matrix A as a heatmap
    Args:
      A (ndarray): connectivity matrix (n_neurons by n_neurons)
      ax: axis on which to display connectivity matrix
  """
  if ax is None:
    ax = plt.gca()
  lim = np.abs(A).max()
  im = ax.imshow(A, vmin=-lim, vmax=lim, cmap="coolwarm")
  ax.tick_params(labelsize=10)
  ax.xaxis.label.set_size(15)
  ax.yaxis.label.set_size(15)
  cbar = ax.figure.colorbar(im, ax=ax, ticks=[0], shrink=.7)
  cbar.ax.set_ylabel("Connectivity Strength", rotation=90,
                     labelpad= 20, va="bottom")
  ax.set(xlabel="Connectivity from", ylabel="Connectivity to")


def generate_null_matrix(cov):
    plt.rcParams["figure.figsize"] = 15,7;count=2
    fig, axs = plt.subplots(2, 5, subplot_kw={'xticks': [], 'yticks': []})
    for i in range(2):
        for j in range(5):   
            model = SpectralBiclustering(n_clusters=count, random_state=0)
            model.fit(cov)
            fit_cov = cov[np.argsort(model.row_labels_)]
            fit_cov = fit_cov[:, np.argsort(model.row_labels_)]#column_labels_  // axs[i, j].imshow(fit_cov, cmap=plt.cm.Blues, vmin=0, vmax=30)
            axs[i, j].scatter(avg_coordinates[:,0],avg_coordinates[:,1], s=50,  c=model.row_labels_, cmap='viridis_r')
            count+=1
    plt.tight_layout()
    plt.tick_params(left = False)
    plt.savefig('raw_cluster', dpi = 100)
    plt.show()

# @jit
def ComputeSpearmanSelfNumba(Matrix):
    for row in np.arange(np.shape(Matrix)[0]):
            temp = Matrix[row, :].argsort()
            Matrix[row, temp] = np.arange(len(temp))
    SpearmanCoeffs = np.corrcoef(Matrix)
    return SpearmanCoeffs

# @jit
def ComputeSpearmanPairNumba(Matrix1, Matrix2):
    SpearmanCoeffs = np.zeros((np.shape(Matrix1)[0]))
    for row in np.arange(np.shape(Matrix1)[0]):
            temp = Matrix1[row, :].argsort()
            Matrix1[row, temp] = np.arange(len(temp))
            temp = Matrix2[row, :].argsort()
            Matrix2[row, temp] = np.arange(len(temp))
            SpearmanCoeffs[row] = np.cov(Matrix1[row, :], Matrix2[row, :])[0, 1] \
            / np.sqrt(np.var(Matrix1[row, :]) * np.var(Matrix2[row, :]))
    return SpearmanCoeffs

def ComputeSpearmanSelf(Matrix):
    SpearmanCoeffs = np.zeros((np.shape(Matrix)[0], np.shape(Matrix)[0]))
    for row1 in np.arange(np.shape(Matrix)[0]):
        for row2 in np.arange(row1, np.shape(Matrix)[0]):
            SpearmanCoeffs[row1, row2] = spearmanr(Matrix[row1, ], Matrix[row2])[0]
            SpearmanCoeffs[row2, row1] = SpearmanCoeffs[row1, row2]
    return SpearmanCoeffs
        
def ComputeSpearmanPair(Matrix1, Matrix2):
    SpearmanCoeffs = np.zeros((np.shape(Matrix1)[0]))
    for row in np.arange(np.shape(Matrix1)[0]):
        SpearmanCoeffs[row] = spearmanr(Matrix1[row, ], Matrix2[row])[0]
    return SpearmanCoeffs


def plot_corr(df,size=10):
    '''Plot a graphical correlation matrix for a dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    import matplotlib.pyplot as plt

    # Compute the correlation matrix for the received dataframe
    corr = df.corr()
    
    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap='RdYlGn')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    
    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)

def find_clusters(traces_data):
    """
    Finds clusters (Alterative function)
    """
    cluster_range = range(2, 13)
    scores = OrderedDict.fromkeys(cluster_range)
    for nc in cluster_range:
        fig = plt.figure(figsize=(10, 13))
        km = KMeans(n_clusters=nc)
        labels = km.fit_predict(traces_data)
        sort_ixs = np.argsort(labels)
        row_colors = map_labels_to_colors(labels[sort_ixs], 'tab20')
        f = sns.clustermap(traces_data[sort_ixs], row_colors=row_colors, row_cluster=False, col_cluster=False)
        f.ax_heatmap.set_title(f"n_clusters = {nc}")
        plt.show()

def calc_fc_matrix(data, todict=False):
    data  = data.astype('float32')
    xcorr = np.corrcoef(data).astype('float32')
    if not todict:
        return xcorr
    else:
        N, _ = xcorr.shape
        d = {n:[] for n in range(N)}
        for i in range(N):
            for j in range(i):
                d[i].append(xcorr[i,j])
        return d  
    
def get_hilbert(X):
    from scipy.signal import hilbert
    return hilbert(X)

def get_phases(X):
    return np.unwrap(np.angle(X))

def get_synch(phases):
    import cmath
    N = len(phases)
    mean_phase = np.mean(phases) # variance arosss
    synch = 1./float(N) * np.sum([cmath.exp(1j*(tp-mean_phase)) for tp in phases])
    return np.absolute(synch)


def get_distance(X, Y, norm=False):
    if len(X) != len(Y):
        return None
    distance = np.sqrt( np.sum((X-Y) ** 2))
    if norm:
        distance /= float(len(X))
    return distance

def pca_analysis(X, n_components=2, kernel='rbf',pca=None):
    from sklearn.decomposition import KernelPCA
    Xtransformed = None
    if pca is None:
        pca = KernelPCA(n_components=n_components, kernel=kernel)
        Xtransformed = pca.fit_transform(X.T).T
    else:
        try:
            Xtransformed = pca.transform(X.T).T
        except:
            raise Exception('passed pca object invalid..')
    return pca, Xtransformed
    
def data2percentile(data, method='average'):
    from scipy.stats import rankdata
    ranked_data = rankdata(data, method=method)
    return ranked_data / float(len(data))

def permute_weights(J, seed=0):
    newJ = np.zeros_like(J).astype('float32')
    srcs,dsts = np.nonzero(J)
    locs, weights = [], []
    for (src, dst) in list(zip(srcs,dsts)):
        locs.append((src,dst)); weights.append(J[src,dst])
        
    rs = np.random.RandomState(seed=seed)
    rs.shuffle(locs)
    for t, (src,dst) in enumerate(locs):
        newJ[src,dst] = weights[t]
        
    return newJ

def get_cutoff(J, percentile_cutoff):
    srcs, dsts = J.nonzero()
    weights_lst = []
    for (src, dst) in list(zip(srcs, dsts)):
        weights_lst.append(J[src,dst])
    cutoff_val = np.percentile(weights_lst, percentile_cutoff)
    return cutoff_val

def jthreshold(J, percentile_cutoff, above=True, pos=True, binarized=False, cutoff_val=None):
    Jcopy = deepcopy(J)
    if not pos:
        Jcopy *= -1
    
    Jcopy = np.clip(Jcopy, 0., np.max(Jcopy))
    if cutoff_val is None:
        cutoff_val = get_cutoff(Jcopy, percentile_cutoff)
    if above:
        Jcopy[Jcopy < cutoff_val]  = 0
        if binarized:
            Jcopy[Jcopy >= cutoff_val] = 1
        Jcopy = Jcopy.astype('float32')     
    else:
        ps,pd = np.where(Jcopy >= cutoff_val)
        Jcopy[ps,pd] = 0
        if binarized:
            zs,zd = np.where((Jcopy < cutoff_val))
            Jcopy[zs,zd]=1
        Jcopy = Jcopy.astype('float32')            
    Jcopy[np.diag_indices(Jcopy.shape[0])] = 0
    return Jcopy

def extract_weights(J, directed=True, nonzero=False):
    Jcopy = deepcopy(J)
    if not directed:
        Jcopy = np.triu(Jcopy)
      
    srcs, dsts = Jcopy.nonzero()
    srcdst = list(zip(srcs, dsts))
    weights_lst = [J[src,dst] for (src,dst) in srcdst]
    return np.asarray(weights_lst, dtype='float32')


def get_ei(w):
    w = np.asarray(w, dtype='float32')
    if len(w.nonzero()[0]) == 0: 
        return 0.0
    pos = np.sum(w[w > 0])
    neg = np.sum(w[w < 0])
    if neg == 0.0:
        return pos
    else:
        return abs(pos / neg)


def gaussian_mixture_modeling(ei, Nmax=11, minn=0., maxx=1.):
    from sklearn.mixture import GaussianMixture
    ei = np.asarray(ei, dtype='float32')    
    N = np.arange(1,Nmax)
    models = [None for i in range(len(N))]

    for i in range(len(N)):
        models[i] = GaussianMixture(N[i]).fit(ei.reshape(-1,1))
    model_AIC = [cm.aic(ei.reshape(-1,1)) for cm in models]


    model_bestM = models[np.argmin(model_AIC)]

    x = np.linspace(minn,maxx,1000)
    logprob = model_bestM.score_samples(x.reshape(-1,1))
    resp    = model_bestM.predict_proba(x.reshape(-1,1))
    pdf     = np.exp(logprob)
    pdf_individual = resp * pdf[:,np.newaxis]


    model_labels = model_bestM.fit_predict(ei.reshape(-1,1)).reshape(-1,)
    return x, pdf_individual, model_labels


def ensemble_detection(X, run_sam=False, **kwargs):
    import umap
    import hdbscan
    from sklearn.cluster import DBSCAN
    metric   = kwargs.get('metric', 'euclidean')
    k        = kwargs.get('k', 5)
    min_dist = kwargs.get('min_dist', 0.05)

    umap_data, cluster_labels, sam = None, None, None
    if run_sam:
        from SAM import SAM

        N, T = X.shape
        counts = (X, np.arange(T), np.arange(N))
        
        npcs = kwargs.get('npcs', 5)
        resolution = kwargs.get('resolution', 2.0)
        stopping_condition = kwargs.get('stopping_condition', 5e-4)
        max_iter = kwargs.get('max_iter', 25)
        
        sam  = SAM(counts)
        sam.run(verbose=False, projection='umap', k=k, npcs=npcs, preprocessing='Normalizer', distance=metric, 
                stopping_condition=stopping_condition, max_iter=max_iter,
                proj_kwargs={'metric': metric, 'n_neighbors': k, 'min_dist': min_dist})
        param = kwargs.get('resolution', 1.0)
        umap_data = sam.adata.obsm['X_umap']    
        sam.clustering(X=None, param=param, method='leiden')
        cluster_labels = sam.adata.obs['leiden_clusters']
        cluster_labels = [cluster_labels.iloc[i] for i in range(N)]
         
    else:

        umapy = umap.UMAP(n_components=2, min_dist=min_dist, n_neighbors=k)
        umap_data = umapy.fit_transform(X)
        
        clustering = hdbscan.HDBSCAN(min_cluster_size=5)
        cluster_labels = clustering.fit_predict(umap_data)
        
    return sam, umap_data, cluster_labels

def find_optimal_cutoff(Z, dmax, min_ensemble_sz=3):
    cutoff = 0.01
    found = False
    while not found:
        ind = sch.fcluster(Z, cutoff*dmax, 'distance')
        szs = get_ensemble_sz(ind)
        if np.min(szs) >= min_ensemble_sz: 
            found = True
        cutoff += 0.01  
    return cutoff*dmax

def hierarchical_clustering(X, k, metric='euclidean', method='ward'):
    import scipy.cluster.hierarchy as sch
    link = sch.linkage(X, metric=metric, method=method)
    inds   = sch.fcluster(link, k, 'maxclust')
    return inds
        
def get_single_IAAFT_surrogate(ts, n_iterations = 10, seed = None):
    """
    Returns single iterative amplitude adjusted FT surrogate.
    n_iterations - number of iterations of the algorithm.
    Seed / integer : when None, random seed, else fixed seed (e.g. for multivariate IAAFT surrogates).
    """

    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)

    xf = np.fft.rfft(ts, axis = 0)
    angle = np.random.uniform(0, 2 * np.pi, (xf.shape[0],))
    del xf

    return _compute_IAAFT_surrogates([ None, n_iterations, ts, angle])[-1]

def _compute_IAAFT_surrogates(a):
    i, n_iters, data, angle = a

    xf = np.fft.rfft(data, axis = 0)
    xf_amps = np.abs(xf)
    sorted_original = data.copy()
    sorted_original.sort(axis = 0)

    # starting point
    R = _compute_AAFT_surrogates([None, data, angle])[-1]

    # iterate
    for _ in range(n_iters):
        r_fft = np.fft.rfft(R, axis = 0)
        r_phases = r_fft / np.abs(r_fft)

        s = np.fft.irfft(xf_amps * r_phases, n = data.shape[0], axis = 0)

        ranks = s.argsort(axis = 0).argsort(axis = 0)
        R = sorted_original[ranks]

    return (i, R)

def _compute_AAFT_surrogates(a):
    i, data, angle = a

    # create Gaussian data
    gaussian = np.random.randn(data.shape[0])
    gaussian.sort(axis = 0)

    # rescale data
    ranks = data.argsort(axis = 0).argsort(axis = 0)
    rescaled_data = gaussian[ranks]

    # transform the time series to Fourier domain
    xf = np.fft.rfft(rescaled_data, axis = 0)
     
    # randomise the time series with random phases     
    cxf = xf * np.exp(1j * angle)
    
    # return randomised time series in time domain
    ft_surr = np.fft.irfft(cxf, n = data.shape[0], axis = 0)

    # rescale back to amplitude distribution of original data
    sorted_original = data.copy()
    sorted_original.sort(axis = 0)
    ranks = ft_surr.argsort(axis = 0).argsort(axis = 0)

    rescaled_data = sorted_original[ranks]
    
    return (i, rescaled_data)

# class RunningdJ(object):
#     def __init__(self,N):
#         self.N = N
#         self.dJ = np.empty_like((N,N), dtype='float32')
#     def clear(self):
#         self.J = np.empty_like((self.N, self.N), dtype='float32')        
    
#     def getJ(self):
#         J = np.empty_like((self.N, self.N), dtype='float32')
#         for (pos,idx) in enumerate(self.idxs):
#             J[idx,:] = self.J[pos]
#         return J
    
#     @classmethod
#     def combine(cls, a, br, idxs):
#         combined = cls(a.N)
#         combined.dJ[idxs] =     = a.J.extend(br)
#         return combined       
        
                
class RunningStats(object):

    def __init__(self):
        self.n = 0
        self.m1 = 0.

    def clear(self):
        self.n = 0
        self.m1 = 0.
       

    def update(self, x):
        n1 = self.n
        self.n += 1
        n = self.n
        delta = x - self.m1
        delta_n = delta / n
        self.m1 += delta_n

    def mean(self):
        return self.m1

    def variance(self):
        return self.m1 / (self.n - 1.0)

    def standard_deviation(self):
        return np.sqrt(self.variance())

    
    @classmethod
    def combine(cls, a, b):
        combined = cls()
        
        combined.n = a.n + b.n    
        combined.m1 = (a.n*a.m1 + b.n*b.m1) / combined.n;
        
        return combined

def k_means_classification():
    """
    Ref - https://github.com/AllenInstitute/swdb_2021/blob/2a09ed348cfe563c607800ce463a099ae1e3c903/DynamicBrain/Tutorials/T03_Classification_tutorial.ipynb
    Ref2 - https://github.com/DedeBrahma/Basic-ML-Fundamentals/blob/5574bb232cc211a81d12d00b1928dbb3efdfb317/KMeans/KMeans_Sklearn.ipynb

    """
def linear_regresssion():
    """
    Source: https://github.com/AllenInstitute/swdb_2021/blob/2a09ed348cfe563c607800ce463a099ae1e3c903/DynamicBrain/Tutorials/T01_Regression.ipynb
    Source 2: https://github.com/NGP-Bootcamp/Bootcamp2020

    """

def plot_correlograms():
    """
    Source: https://github.com/berenslab/neural_data_science/blob/main/notebooks/CodingLab3.ipynb
    Plot correlogramgs
    """

def PCA_Implement():
    """
    Source: 
    https://github.com/afronski/playground-courses/blob/master/scalable-machine-learning/lab-5/ML_lab5_pca_student.ipynb
    Learn about dimensionality reduction
    """


# def k_nearest_neighbour_classifier():
#     # Ref - https://github.com/Scott-Lab-QBI/Brainwide_auditory_processing/blob/master/NN_analysis.ipynb
#     for key, value in ROIS.items():
#     if key not in metadata: 
#         print("Adding 0 labels to {}:".format(key))
#         value = np.vstack((value, np.array(list(zero_to_add[key]),dtype='int32')))
        
#         # find optimal radius for RNC
#         print('Working on the', key)
#         neighbors_list = list(range(1,150,1))
#         cv_scores = [ ]
#         for K in neighbors_list:
#             if (key == 'Telencephalon' or key == 'TS')and K > 100:
#                 pass
#             else:
#                 knn =  KNeighborsClassifier(n_neighbors = K, weights='distance')
#                 scores = cross_val_score(knn, value[:,0:3], value[:,-1], cv=cv_split[key], scoring="accuracy")
#                 cv_scores.append(scores.mean())
#         # Changing to misclassification error
#         mse = [1-x for x in cv_scores]
#         # determing best k
#         optimal_k = neighbors_list[mse.index(min(mse))]
#         cv_results_dict.update({key : {'opt_k' : optimal_k, 'scores' : cv_scores}})
#         print("The optimal K is {}".format(optimal_k))
#         # print results and point out the optimal (red) in the plot as well as the chosen radius (green)
#         plt.plot(cv_scores)
#         plt.scatter(optimal_k-1,cv_scores[optimal_k-1], c='red', alpha=0.6)
#         plt.xlim([0,100])
#         plt.title(key)
#         plt.ylabel('Accuracy')
#         plt.xlabel('k neighbours')
#         plt.show()



## Regressors and correlation analysis
# # Prepare regressors?
# # Prepare regressors using cropped time series (100:200), corresponds to imaging stack series
# # Ref - https://github.com/optofish-paper/ZebrafishFunctionalMaps_LinearRegression/blob/master/FunctionalMaps_2016-07-26fish1(cyto).ipynb
# stack_ind_range = np.arange(100, 200)
# regressors = np.column_stack((
#     swim_fdrift_convolved[stack_ind_range]/swim_fdrift_convolved[stack_ind_range].std(),
#     f_drift_convolved[stack_ind_range]/f_drift_convolved[stack_ind_range].std(),
#     b_drift_convoled[stack_ind_range]/b_drift_convoled[stack_ind_range].std()))
# regressors.shape


# def design_manual_regressor():
#     # Ref: https://github.com/xiuyechen/FishExplorer/blob/master/GUI%20hack%20scripts/manualRegressor_regression.m
    

# def run_linear_regression():
#     # K-means to look for responses on ROIs:
#     # Convert to linear regression
#     model = FastLinearRegression(fit_intercept=True).fit(regressors, imaging_time_series)
#     results = model.betas_and_scores.toarray()
#     betas, r_squared = results[:, :, :, 1:4], results[:, :, :, 4]

#     # Convert (regression coeffs x R2) into weights
#     imMotorForw = (betas[:, :, :, 0] * r_squared)
#     imForw = (betas[:, :, :, 1] * r_squared)
#     imBackw = (betas[:, :, :, 2] * r_squared)


# # Explore response towards decision-making

# # Compute general correlation with localized data

# # Compute correlation matrices
# @jit
# def ComputeSpearmanSelfNumba(Matrix):
#     for row in np.arange(np.shape(Matrix)[0]):
#             temp = Matrix[row, :].argsort()
#             Matrix[row, temp] = np.arange(len(temp))
#     SpearmanCoeffs = np.corrcoef(Matrix)
#     return SpearmanCoeffs

