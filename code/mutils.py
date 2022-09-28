import numpy as np
from copy import deepcopy
from pathlib import Path
import numpy.ma as ma
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import h5py
import json
from scipy.stats import skew, kurtosis, zscore
import pandas as pd
import seaborn as sns
import itertools 

# 

def read_data(fish_id, master_path):
    data_path = master_path / fish_id
    traces_path = data_path / "data_from_suite2p_cells.h5"
    filename = traces_path
    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        print("Keys: %s" % f.keys())

        # Get the relevant group keys
        traces_group_key = 'traces'
        coords_group_key = 'coords'

        # get the object type for a_group_key: usually group or dataset
        # print(type(f[traces_group_key]))
        # print(type(f[coords_group_key]))  

        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        # data = list(f[traces_group_key])

        # If a_group_key is a dataset name, 
        # this gets the dataset values and returns as a list
        # data = list(f[traces_group_key])
        
        # preferred methods to get dataset values:
        # ds_obj = f[a_group_key]      # returns as a h5py dataset object
        trace_all = f[traces_group_key][()]  # returns as a numpy array
        ROIs_all = f[coords_group_key][()]  # returns as a numpy array
    # Load metadata as well: 
    # This is mainly to get scanning frequency and lag condition
    with open(next(data_path.glob("*metadata.json")), "r") as f:
        metadata = json.load(f)
    zf = metadata["imaging"]["microscope_config"]["lightsheet"]["scanning"]["z"]["frequency"]
    lsconfig = metadata["imaging"]["microscope_config"]['lightsheet']['scanning']
    z_tot_span = lsconfig["z"]["piezo_max"] - lsconfig["z"]["piezo_min"]
    n_planes = lsconfig["triggering"]["n_planes"]
    z_res = z_tot_span / n_planes
    protocol_params = metadata["stimulus"]["protocol"]["E0040_motions_cardinal"]["v20_cl_gainmod"]

    return trace_all, ROIs_all, zf, protocol_params, z_res

def plot_heatmap(traces, ax, dt=1.0, 
                         **kwargs):
    """ Plot a heatmap of traces separated,    
    :param traces: The fish neuronal traces to plot
    :param ax: The ax object needed to plot
    :param dt: the time difference constant
    :param kwargs: any other parameters that may be needed
    :return:
    """
    n_rois = traces.shape[0]
    time_imaging_trial = np.arange(traces.shape[1])*dt
    im = ax.imshow(traces,
         aspect='auto',
        extent=[0, time_imaging_trial[-1], n_rois, 0],
              )
    ax.set_yticks([])
    ax.set_title('Traces for condition: {}'.format(kwargs['plot_title']))
    ax.set_ylabel('Neuronal trace')
    ax.set_xlabel('Time (s)')
    return im

def subsample_data(traces_data, sample_size = 100):
    """
    Acquires a subsample of the Neuron dataset for analysis and clusterings
    This function samples a set of neurons without replacement.  
    Inputs:
        traces_data: Neuronal data from imaging, should be in numpy format
    Returns:
        rand_ix (array-like):
            Array containing the chosen indices
        sample_neurons (array-like ):
            Array with shape (sample_size, neuron_data.shape[1])
            containing a subset of the neuron traces. 
    """
    # Get random indices sampling without replacement
    rand_ix = np.random.choice(
        np.arange(traces_data.shape[0]), size= sample_size, replace=False
    )
    # Get subsample by choosing indices along rows
    sample_neurons = traces_data[rand_ix, :]

    return rand_ix, sample_neurons 


def _ridge_plot(array, 
               times,
               trace_spacing=5,
               ytick_spacing=10,
               title=None,
               color='black',
               alpha=0.5,
               line_width=1, 
               ax = None):
    """
    Plot ridge plot of all rows in array, given x values.
    Inputs:
        array: num_components x num_times array of traces
        times: 1-d array 
        trace_spacing (scalar): distance between each plot on y axis
        ytick_spacing (int): period between ytick labels (every ytick_spacing traces)
        color: line color (r,g,b) or standard matplotlib color string
        alpha (scalar): alpha for each trace, if you want them more see-through when density is high.
        line_width (scalar): how wide?
        ax: axes object if you want to draw in pre-existing axes (None if you want new axes)
    
    Outputs:
        ax: axes object with lines drawn
    """
    num_traces = array.shape[0]
    num_yticks = int(1+num_traces//ytick_spacing)
    
    # set y position of each trace
    y_position_traces = np.linspace(0, num_traces*trace_spacing, num=num_traces) 
    
    # set y tick properties 
    y_ticks = np.linspace(0, (num_traces-1)*trace_spacing, num=num_yticks)
    y_tick_labels = np.arange(0, num_traces+2*ytick_spacing, ytick_spacing, dtype=np.uint8) # +2*y_tick_spacing just for insurance
    y_tick_labels = y_tick_labels[:num_yticks]
    
    if ax is None:
        f, ax = plt.subplots()
    for ind, line in enumerate(array):
        ax.plot(times, 
                line+y_position_traces[ind], 
                color=color,
                alpha=alpha,
                linewidth=line_width)
    # only show left/bottom axis lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ylims = ax.get_ylim()
    # set ylimits to make it pretty (this could use some tweaking probably)
    ax.set_ylim(0.1*ylims[0], ylims[1]-0.05*ylims[1])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    plt.autoscale(enable=True, axis='x', tight=True)
    return ax
    print('done')

def plot_neuronal_ridgeplot(denoised_traces, select_components, frame_rate, traces_ax=None,  title=None):
    if traces_ax is None:
        traces_ax = plt.gca()
    denoised_traces = denoised_traces[1:select_components]
    # num_components = denoised_traces.shape[0]
    num_samples = denoised_traces.shape[1]
    sampling_pd = 1/frame_rate
    frame_times = np.arange(0, sampling_pd*num_samples, sampling_pd)
    component_spacing = int(np.max(denoised_traces)*1)
    print(f"Traces will be spread {component_spacing} units apart")

    traces_ax = _ridge_plot(denoised_traces, 
                        frame_times, 
                        color='black',
                        trace_spacing=component_spacing, 
                        ytick_spacing=5,
                        alpha=0.6,
                        line_width=0.75,
                        ax=traces_ax)
    traces_ax.set_xlabel("Time (s)", fontsize=12)
    traces_ax.set_ylabel("Component #", fontsize=12)
    traces_ax.set_title("Ridge Plot for condition: {}".format(title), fontsize=16)
    plt.tight_layout()
    # plt.savefig('ridge_plot.png',bbox_inches='tight')

def examine_explained_variance(model, n_com, var_ax=None, title=None):
    '''
    The plots the amount of variance explained by select dimentionality reduced by plotting the cumulative explained variance by the main PCs alongside components
    Inputs:
    - Model
    - num_comp_arr
    Returns: 
     None
    '''
    if var_ax is None:
        var_ax = plt.gca()
    num_comp_arr=np.arange(0,n_com ,1)
    # Calculate explaiend variacnce
    expl_var=np.cumsum(model.explained_variance_ratio_) 
    var_ax.plot(num_comp_arr, expl_var)
    var_ax.set_title("Explained variance for condition: {}".format(title), fontsize=16)
    var_ax.set_xlabel('PCs')
    var_ax.set_ylabel('Explained variance')
    var_ax.grid()

def extract_features(Traces):
    """
    Extract various feature of interest from extracted ROI's
    Input:
        - Traces: A segmented array of traces according to their laps
        - Threshold
    Output: 
        The function will output the following statistical measures for different traces
        - n_peaks: Number of peaks
        - width.median: Width of median
        - height.median: Height of median
        - Decay_time: The decay time of signal
        - width.sum
        - skewness
        - Integral
        - Baseline
        - Number of active neurons
        - Correlation with stimulus
        - Absolute correlation
        - Baseline (baseline change)
        - Median of baselines
        - Kurtosis
    """
    neuronID = ma.arange(len(Traces))
    # quarterID = np.full(len(Traces),count+1)
    # Extract peaks and peaks properties
    num_cells = np.shape(Traces)[0]
    # n_peaks = np.zeros((num_cells))
    # height = np.zeros((num_cells))
    # width = np.zeros((num_cells))
    averages = np.zeros((num_cells))
    medians = np.zeros((num_cells))
    stds = np.zeros((num_cells))
    mins = np.zeros((num_cells))
    maxs = np.zeros((num_cells))
    skewness = np.zeros((num_cells))
    kurtosis_arr = np.zeros((num_cells))

    # for lap in lapID:
    #     Traces = segmented_movie[lap ]
    for neuron in range(num_cells):
        averages[neuron] = ma.mean(Traces[neuron], axis=0)
        medians[neuron] = ma.median(Traces[neuron], axis=0)
        stds[neuron] = ma.std(Traces[neuron], axis=0)
        mins[neuron] = ma.min(Traces[neuron], axis=0)
        maxs[neuron] = ma.max(Traces[neuron], axis=0)
        skewness[neuron] = skew(Traces[neuron],axis=0)
        kurtosis_arr[neuron] = kurtosis(Traces[neuron], axis=0)
    # Buildd the data frame and return it:
    df_estimators = pd.DataFrame({
                                "neuronID":neuronID,
                                "trace.average": averages,
                                "trace.median":medians,
                                "trace.std":stds,
                                "trace.min":mins,
                                "trace.max":maxs,
                                "trace.skewness":skewness,
                                "trace.kurtosis":kurtosis_arr,                  
                            })
    return df_estimators

def plot_boxplot_features(df_estimators, headers):
    f, axes = plt.subplots(1, len(headers), figsize=(16, 12)) # sharex=Truerex=True
    sns.despine(left=True)
    for i, header in enumerate(headers):
        sns.boxplot(y=header, data=df_estimators, ax=axes[i])
        # axes[i].set_ylim([0.0,1200.0])
        axes[i].legend(loc='upper right')
        if i > 0:
            axes[i].get_legend().remove()
        axes[i].set_title(str(header),fontsize='14')
        axes[i].xaxis.label.set_visible(False)
        axes[i].yaxis.label.set_visible(False)
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.4,
                            wspace=0.35)
    plt.savefig('box_plot.png',bbox_inches='tight')
    plt.show()
    return f

def default_colors():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    return itertools.cycle(colors)

def plot_cluster_heatmap(traces, clust_ids, ax, dt=1.0, show_cluster_borders=True,
                         clust_separator_color="r",
                         clust_colors=None,
                         clust_colorbar_width=30,
                         **kwargs):
    """ Plot a heatmap of traces separated by clusters, drawing the separators
    and color-coding the clusters
    :param traces:
    :param clust_ids:
    :param ax:
    :param dt:
    :param show_cluster_borders:
    :param clust_separator_color:
    :param clust_colors:
    :param clust_colorbar_width:
    :param kwargs:
    :return:
    """
    order_clust = np.argsort(clust_ids)
    clust_borders = np.nonzero(np.diff(clust_ids[order_clust]))[0]
    n_rois = traces.shape[0]
    time_imaging_trial = np.arange(traces.shape[1])*dt
    im = ax.imshow(traces[order_clust],
         aspect='auto',
        extent=[0, time_imaging_trial[-1], n_rois, 0],
              **kwargs,
              )
    clust_borders_all = np.r_[0, clust_borders, len(order_clust)]
    if show_cluster_borders:
        ax.set_yticks(
            clust_borders_all[1:] - np.diff(clust_borders_all) / 2)
        # ax.set_yticklabels([str(dif) for dif in np.diff(clust_borders_all)])
        ax.set_yticklabels([str(id) for id in np.unique(clust_ids)])
    else:
        ax.set_yticks([])
    
    for cb in clust_borders:
        ax.axhline(cb, color=clust_separator_color, lw=0.5)

    prev_border = 0
    clust_colors = clust_colors or default_colors()

    for cb, color in zip(np.r_[clust_borders, n_rois], clust_colors):
        pt = ax.add_patch(Rectangle((time_imaging_trial[-1], prev_border),
                                    clust_colorbar_width, cb-prev_border,
                                    facecolor=color))
        pt.set_clip_box(None)
        prev_border = cb

    return im

def calc_fc_matrix(data, todict=False):
    # Adapt to our fc matrix calculation
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
    """
    A function which obtains the distance between coordinates in either normalized and non-normalized forms
    """
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
    '''
    Converts matrix of values to their percentiles using differnet methods
    Inputs:
     - Data: A matrix of functional connectivity
     - Method: Percentile method computation 
    Output:
     - 
    '''
    from scipy.stats import rankdata
    
    # Flatten the data
    ranked_data = rankdata(data, method=method)
    
    # Calculate percentile off
    
  
    # Reshape onto square matrix    
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


def filterRegionResponse(region_response, cutoff=None, fs=None):
    """
    Low pass filter region response trace.
    :region_response: np array
    :cutoff: Hz
    :fs: Hz
    """
    if fs is not None:
        sos = signal.butter(1, cutoff, 'hp', fs=fs, output='sos')
        region_response_filtered = signal.sosfilt(sos, region_response)
    else:
        region_response_filtered = region_response

    return region_response_filtered


def trimRegionResponse(file_id, region_response, start_include=100, end_include=None):
    """
    Trim artifacts from selected brain data.
    Dropouts, weird baseline shifts etc.
    :file_id: string
    :region_response: np array
        either:
            nrois x frames (region responses)
            1 x frames (binary behavior response)
    :start_include: beginning timepoint of trace
    :end_include: end timepoint of trace
    """
    # Key: brain file id
    # Val: time inds to include
    brains_to_trim = {'2018-10-19_1': np.array(list(range(100, 900)) + list(range(1100, 2000))), # transient dropout spikes
                      '2017-11-08_1': np.array(list(range(100, 1900)) + list(range(2000, 4000))), # baseline shift
                      '2018-10-20_1': np.array(list(range(100, 1000)))} # dropout halfway through

    if file_id in brains_to_trim.keys():
        include_inds = brains_to_trim[file_id]
        if len(region_response.shape) == 2:
            region_response_trimmed = region_response[:, include_inds]
        elif len(region_response.shape) == 1:
            region_response_trimmed = region_response[include_inds]
    else: # use default start / end
        if len(region_response.shape) == 2:
            region_response_trimmed = region_response[:, start_include:end_include]
        elif len(region_response.shape) == 1:
            region_response_trimmed = region_response[start_include:end_include]

    return region_response_trimmed


def getProcessedRegionResponse(resp_fp, cutoff=None, fs=None):
    """
    Filter + trim region response.
    :resp_fp: filepath to response traces
    :cutoff: highpass cutoff (Hz)
    :fs: sampling frequency (Hz)
    """
    file_id = resp_fp.split('/')[-1].replace('.pkl', '')
    region_responses = pd.read_pickle(resp_fp)

    resp = filterRegionResponse(region_responses.to_numpy(), cutoff=cutoff, fs=fs)
    resp = trimRegionResponse(file_id, resp)

    region_responses_processed = pd.DataFrame(data=resp, index=region_responses.index)
    return region_responses_processed


def computeRegionResponses(brain, region_masks):
    """
    Get region responses from brain and list of region masks.
    :brain: xyzt array
    :region_masks: list of xyz mask arrays
    """
    region_responses = []
    for r_ind, mask in enumerate(region_masks):
        region_responses.append(np.mean(brain[mask, :], axis=0))

    return np.vstack(region_responses)


def getCmat(response_filepaths, include_inds, name_list):
    """Compute fxnal corrmat from response files.
    :response_filepaths: list of filepaths where responses live as .pkl files
    :include_inds: list of ROI number IDs to select
    :name_list: associated list of ROI names
    """
    cmats_z = []
    for resp_fp in response_filepaths:
        tmp = getProcessedRegionResponse(resp_fp, cutoff=0.01, fs=1.2)
        resp_included = tmp.reindex(include_inds).to_numpy()

        correlation_matrix = np.corrcoef(resp_included)

        np.fill_diagonal(correlation_matrix, np.nan)
        # fischer z transform (arctanh) and append
        new_cmat_z = np.arctanh(correlation_matrix)
        cmats_z.append(new_cmat_z)

    # Make mean pd Dataframe
    mean_cmat = np.nanmean(np.stack(cmats_z, axis=2), axis=2)
    np.fill_diagonal(mean_cmat, np.nan)
    CorrelationMatrix = pd.DataFrame(data=mean_cmat, index=name_list, columns=name_list)

    return CorrelationMatrix, cmats_z


def getMeanBrain(filepath):
    """Return time-average brain as np array."""
    meanbrain = np.asanyarray(nib.load(filepath).dataobj).astype('uint16')
    return meanbrain


def loadAtlasData(atlas_path, include_inds, name_list):
    """
    Load region atlas data.
    :atlas_path: fp to atlas brain
    :include_inds: list of ROI number IDs to select
    :name_list: associated list of ROI names
    """
    mask_brain = np.asarray(np.squeeze(nib.load(atlas_path).get_fdata()), 'uint16')

    roi_mask = []
    for r_ind, r_name in enumerate(name_list):
        new_roi_mask = np.zeros_like(mask_brain)
        new_roi_mask = mask_brain == include_inds[r_ind] # bool
        roi_mask.append(new_roi_mask)

    return roi_mask


def getRegionGeometry(atlas_path, include_inds, name_list):
    """
    Return atlas region geometry, size etc.
    :atlas_path: fp to atlas brain
    :include_inds: list of ROI number IDs to select
    :name_list: associated list of ROI names
    """
    roi_mask = loadAtlasData(atlas_path, include_inds, name_list)

    roi_size = [x.sum() for x in roi_mask]

    coms = np.vstack([center_of_mass(x) for x in roi_mask])

    # calulcate euclidean distance matrix between roi centers of mass
    dist_mat = np.zeros((len(roi_mask), len(roi_mask)))
    dist_mat[np.triu_indices(len(roi_mask), k=1)] = pdist(coms)
    dist_mat += dist_mat.T # symmetrize to fill in below diagonal

    DistanceMatrix = pd.DataFrame(data=dist_mat, index=name_list, columns=name_list)

    # geometric mean of the sizes for each pair of ROIs
    sz_mat = np.sqrt(np.outer(np.array(roi_size), np.array(roi_size)))
    SizeMatrix = pd.DataFrame(data=sz_mat, index=name_list, columns=name_list)

    return coms, roi_size, DistanceMatrix, SizeMatrix

        
                
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