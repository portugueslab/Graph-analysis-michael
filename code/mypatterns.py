import numpy as np 


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing

from scipy import signal
from scipy.stats import zscore



def clean(signals, detrend = True, sampling_frq = 2, highpass = 1/300,order=3):
    
    if detrend:
        signals = signal.detrend(signals)
        
    if highpass is not None:
        sampling_rate = 1./ sampling_frq 
        nyq = sampling_rate * 0.5
        wn = highpass / float(nyq)
        b, a = signal.butter(order,wn,'high')
        signal.filtfilt(b, a, signals, method='gust')
        signals = zscore(signals,axis=-1)           
    
    return signals.T




def select_frames(signals, method = None, time_mask=None, seed=0, thresh = 1 ):
    
    if  method  == 'REG':
            selected_frames = np.argwhere(time_mask)
            signals = signals[np.squeeze(selected_frames),:]
    elif method == 'SEED':
            time_mask ==  X[:,seed] > thresh
            selected_frames = np.argwhere(time_mask)
            signals = signals[np.squeeze(selected_frames),:]
            

    return signals, selected_frames


def get_patterns(X, nb_clusters=6, randsta=0, normalize = True ):
    
    if normalize is not None:
        X = preprocessing.normalize(X)

    clusterer = KMeans(n_clusters=nb_clusters, random_state=0)
    cluster_labels = clusterer.fit_predict(X)
    centers = clusterer.cluster_centers_

    
    return centers, cluster_labels, nb_clusters



