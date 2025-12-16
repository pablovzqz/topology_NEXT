import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN

def clusterize_hits(df_pe_peak: pd.DataFrame, eps=2.3, npt=5)-> pd.DataFrame:

    """
    Cluster hits in 3D space for each event using DBSCAN.
    
    The coordinates are scaled to account for detector geometry differences 
    in samplig 
    
    Parameters
    ----------
    df_pe_peak : pd.DataFrame
    DataFrame containing hit information with columns 'X', 'Y', 'Z', and 'event'.
    
    Returns
    -------
    pd.DataFrame
    Modified DataFrame with an added 'cluster' column indicating the cluster label 
    for each hit (-1 for noise).
    """
    
    a = 14.55  # XY scale
    b = 3.7  # Z scale

    # Pre-allocate array for cluster labels
    cluster_labels = np.full(len(df_pe_peak), -9999, dtype=int)

    # Get values once (faster than repeatedly accessing DataFrame columns)
    coords = df_pe_peak[['X', 'Y', 'Z']].to_numpy()
    events = df_pe_peak['event'].to_numpy()
    
    # Use np.unique to get sorted event IDs
    unique_events = np.unique(events)
    
    for event_id in unique_events:
        mask = (events == event_id)
        X = coords[mask].copy()
        
        # Scale
        X[:, :2] /= a
        X[:, 2] /= b
        
        labels = DBSCAN(eps=eps, min_samples=npt).fit_predict(X)
        cluster_labels[mask] = labels

    df_pe_peak['cluster'] = cluster_labels

    return df_pe_peak


