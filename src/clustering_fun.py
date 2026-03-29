import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

"""
Clustering algorithm based on kdtree
"""
def cluster_kdtree(df, d):
    """
    Cluster points per event using KD-tree + union-find.
    
    df: pandas DataFrame with columns ['x','y','z','event_id']
    d: distance threshold for clustering
    Returns: DataFrame with new 'label' column
    """
    result_list = []
    event_ids = df["event_id"].unique()

    # Iterate over events with a progress bar
    for ev in tqdm(event_ids, desc="Clustering events"):
        df_ev = df[df.event_id == ev]
        coords = df_ev[['x','y','z']].to_numpy()
        tree = cKDTree(coords)
        pairs = tree.query_pairs(d)  # only close pairs
        n = len(coords)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        # Union connected pairs
        for i, j in pairs:
            union(i, j)

        roots = [find(i) for i in range(n)]
        unique = {r: idx for idx, r in enumerate(sorted(set(roots)))}
        labels = [unique[r] for r in roots]

        df_ev = df_ev.copy()
        df_ev['label'] = labels
        result_list.append(df_ev)

    return pd.concat(result_list, ignore_index=True)


def clustercounts_kdtree_single_ev(x, y, d):
    """
    Cluster (x, y) points for a single event using KD-tree + union-find.
    Returns the number of clusters.
    """
    coords = np.column_stack((x, y))
    tree = cKDTree(coords)
    pairs = tree.query_pairs(d)

    n = len(coords)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i, j in pairs:
        union(i, j)

    roots = np.array([find(i) for i in range(n)])
    n_clusters = len(np.unique(roots))

    return n_clusters

def clusterize_hits_sophr(df_pe_peak: pd.DataFrame, eps=2.3, npt=5)-> pd.DataFrame:

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