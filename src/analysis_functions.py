import pandas as pd 
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import itertools
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
import networkx as nx
from sklearn.decomposition import PCA

import sys
sys.path.append(".")

import clustering_fun as cf

def add_random_z_offset(df):
    """
    For each event_id, draw a random number in [0, 1000)
    and add it to all z values of that event.
    Always shifts by |z_min| to keep z positive.
    """
    df = df.copy()

    unique_ids = df["event_id"].unique()
    offsets = pd.Series(
        np.random.uniform(0, 1000, size=len(unique_ids)),
        index=unique_ids
    )

    df["z"] = df["z"] + df["event_id"].map(offsets)
    z_min = df["z"].min()
    df["z"] += abs(z_min)

    return df

def drop_events_with_negative_z(df):
    """Drop entire events if any point in that event has z < 0."""
    
    bad_events = df.loc[df["z"] < 0, "event_id"].unique()
    return df[~df["event_id"].isin(bad_events)].reset_index(drop=True)

def compute_el_ion_pairs(df, wval):
    """
    Calculate the number of e-ion pairs in a given point

    """
    df = df.copy()
    
    df["nel"] = np.random.poisson(df["energy"] / wval)
    
    return df


def compute_sigma_tr(z, D):
    """
    Given z and D, return sqrt(D^2 * z).
    """
    return np.sqrt(D**2 * z)

def compute_sigma_lon(z, D):
    """
    Given z and D, return sqrt(D^2 * z).
    """
    return np.sqrt(D**2 * z)



def clustercounts_smeared(df, diff_coeff, granularity):
    """
    For each event:
      - For all hits in the event:
          * compute sigma = compute_ionsigma_tr(z, D)
          * generate nel smeared (x, y) points
      - Merge all smeared hits for that event
      - Cluster
      - Count clusters
    Returns a dataframe with event_id and n_clusters.
    """
    results = []

    for event_id, event_hits in tqdm(df.groupby("event_id"), desc="Processing events", total=df["event_id"].nunique()):

        all_x, all_y = [], []
        
        for i, row in event_hits.iterrows():
            sigma = compute_ionsigma_tr(row["z"], diff_coeff)
            n = int(row["nel"])
            all_x.extend(np.random.normal(row["x"], sigma, n))
            all_y.extend(np.random.normal(row["y"], sigma, n))

        n_clusters = cf.clustercounts_kdtree_single_ev(all_x,all_y,granularity)

        results.append({"event_id": event_id, "n_clusters": n_clusters})

    return results

def save_hist2d(H, xedges, yedges, event_id, ncl, thr, outdir="plots2D"):
    """Save 2D histogram H as PNG heatmap with zero bins set to NaN."""
    H_plot = H.astype(float)
    H_plot[H_plot < thr] = np.nan  # make zero bins transparent

    plt.figure(figsize=(6, 5), dpi=120)
    plt.imshow(H_plot.T, origin="lower",
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               aspect="auto", cmap="viridis")
    plt.colorbar(label="Counts")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"2D histogram - event {event_id} - cl = {ncl}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/hist2d_event_{event_id:05d}.png")
    plt.close()

def clustercounts_smeared_histo(df, diff_coeff, granularity, thr, outfolder):
    """
    For each event:
      - Smear hits based on sigma(z, D)
      - Create 2D histogram with bin size = granularity
      - Cluster adjacent (nonzero) bins using KDTree nearest-neighbor
      - Count clusters
    Returns: list of dicts [{event_id, n_clusters}, ...]
    """
    results = []

    for event_id, event_hits in tqdm(df.groupby("event_id"), desc="Processing events", total=df["event_id"].nunique()):
        all_x, all_y = [], []

        # Smear all hits for this event
        for _, row in event_hits.iterrows():
            sigma = compute_ionsigma_tr(row["z"], diff_coeff)
            n = int(row["nel"])
            all_x.extend(np.random.normal(row["x"], sigma, n))
            all_y.extend(np.random.normal(row["y"], sigma, n))

        # Build 2D histogram grid
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        x_bins = np.arange(x_min, x_max + granularity, granularity)
        y_bins = np.arange(y_min, y_max + granularity, granularity)

        H, xedges, yedges = np.histogram2d(all_x, all_y, bins=[x_bins, y_bins])

        # Get coordinates of nonzero bins
        coords = np.argwhere(H > 0)

        # Cluster nonzero bins (nearest-neighbor)
        tree = cKDTree(coords)
        pairs = tree.query_pairs(1.5)  # adjacency in grid space

        parent = list(range(len(coords)))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for i, j in pairs:
            union(i, j)

        roots = [find(i) for i in range(len(coords))]
        unique, counts = np.unique(roots, return_counts=True)
        n_clusters = np.sum(counts > 10)
        
        # ✅ Save 2D histogram image every 200 events
        if (event_id%100 == 0):
            save_hist2d(H, xedges, yedges, event_id, n_clusters, thr, outfolder)
            
        results.append({"event_id": event_id, "n_clusters": n_clusters})

    return results


def clustercounts_smeared_histo_withpoints(df, diff_coeff, granularity, thr, outfolder):
    """
    For each event:
      - Smear hits based on sigma(z, D)
      - Create 2D histogram with bin size = granularity
      - Cluster adjacent (nonzero) bins using KDTree nearest-neighbor
      - Count clusters and record their sizes
    Returns: list of dicts [{event_id, n_clusters, cluster_sizes}, ...]
    """
    results = []

    for event_id, event_hits in tqdm(df.groupby("event_id"), desc="Processing events", total=df["event_id"].nunique()):
        all_x, all_y = [], []

        # Smear all hits for this event
        for _, row in event_hits.iterrows():
            sigma = compute_ionsigma_tr(row["z"], diff_coeff)
            n = int(row["nel"])
            all_x.extend(np.random.normal(row["x"], sigma, n))
            all_y.extend(np.random.normal(row["y"], sigma, n))

        # Build 2D histogram grid
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_bins = np.arange(x_min, x_max + granularity, granularity)
        y_bins = np.arange(y_min, y_max + granularity, granularity)
        H, xedges, yedges = np.histogram2d(all_x, all_y, bins=[x_bins, y_bins])

        # Get coordinates of nonzero bins
        coords = np.argwhere(H > 0)

        # Cluster nonzero bins (nearest-neighbor)
        tree = cKDTree(coords)
        pairs = tree.query_pairs(1.5)  # adjacency in grid space

        parent = list(range(len(coords)))
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for i, j in pairs:
            union(i, j)

        roots = [find(i) for i in range(len(coords))]
        unique, counts = np.unique(roots, return_counts=True)

        # Select clusters above threshold
        valid_mask = counts > 10
        n_clusters = np.sum(valid_mask)
        cluster_sizes = counts[valid_mask].tolist()

        # ✅ Save 2D histogram image every 200 events
        if event_id % 200 == 0:
            save_hist2d(H, xedges, yedges, event_id, n_clusters, thr, outfolder)

        results.append({
            "event_id": event_id,
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes
        })

    return results


def remove_Xray(df, energy_threshold=0.35):
    """
    For each event and cluster, remove clusters with total energy < threshold (keV).
    Expects columns: ['event_id', 'label', 'energy'].
    Returns filtered dataframe.
    """
    energy_sum = df.groupby(["event_id", "label"])["energy"].sum().reset_index()
    valid_clusters = energy_sum[energy_sum["energy"] >= energy_threshold]
    df_filtered = df.merge(valid_clusters[["event_id", "label"]], on=["event_id", "label"], how="inner")
    return df_filtered


def get_track_length_mst_V2(points):
    """
    Builds MST of the given points and returns the longest path (approximate 'track').
    """
    # Compute full distance matrix
    D = distance_matrix(points, points)

    # Build MST (undirected)
    mst = minimum_spanning_tree(D)
    mst = mst.toarray()

    # Build graph in NetworkX for easier traversal
    G = nx.Graph()
    for i in range(len(points)):
        for j in range(len(points)):
            if mst[i, j] > 0:
                G.add_edge(i, j, weight=mst[i, j])

    # Find all pairs shortest paths on MST (efficient in sparse graphs)
    lengths = dict(nx.all_pairs_dijkstra_path_length(G))

    # Find two most distant nodes (the “diameter”)
    max_len, endpoints = 0, (0, 0)
    for i in lengths:
        for j in lengths[i]:
            if lengths[i][j] > max_len:
                max_len = lengths[i][j]
                endpoints = (i, j)

    # Reconstruct the full path
    path = nx.shortest_path(G, source=endpoints[0], target=endpoints[1], weight='weight')

    return path, max_len

def compute_z_extent_diff(df, coeff):
    zmin = df["z"].min()
    zmax = df["z"].max()
    sig_zmin = compute_sigma_lon(zmin, coeff)
    sig_zmax = compute_sigma_lon(zmax, coeff)
    return (zmax + 3 * sig_zmax) - (zmin - 3 * sig_zmin)

def apply_3d_gaussian_smearing(df, a, b, seed=None):
    """
    Replace each point by a random 3D point drawn from a Gaussian:
      - sigma_x = sigma_y = sqrt(a^2 * z)
      - sigma_z = sqrt(b^2 * z)
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ['x','y','z'].
    a : float
        XY scaling factor.
    b : float
        Z scaling factor.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    points_smeared : np.ndarray
        Array of shape (N,3) with smeared points.
    """
    
    points = df[['x','y','z']].to_numpy()
    N = points.shape[0]
    points_smeared = np.empty_like(points)
    
    for i in range(N):
        x0, y0, z0 = points[i]
        sigma_xy = np.sqrt(a**2 * z0)
        sigma_z  = np.sqrt(b**2 * z0)
        x_new = np.random.normal(loc=x0, scale=sigma_xy)
        y_new = np.random.normal(loc=y0, scale=sigma_xy)
        z_new = np.random.normal(loc=z0, scale=sigma_z)
        points_smeared[i] = [x_new, y_new, z_new]
    
    return points_smeared

def ellipse_axes_2D(df,a,b):
    pca = PCA(n_components=2)
    points = apply_3d_gaussian_smearing(df,a,b)
    pts2 = pca.fit_transform(points)   # project to 2D

    # covariance in 2D
    cov = np.cov(pts2.T)
    eigvals, _ = np.linalg.eig(cov)

    # ellipse semi-axes = sqrt(eigenvalues)
    axes = np.sqrt(np.sort(eigvals)[::-1])
    return axes  # major, minor

def ellipse_axes_2D_plot(points):
    pca = PCA(n_components=2)
    pts2 = pca.fit_transform(points)

    cov = np.cov(pts2.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    a, b = np.sqrt(eigvals)   # ellipse semi-axes
    return pca, a, b, eigvecs
