import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm
import pandas as pd

def find_tree_extremities(df):
    """
    df: DataFrame with columns 'X','Y','Z'
    returns: two tuples (X,Y,Z) corresponding to the extremities of the MST
    """
    # Convert DataFrame to Nx3 array
    points = df[['x','y','z']].to_numpy()

    # Build distance matrix
    dist_matrix = squareform(pdist(points))
    
    # Build Minimum Spanning Tree (MST)
    mst = minimum_spanning_tree(dist_matrix)
    mst = mst.toarray()  # dense array
    
    # Compute all-pairs shortest paths along MST
    dist_along_tree = dijkstra(mst, directed=False)
    
    # Find indices of points with maximum distance along the tree
    i, j = np.unravel_index(np.argmax(dist_along_tree), dist_along_tree.shape)
    
    # Return X,Y,Z coordinates as tuples
    return tuple(points[i]), tuple(points[j])

def build_extremities_df(df):
    """
    df: original DataFrame with columns ['event_id', 'x','y','z', ...]
    returns: new DataFrame with columns ['event_id','p1','p2']
    """
    results = []

    # Group by event_id
    for event_id, group in tqdm(df.groupby('event_id'), desc="Processing events"):
        p1, p2 = find_tree_extremities(group)
        results.append({
            'event_id': event_id,
            'p1': p1,
            'p2': p2
        })
    
    return pd.DataFrame(results)

def build_2d_histogram(df, DL, DT):
    """
    df: DataFrame with columns ['x','y','z','nel']
    DL, DT: parameters to compute bin sizes
    returns: histogram (3D numpy array), edges (x_edges, y_edges, z_edges)
    """
    # Compute bin sizes based on mean Z
    print(DT**2 , df['z'].mean())
    bin_xy = np.sqrt(DT**2 * df['z'].mean())

    # Histogram min/max edges
    # Add 5% padding to min/max, accounting for possible negative ranges
    x_range = df['x'].max() - df['x'].min()
    y_range = df['y'].max() - df['y'].min()
    x_pad = 0.1 * x_range
    y_pad = 0.1 * y_range
    x_min = df['x'].min() - x_pad
    x_max = df['x'].max() + x_pad
    y_min = df['y'].min() - y_pad
    y_max = df['y'].max() + y_pad

    # Build bin edges using computed bin sizes
    x_edges = np.arange(x_min, x_max , 0.150)
    y_edges = np.arange(y_min, y_max , 0.150)
    
    # Initialize histogram
    hist = np.zeros((len(x_edges)-1, len(y_edges)-1))
    
    # Loop over points
    for _, row in df.iterrows():
        x0, y0, z0 = row['x'], row['y'], row['z']
        nel = int(row['nel'])
        
        # Gaussian sigmas
        sigma_xy = np.sqrt(DT**2 * z0)
        
        # Sample points
        xs = np.random.normal(loc=x0, scale=sigma_xy, size=nel)
        ys = np.random.normal(loc=y0, scale=sigma_xy, size=nel)
        
        # Fill histogram
        H, _ = np.histogramdd(np.stack([xs, ys], axis=1),
                              bins=[x_edges, y_edges])
        hist += H
    
    return hist, (x_edges, y_edges)

def build_2d_bool_hist(df, DT, threshold=1):
    """
    Memory-friendly 2D histogram in the XY plane with thresholding.
    A bin is True only if it collects at least `threshold` simulated hits.
    """

    if threshold < 1:
        raise ValueError("threshold must be >= 1")

    bin_xy = np.sqrt(DT**2 * df['z'].mean())

    x_range = df['x'].max() - df['x'].min()
    y_range = df['y'].max() - df['y'].min()
    x_pad = 0.1 * x_range
    y_pad = 0.1 * y_range
    x_min, x_max = df['x'].min() - x_pad, df['x'].max() + x_pad
    y_min, y_max = df['y'].min() - y_pad, df['y'].max() + y_pad

    x_edges = np.arange(x_min, x_max, bin_xy)
    y_edges = np.arange(y_min, y_max, bin_xy)

    Nx = len(x_edges) - 1
    Ny = len(y_edges) - 1

    hist_counts = np.zeros((Nx, Ny), dtype=np.uint32)

    for _, row in df.iterrows():
        x0, y0, z0 = row['x'], row['y'], row['z']
        nel = np.random.poisson(lam=row['nel'])

        sigma_xy = np.sqrt(DT**2 * z0)

        xs = np.random.normal(loc=x0, scale=sigma_xy, size=nel)
        ys = np.random.normal(loc=y0, scale=sigma_xy, size=nel)

        xi = np.digitize(xs, x_edges) - 1
        yi = np.digitize(ys, y_edges) - 1

        valid = (
            (xi >= 0) & (xi < Nx) &
            (yi >= 0) & (yi < Ny)
        )
        xi, yi = xi[valid], yi[valid]

        np.add.at(hist_counts, (xi, yi), 1)

    hist_bool = hist_counts >= threshold

    return hist_bool, x_edges, y_edges

def build_2d_counts_df(df, DT):
    """
    df: DataFrame with columns ['x','y','z','nel']
    DT: parameter for Gaussian sigma
    returns: DataFrame with columns ['x','y','counts'], only counts > 0
    """
    # Compute bin sizes based on mean Z
    bin_xy = np.sqrt(DT**2 * df['z'].mean())

    # Histogram min/max edges with 5% padding
    x_range = df['x'].max() - df['x'].min()
    y_range = df['y'].max() - df['y'].min()
    x_pad = 0.1 * x_range
    y_pad = 0.1 * y_range
    x_min, x_max = df['x'].min() - x_pad, df['x'].max() + x_pad
    y_min, y_max = df['y'].min() - y_pad, df['y'].max() + y_pad

    # Build bin edges (fixed step)
    x_edges = np.arange(x_min, x_max, 0.150)
    y_edges = np.arange(y_min, y_max, 0.150)
    
    # Dictionary to store counts
    counts_dict = {}

    # Loop over points
    for _, row in df.iterrows():
        x0, y0, z0 = row['x'], row['y'], row['z']
        nel = int(row['nel'])
        sigma_xy = np.sqrt(DT**2 * z0)
        
        # Sample points
        xs = np.random.normal(loc=x0, scale=sigma_xy, size=nel)
        ys = np.random.normal(loc=y0, scale=sigma_xy, size=nel)
        
        # Find bin indices
        x_idx = np.digitize(xs, x_edges) - 1
        y_idx = np.digitize(ys, y_edges) - 1
        
        # Only keep valid indices inside the bins
        valid = (x_idx >= 0) & (x_idx < len(x_edges)-1) & (y_idx >= 0) & (y_idx < len(y_edges)-1)
        x_idx, y_idx = x_idx[valid], y_idx[valid]
        
        # Count occurrences
        for xi, yi in zip(x_idx, y_idx):
            key = (x_edges[xi], y_edges[yi])
            counts_dict[key] = counts_dict.get(key, 0) + 1

    # Convert dictionary to DataFrame
    data = {'x': [], 'y': [], 'counts': []}
    for (x, y), c in counts_dict.items():
        if c > 0:
            data['x'].append(x)
            data['y'].append(y)
            data['counts'].append(c)
    
    return pd.DataFrame(data)

def build_3d_counts_df(df, DT, DL, rebin=1):

  
    """
    df: DataFrame with columns ['x','y','z','nel']
    DT, DL: parameters to compute Gaussian sigmas in XY and Z
    returns: DataFrame with columns ['x','y','z','counts'], only counts > 0
    """
    # Compute bin sizes based on mean Z
    bin_xy = rebin*np.sqrt(DT**2 * df['z'].mean())
    bin_z  = rebin*np.sqrt(DL**2 * df['z'].mean())

    # Histogram min/max edges with 5% padding
    x_range = df['x'].max() - df['x'].min()+0.6
    y_range = df['y'].max() - df['y'].min()+0.6
    z_range = df['z'].max() - df['z'].min()+0.6
    x_pad = 1.5 * x_range
    y_pad = 1.5 * y_range
    z_pad = 1.5 * z_range
    x_min, x_max = df['x'].min() - x_pad, df['x'].max() + x_pad
    y_min, y_max = df['y'].min() - y_pad, df['y'].max() + y_pad
    z_min, z_max = df['z'].min() - z_pad, df['z'].max() + z_pad

    # Build bin edges (fixed step)
    x_edges = np.arange(x_min, x_max, bin_xy)
    y_edges = np.arange(y_min, y_max, bin_xy)
    z_edges = np.arange(z_min, z_max, bin_z)


    # Dictionary to store counts
    counts_dict = {}

    # Loop over points
    for _, row in df.iterrows():
        x0, y0, z0 = row['x'], row['y'], row['z']
        nel = np.random.poisson(lam=row['nel'])
        
        # Gaussian sigmas
        sigma_xy = np.sqrt(DT**2 * z0)
        sigma_z  = np.sqrt(DL**2 * z0)
        
        # Sample points
        xs = np.random.normal(loc=x0, scale=sigma_xy, size=nel)
        ys = np.random.normal(loc=y0, scale=sigma_xy, size=nel)
        zs = np.random.normal(loc=z0, scale=sigma_z,  size=nel)
        
        # Find bin indices
        x_idx = np.digitize(xs, x_edges) - 1
        y_idx = np.digitize(ys, y_edges) - 1
        z_idx = np.digitize(zs, z_edges) - 1
        
        # Only keep valid indices inside the bins
        valid = (x_idx >= 0) & (x_idx < len(x_edges)-1) & \
                (y_idx >= 0) & (y_idx < len(y_edges)-1) & \
                (z_idx >= 0) & (z_idx < len(z_edges)-1)
        x_idx, y_idx, z_idx = x_idx[valid], y_idx[valid], z_idx[valid]
        
        # Count occurrences
        for xi, yi, zi in zip(x_idx, y_idx, z_idx):
            key = (x_edges[xi], y_edges[yi], z_edges[zi])
            counts_dict[key] = counts_dict.get(key, 0) + 1

    # Convert dictionary to DataFrame
    data = {'X': [], 'Y': [], 'Z': [], 'Q': []}
    for (x, y, z), c in counts_dict.items():
        if c > 0:
            data['X'].append(x)
            data['Y'].append(y)
            data['Z'].append(z)
            data['Q'].append(c)
    
    return pd.DataFrame(data)


def build_3d_counts_df_ele(df, DT, DL, rebin = 1):
    """
    df: DataFrame with columns ['x','y','z','nel']
    DT, DL: parameters to compute Gaussian sigmas in XY and Z
    returns: DataFrame with columns ['x','y','z','counts'], only counts > 0
    """
    # Compute bin sizes based on mean Z
    bin_xy = rebin*np.sqrt(DT**2 * max(100,df['z'].mean()))
    bin_z  = rebin*np.sqrt(DL**2 * max(100,df['z'].mean()))

    # Histogram min/max edges with 5% padding
    x_range = df['x'].max() - df['x'].min()+3*bin_xy
    y_range = df['y'].max() - df['y'].min()+3*bin_xy
    z_range = df['z'].max() - df['z'].min()+3*bin_z
    x_pad = 10.5 * x_range
    y_pad = 10.5 * y_range
    z_pad = 10.5 * z_range
    x_min, x_max = df['x'].min() - x_pad, df['x'].max() + x_pad
    y_min, y_max = df['y'].min() - y_pad, df['y'].max() + y_pad
    z_min, z_max = df['z'].min() - z_pad, df['z'].max() + z_pad

    # Build bin edges (fixed step)
    x_edges = np.arange(x_min, x_max, bin_xy)
    y_edges = np.arange(y_min, y_max, bin_xy)
    z_edges = np.arange(z_min, z_max, bin_z)

    # Dictionary to store counts
    counts_dict = {}

    # Loop over points
    for _, row in df.iterrows():
        x0, y0, z0 = row['x'], row['y'], row['z']
        nel = np.random.poisson(lam=row['nel'])
        
        # Gaussian sigmas
        sigma_xy = np.sqrt(DT**2 * max(100,z0))
        sigma_z  = np.sqrt(DL**2 * max(100,z0))
        
        # Sample points
        xs = np.random.normal(loc=x0, scale=sigma_xy, size=nel)
        ys = np.random.normal(loc=y0, scale=sigma_xy, size=nel)
        zs = np.random.normal(loc=z0, scale=sigma_z,  size=nel)
                
        # Find bin indices
        x_idx = np.digitize(xs, x_edges) - 1
        y_idx = np.digitize(ys, y_edges) - 1
        z_idx = np.digitize(zs, z_edges) - 1

        # Only keep valid indices inside the bins
        valid = (x_idx >= 0) & (x_idx < len(x_edges)-1) & \
                (y_idx >= 0) & (y_idx < len(y_edges)-1) & \
                (z_idx >= 0) & (z_idx < len(z_edges)-1)
        x_idx, y_idx, z_idx = x_idx[valid], y_idx[valid], z_idx[valid]
        
        # Count occurrences
        for xi, yi, zi in zip(x_idx, y_idx, z_idx):
            key = (x_edges[xi], y_edges[yi], z_edges[zi])
            counts_dict[key] = counts_dict.get(key, 0) + 1

    # Convert dictionary to DataFrame
    data = {'X': [], 'Y': [], 'Z': [], 'Q': []}
    for (x, y, z), c in counts_dict.items():
        if c > 0:
            data['X'].append(x)
            data['Y'].append(y)
            data['Z'].append(z)
            data['Q'].append(c)
    
    return pd.DataFrame(data)


def build_3d_bool_hist(df, DT, DL, threshold=1):
    """
    Build a memory-optimized 3D boolean histogram:
        hist_bool[i,j,k] = True  if at least 1 sample falls in bin (i,j,k)
                           False otherwise

    Returns:
        hist_bool : np.ndarray, dtype=bool, shape (Nx,Ny,Nz)
        x_edges, y_edges, z_edges
    """

    # --- Compute bin edges (same logic as your original function) ---
    bin_xy = np.sqrt(DT**2 * df['z'].mean())
    bin_z  = np.sqrt(DL**2 * df['z'].mean())

    x_range = df['x'].max() - df['x'].min()
    y_range = df['y'].max() - df['y'].min()
    z_range = df['z'].max() - df['z'].min()

    x_pad = 0.3 * x_range
    y_pad = 0.3 * y_range
    z_pad = 0.5 * z_range

    x_min, x_max = df['x'].min() - x_pad, df['x'].max() + x_pad
    y_min, y_max = df['y'].min() - y_pad, df['y'].max() + y_pad
    z_min, z_max = df['z'].min() - z_pad, df['z'].max() + z_pad

    x_edges = np.arange(x_min, x_max, bin_z)
    y_edges = np.arange(y_min, y_max, bin_z)
    z_edges = np.arange(z_min, z_max, bin_z)

    Nx = len(x_edges) - 1
    Ny = len(y_edges) - 1
    Nz = len(z_edges) - 1

    if threshold < 1:
        raise ValueError("threshold must be >= 1")

    # --- MEMORY-OPTIMIZED HISTOGRAM ---
    # keep integer counts so thresholding can be applied at the end
    hist_counts = np.zeros((Nx, Ny, Nz), dtype=np.uint32)

    # --- Fill the histogram ---
    for _, row in df.iterrows():

        # point and nel
        x0, y0, z0 = row['x'], row['y'], row['z']
        nel = np.random.poisson(lam=row['nel'])

        # Gaussian sigmas
        sigma_xy = np.sqrt(DT**2 * z0)
        sigma_z  = np.sqrt(DL**2 * z0)

        # Draw random points (temporary arrays, immediately thrown away)
        xs = np.random.normal(loc=x0, scale=sigma_xy, size=nel)
        ys = np.random.normal(loc=y0, scale=sigma_xy, size=nel)
        zs = np.random.normal(loc=z0, scale=sigma_z,  size=nel)

        # Set all zs to z0
        zs = np.full(nel, z0)
        # Get bin indices
        xi = np.digitize(xs, x_edges) - 1
        yi = np.digitize(ys, y_edges) - 1
        zi = np.digitize(zs, z_edges) - 1

        valid = (
            (xi >= 0) & (xi < Nx) &
            (yi >= 0) & (yi < Ny) &
            (zi >= 0) & (zi < Nz)
        )

        xi, yi, zi = xi[valid], yi[valid], zi[valid]

        # Accumulate counts for each valid bin hit
        np.add.at(hist_counts, (xi, yi, zi), 1)

    hist_bool = hist_counts >= threshold

    return hist_bool, x_edges, y_edges, z_edges


def sum_Q_in_sphere(pt, df_counts, radius):
    """
    pt: numpy array (x,y,z)
    df_counts: DataFrame with columns ['x','y','z','Q']
    radius: sphere radius
    """
    # convert to numpy array if needed
    xyz = df_counts[['X','Y','Z']].to_numpy()

    # squared distance to avoid sqrt
    dist2 = np.sum((xyz - pt)**2, axis=1)
    mask = dist2 <= radius**2

    # sum Q (or counts)
    return df_counts.loc[mask, 'Q'].sum()


def sum_Q_in_ellipsoid(pt, df_counts, ellipse_size):
    """
    pt: numpy array (x,y,z) center of ellipsoid
    df_counts: DataFrame with columns ['x','y','z','counts']
    a: semi-axis along z
    b: semi-axis in x-y plane
    """
    a,b = ellipse_size[0],ellipse_size[1]

    # Extract xyz as array
    xyz = df_counts[['X','Y','Z']].to_numpy()

    # Shift coordinates
    dxyz = xyz - pt  # shape (N,3)

    dx = dxyz[:,0]
    dy = dxyz[:,1]
    dz = dxyz[:,2]

    # Ellipsoid condition
    inside = (dx**2 + dy**2) / (b**2) + (dz**2) / (a**2) <= 1.0

    # Sum counts
    return df_counts.loc[inside, 'Q'].sum()
