import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde


def compute_primary_path(df, thresh):
    """
    Computes the 'primary path' of a set of 3D points:
    - longest among all shortest paths in the MST
    - if multiple, selects the one with minimum deflection
    """
    points = df[['X', 'Y', 'Z']].to_numpy()
    
    # --- 1. Compute MST ---
    D = distance_matrix(points, points)
    D[D > thresh] = 999
    mst = minimum_spanning_tree(D).toarray()

    # --- 2. Build graph ---
    G = nx.Graph()
    for i in range(len(points)):
        for j in range(len(points)):
            if mst[i, j] > 0:
                G.add_edge(i, j, weight=mst[i, j])

    # --- 3. Compute all pairs shortest paths ---
    lengths = dict(nx.all_pairs_dijkstra_path_length(G))
    paths = dict(nx.all_pairs_dijkstra_path(G))

    # --- 4. Find all paths with maximum length ---
    max_len = 0
    max_paths = []

    for i in lengths:
        for j in lengths[i]:
            l = lengths[i][j]

            if l > max_len:
                max_len = l
                max_paths = [paths[i][j]]
            elif l == max_len:
                max_paths.append(paths[i][j])

    # --- 5. If multiple, choose path with minimum deflection ---
    def compute_deflection(path_pts):
        """
        Compute a simple deflection measure:
        sum of angles between consecutive segments
        """
        if len(path_pts) < 3:
            return 0
        def angle(v1, v2):
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
            cos_theta = np.clip(cos_theta, -1, 1)
            return np.arccos(cos_theta)
        total_angle = 0
        for k in range(1, len(path_pts)-1):
            v1 = path_pts[k] - path_pts[k-1]
            v2 = path_pts[k+1] - path_pts[k]
            total_angle += angle(v1, v2)
        return total_angle

    # Select path with minimum deflection
    best_path = max_paths[0]
    min_deflection = compute_deflection(points[best_path])
    for p in max_paths[1:]:
        defl = compute_deflection(points[p])
        if defl < min_deflection:
            best_path = p
            min_deflection = defl

    # --- 6. Return ---
    primary_path_points = points[best_path]
    return primary_path_points


def compute_primary_path_fast(df, thresh):
    """
    Compute the centerline through a 3D point cloud using:
    1. Minimum Spanning Tree (MST)
    2. Tree diameter (two-pass Dijkstra)
    
    Returns:
        path_indices: ordered list of indices along the best path
        length: total geodesic length of the path
        endpoints: (start_node, end_node)
    """

    points = df[['X', 'Y', 'Z']].to_numpy()
    
    # --- 1. Distance matrix ----
    D = distance_matrix(points, points)
    D[D > thresh] = 999

    # --- 2. Compute MST ----
    mst = minimum_spanning_tree(D).toarray()

    # --- 3. Build graph from MST ----
    G = nx.Graph()
    N = len(points)
    for i in range(N):
        for j in range(N):
            if mst[i, j] > 0:
                G.add_edge(i, j, weight=mst[i, j])

    # --- 4. First Dijkstra: from arbitrary node (0) to find farthest node A ---
    lengths_0 = nx.single_source_dijkstra_path_length(G, 0)
    A = max(lengths_0, key=lengths_0.get)

    # --- 5. Second Dijkstra: from A to find farthest node B (diameter endpoint) ---
    lengths_A, paths_A = nx.single_source_dijkstra(G, A)
    B = max(lengths_A, key=lengths_A.get)

    # --- 6. The diameter path is the path from A to B ---
    best_path = paths_A[B]
    best_length = lengths_A[B]

    return df[['X', 'Y', 'Z']].to_numpy()[best_path]


def compute_primary_path_angle_penalized(
    df,
    k=8,
    lambda_angle=0.8
):
    """
    Compute a smooth primary path through a 3D point cloud by:
      - enforcing monotone progression along a global axis
      - penalizing sharp turns (high curvature)

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ['X', 'Y', 'Z']
    k : int
        Number of nearest neighbors
    lambda_angle : float
        Strength of angle penalty

    Returns
    -------
    path_xyz : (M, 3) ndarray
        Ordered path coordinates
    """

    # ------------------------------------------------------------
    # 1. Input points
    # ------------------------------------------------------------
    points = df[['X', 'Y', 'Z']].to_numpy()
    N = len(points)

    # ------------------------------------------------------------
    # 2. Compute monotone parameter (PCA axis)
    # ------------------------------------------------------------
    axis = PCA(n_components=1).fit(points).components_[0]
    s = points @ axis

    # ------------------------------------------------------------
    # 3. Build forward-only kNN graph (DAG)
    # ------------------------------------------------------------
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = nbrs.kneighbors(points)

    G_base = nx.DiGraph()

    for i in range(N):
        for d, j in zip(distances[i], indices[i]):
            if j != i and s[j] > s[i]:
                G_base.add_edge(i, j, weight=d)

    # Guard: ensure graph is not empty
    if G_base.number_of_edges() == 0:
        raise RuntimeError("No forward edges found; try increasing k or flipping axis")

    # ------------------------------------------------------------
    # 4. Build angle-penalized state graph (still a DAG)
    #    State = (prev_node, current_node)
    # ------------------------------------------------------------
    def turning_angle(u, v, w):
        a = points[v] - points[u]
        b = points[w] - points[v]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        cos_theta = np.dot(a, b) / (na * nb)
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))

    G_state = nx.DiGraph()

    for v in G_base.nodes:
        for u in G_base.predecessors(v):
            for w in G_base.successors(v):

                theta = turning_angle(u, v, w)
                angle_penalty = lambda_angle * theta * theta
                edge_len = np.linalg.norm(points[w] - points[v])

                cost = edge_len + angle_penalty
                G_state.add_edge((u, v), (v, w), weight=cost)

    # ------------------------------------------------------------
    # 5. Add super-source for path start
    # ------------------------------------------------------------
    SOURCE = (-1, -1)
    G_state.add_node(SOURCE)

    for i in G_base.nodes:
        for j in G_base.successors(i):
            G_state.add_edge(SOURCE, (i, j), weight=0.0)

    # ------------------------------------------------------------
    # 6. Longest path in DAG (safe and exact)
    # ------------------------------------------------------------
    state_path = nx.dag_longest_path(G_state, weight='weight')

    # ------------------------------------------------------------
    # 7. Recover node path
    # ------------------------------------------------------------
    node_path = [v for (_, v) in state_path if v >= 0]

    # Remove consecutive duplicates
    cleaned = [node_path[0]]
    for n in node_path[1:]:
        if n != cleaned[-1]:
            cleaned.append(n)

    return points[cleaned]


def compute_primary_path_mst_angle_conditioned(
    df,
    k=8,
    max_angle_deg=60.0
):
    """
    Compute a smooth primary path using a Minimum Spanning Tree
    with an angle condition to prevent sharp turns.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns: ['X', 'Y', 'Z']
    k : int
        Number of nearest neighbors for candidate edges
    max_angle_deg : float
        Maximum allowed angle (degrees) between incident edges in MST

    Returns
    -------
    path_xyz : (M, 3) ndarray
        Ordered path coordinates
    """

    points = df[['X', 'Y', 'Z']].to_numpy()
    N = len(points)

    # ------------------------------------------------------------
    # 1. Global monotone direction (PCA)
    # ------------------------------------------------------------
    axis = PCA(n_components=1).fit(points).components_[0]
    s = points @ axis

    # ------------------------------------------------------------
    # 2. Candidate edges (forward-only kNN)
    # ------------------------------------------------------------
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = nbrs.kneighbors(points)

    candidate_edges = []
    for i in range(N):
        for d, j in zip(distances[i], indices[i]):
            if j != i and s[j] > s[i]:
                candidate_edges.append((d, i, j))

    candidate_edges.sort(key=lambda x: x[0])

    # ------------------------------------------------------------
    # 3. Angle helper
    # ------------------------------------------------------------
    def angle(u, v, w):
        a = points[u] - points[v]
        b = points[w] - points[v]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        cosang = np.dot(a, b) / (na * nb)
        return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

    # ------------------------------------------------------------
    # 4. Build MST with angle condition (FIXED)
    # ------------------------------------------------------------
    parent = list(range(N))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[rv] = ru
            return True
        return False

    G = nx.Graph()

    for w, u, v in candidate_edges:
        if find(u) == find(v):
            continue

        valid = True

        # --- Angle condition at u ---
        if G.has_node(u):
            for nbr in G.neighbors(u):
                if angle(nbr, u, v) > max_angle_deg:
                    valid = False
                    break

        # --- Angle condition at v ---
        if valid and G.has_node(v):
            for nbr in G.neighbors(v):
                if angle(u, v, nbr) > max_angle_deg:
                    valid = False
                    break

        if not valid:
            continue

        union(u, v)
        G.add_edge(u, v, weight=w)

    # ------------------------------------------------------------
    # 5. Tree diameter (two-pass Dijkstra)
    # ------------------------------------------------------------
    start = list(G.nodes)[0]
    dist0 = nx.single_source_dijkstra_path_length(G, start)
    A = max(dist0, key=dist0.get)

    distA, pathsA = nx.single_source_dijkstra(G, A)
    B = max(distA, key=distA.get)

    path_nodes = pathsA[B]

    return points[path_nodes]


def reconstruct_path(df, primary_path_points, radius=40):
    """
    Smooth the primary path by replacing each point with the Q-weighted centroid
    of points within a given radius.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'X', 'Y', 'Z', 'Q'.
    primary_path_points : np.ndarray
        Array of shape (N,3) containing points along the primary path.
    radius : float
        Neighborhood radius to consider for the barycentre (same units as X,Y,Z).
    
    Returns
    -------
    reconstructed_path : np.ndarray
        Smoothed path points (N,3).
    """
    # Build KDTree of all points
    pts = df[['X','Y','Z']].to_numpy()
    Q = df['Q'].to_numpy()
    tree = cKDTree(pts)

    smoothed_path = []

    for p in primary_path_points:
        # Find all points within radius
        idx = tree.query_ball_point(p, radius)
        if len(idx) == 0:
            # No points nearby: keep original
            smoothed_path.append(p)
        else:
            # Energy-weighted centroid
            local_pts = pts[idx]
            local_Q = Q[idx]
            centroid = np.average(local_pts, axis=0, weights=local_Q)
            smoothed_path.append(centroid)

    return np.array(smoothed_path)


def reconstruct_path_ellipse(df, primary_path_points, ellipse_size):
    """
    Smooth the primary path by replacing each point with the Q-weighted centroid
    of points inside an ellipsoidal neighborhood (axis `a` along Z, `b` on X/Y).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'X', 'Y', 'Z', 'Q'.
    primary_path_points : np.ndarray
        Array of shape (N,3) containing points along the primary path.
    a : float
        Semi-axis of the ellipsoid along Z.
    b : float
        Semi-axis of the ellipsoid along X and Y (assumed equal for both).

    Returns
    -------
    reconstructed_path : np.ndarray
        Smoothed path points (N,3).
    """
    # Build KDTree in scaled coordinate system so ellipsoid check becomes spherical
    pts = df[['X', 'Y', 'Z']].to_numpy()
    Q = df['Q'].to_numpy()

    a=ellipse_size[0]
    b=ellipse_size[1]

    scaled_pts = np.column_stack((pts[:, 0] / b, pts[:, 1] / b, pts[:, 2] / a))
    tree = cKDTree(scaled_pts)

    smoothed_path = []

    for p in primary_path_points:
        scaled_p = np.array([p[0] / b, p[1] / b, p[2] / a])
        # radius 1 in scaled space corresponds to ellipsoid in original space
        idx = tree.query_ball_point(scaled_p, 1.0)
        if len(idx) == 0:
            smoothed_path.append(p)
        else:
            local_pts = pts[idx]
            local_Q = Q[idx]
            centroid = np.average(local_pts, axis=0, weights=local_Q)
            smoothed_path.append(centroid)

    return np.array(smoothed_path)

def mean_filter_path(path_points, window=5):
    """
    Applies a mean (moving average) filter to a 3D path.

    Parameters
    ----------
    path_points : (N,3) array
        Input polyline (e.g., smoothed_path).
    window : int
        Number of points in the moving average window (must be odd).

    Returns
    -------
    filtered : (N,3) array
        Mean-filtered path.
    """
    if window < 1:
        return path_points.copy()

    if window % 2 == 0:
        raise ValueError("Window size must be odd.")

    N = len(path_points)
    half = window // 2

    filtered = np.zeros_like(path_points)

    # Pad edges to preserve length
    padded = np.pad(path_points, ((half, half), (0, 0)), mode="edge")

    # Mean filter
    for i in range(N):
        filtered[i] = padded[i:i+window].mean(axis=0)

    return filtered



def compute_min_axis_spacing(df):
    """
    Computes the minimum spacing between unique coordinate values
    along X, Y, and Z.
    
    Returns
    -------
    (dx_min, dy_min, dz_min)
    """

    def min_diff(values):
        """Return the smallest difference between sorted unique values."""
        uniq = np.sort(np.unique(values))
        if len(uniq) < 2:
            return np.nan
        diffs = np.diff(uniq)
        return np.min(diffs)

    dx = min_diff(df["X"].to_numpy())
    dy = min_diff(df["Y"].to_numpy())
    dz = min_diff(df["Z"].to_numpy())

    return dx, dy, dz


def remove_isolated_points(df, radius=1.2):
    """
    Removes points that have no neighbors within a given radius.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'X', 'Y', 'Z'.
    radius : float
        Distance threshold to consider a point 'connected'.

    Returns
    -------
    df_filtered : pandas.DataFrame
        DataFrame with isolated points removed.
    """

    dx, dy, dz = compute_min_axis_spacing(df)

    # Build KD-tree
    points = df[['X', 'Y', 'Z']].to_numpy()

    points[:, 0] /= dx  # X
    points[:, 1] /= dy  # Y
    points[:, 2] /= dz  # Z

    tree = cKDTree(points)

    # Count neighbors for each point (including itself, so ≥2 means not isolated)
    neighbor_lists = tree.query_ball_tree(tree, r=radius)

    # A point is isolated if len(neighbors) == 1 (only itself)
    mask = np.array([len(neigh) > 3 for neigh in neighbor_lists])

    # Return only points that are not isolated
    return df[mask].copy()


def prune_edges(df, distance_threshold=1.5):
    """
    Iteratively remove edge points without breaking connectivity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ['X','Y','Z','Q'].
    distance_threshold : float
        Maximum distance to consider points as neighbors.
        
    Returns
    -------
    pd.DataFrame
        Pruned DataFrame.
    """
    # Copy input
    df_pruned = df.copy()
    
    # Build k-d tree for neighbors
    coords = df_pruned[['X','Y','Z']].values
    tree = cKDTree(coords)
    
    # Build graph: edges between points within distance_threshold
    G = nx.Graph()
    for i, coord in enumerate(coords):
        neighbors = tree.query_ball_point(coord, r=distance_threshold)
        for j in neighbors:
            if i != j:
                G.add_edge(i, j)
    
    # Function to check if a node is a boundary (has less than max neighbors)
    def is_edge_node(node):
        return len(list(G.neighbors(node))) < 6  # heuristic
    
    # Iteratively remove edge nodes if connectivity is preserved
    nodes_to_check = list(G.nodes)
    while True:
        removed_any = False
        for node in nodes_to_check:
            if node not in G:
                continue
            if is_edge_node(node):
                # Remove node and check connectivity
                G.remove_node(node)
                if nx.is_connected(G):
                    removed_any = True
                else:
                    # Restore
                    G.add_node(node)
                    # Reconnect edges
                    coord = coords[node]
                    neighbors = tree.query_ball_point(coord, r=distance_threshold)
                    for j in neighbors:
                        if j != node and j in G:
                            G.add_edge(node, j)
        if not removed_any:
            break
    
    # Return pruned DataFrame
    remaining_indices = list(G.nodes)
    return df_pruned.iloc[remaining_indices].reset_index(drop=True)


def skeletonize_point_cloud_from_df(df, distance_threshold=2.0, min_degree=1):
    """
    Skeletonize 3D point cloud from a dataframe with columns ['X','Y','Z','Q'].
    """
    coords = df[['X','Y','Z']].to_numpy()
    Q = df['Q'].to_numpy() if 'Q' in df.columns else None
    
    N = len(coords)
    G = nx.Graph()

    # Add nodes with indices 0..N-1
    for i in range(N):
        G.add_node(i, pos=coords[i], Q=Q[i] if Q is not None else 1.0)

    # Build KD-tree
    tree = cKDTree(coords)

    for i, pt in enumerate(coords):
        neighbors = tree.query_ball_point(pt, r=distance_threshold)
        for j in neighbors:
            if i >= j:
                continue
            dist = np.linalg.norm(pt - coords[j])
            weight = dist #/ (G.nodes[i]['Q'] + G.nodes[j]['Q'])
            G.add_edge(i, j, weight=weight)

    # Extremities and shortest paths (same as before)
    extremities = [n for n in G.nodes if G.degree[n] <= min_degree]
    if len(extremities) < 2:
        extremities = list(G.nodes)
    
    skeleton_edges = set()
    for i, src in enumerate(extremities):
        for dst in extremities[i+1:]:
            try:
                path = nx.shortest_path(G, source=src, target=dst, weight='distance')
                skeleton_edges.update([(path[k], path[k+1]) for k in range(len(path)-1)])
            except nx.NetworkXNoPath:
                continue
    
    G_skel = nx.Graph()
    G_skel.add_nodes_from(range(N))
    G_skel.add_edges_from(skeleton_edges)
    
    # Optional pruning leaves
    while True:
        leaves = [n for n in G_skel.nodes if G_skel.degree[n] == 1]
        if not leaves:
            break
        G_skel.remove_nodes_from(leaves)
    
    skeleton_nodes = coords[list(G_skel.nodes)]
    skeleton_edges = list(G_skel.edges)
    
    return skeleton_nodes, skeleton_edges


def filter_points(filtered_nodes, max_distance=5.0):
    """
    Filter consecutive points in ordered array - remove point if it's too far from previous point
    """
    if len(filtered_nodes) == 0:
        return filtered_nodes
    
    filtered_result = [filtered_nodes[0]]  # Always keep first point
    
    for i in range(1, len(filtered_nodes)):
        current_point = filtered_nodes[i]
        previous_point = filtered_result[-1]  # Last point we kept
        
        dist = np.linalg.norm(current_point - previous_point)
        
        if dist <= max_distance:
            filtered_result.append(current_point)
        # else: skip this point (don't add it to filtered_result)
    
    return np.array(filtered_result)


def sort_skeleton_points(skeleton_nodes, skeleton_edges):
    """
    Sort skeleton points in order between the two extremities
    """
    if len(skeleton_nodes) == 0:
        return skeleton_nodes
    
    # Create graph from skeleton
    G = nx.Graph()
    for i in range(len(skeleton_nodes)):
        G.add_node(i, pos=skeleton_nodes[i])
    G.add_edges_from(skeleton_edges)
    
    # Find extremities (degree 1 nodes)
    extremities = [n for n in G.nodes if G.degree(n) == 1]
    
    if len(extremities) != 2:
        # If not exactly 2 extremities, can't sort linearly
        return skeleton_nodes
    
    # Find path between the two extremities
    start, end = extremities[0], extremities[1]
    path = nx.shortest_path(G, source=start, target=end)
    
    # Return nodes in order along the path
    sorted_nodes = np.array([skeleton_nodes[i] for i in path])
    return sorted_nodes

# Usage:
#skeleton_nodes, skeleton_edges = skeletonize_point_cloud_from_df(df, distance_threshold=2.0)
#sorted_skeleton = sort_skeleton_points(skeleton_nodes, skeleton_edges)


def skeletonize_point_cloud_from_dfV2(df, distance_threshold=2.0, min_degree=1):
    """
    Skeletonize 3D point cloud from a dataframe with columns ['X','Y','Z','Q'].
    Only connects points whose pairwise distance <= distance_threshold.
    """
    coords = df[['X','Y','Z']].to_numpy()
    Q = df['Q'].to_numpy() if 'Q' in df.columns else None
    
    N = len(coords)
    G = nx.Graph()

    # Add nodes
    for i in range(N):
        G.add_node(i, pos=coords[i], Q=Q[i] if Q is not None else 1.0)

    # Build KD-tree
    tree = cKDTree(coords)

    # Build graph with hard distance cutoff
    for i, pt in enumerate(coords):
        neighbors = tree.query_ball_point(pt, r=distance_threshold)
        for j in neighbors:
            if i >= j:
                continue
            dist = np.linalg.norm(pt - coords[j])
            if dist <= distance_threshold:           # <-- strict local threshold
                G.add_edge(i, j, weight=dist)        # attribute name: 'weight'

    # Extremities
    extremities = [n for n in G.nodes if G.degree[n] <= min_degree]
    if len(extremities) < 2:
        extremities = list(G.nodes)

    skeleton_edges = set()

    # Shortest paths between extremities using correct weight
    for i, src in enumerate(extremities):
        for dst in extremities[i+1:]:
            max_endpoint_distance = 3.0  # or whatever
            if np.linalg.norm(coords[src] - coords[dst]) > max_endpoint_distance:
                continue
            try:
                path = nx.shortest_path(G, source=src, target=dst, weight='weight')
                # add all edges of the path, they are already <= distance_threshold
                for k in range(len(path) - 1):
                    u, v = path[k], path[k+1]
                    # optional: safety check (should always pass)
                    if G[u][v]['weight'] <= distance_threshold:
                        skeleton_edges.add((u, v))
            except nx.NetworkXNoPath:
                continue

    # Build skeleton graph
    G_skel = nx.Graph()
    G_skel.add_nodes_from(range(N))
    G_skel.add_edges_from(skeleton_edges)

    # Optional: prune leaves iteratively
    #while True:
    #    leaves = [n for n in G_skel.nodes if G_skel.degree[n] == 1]
    #    if not leaves:
    #        break
    #    G_skel.remove_nodes_from(leaves)

    skeleton_nodes = coords[list(G_skel.nodes)]
    skeleton_edges = list(G_skel.edges)

    return skeleton_nodes, skeleton_edges



from skimage.morphology import skeletonize
def skeleton_voxel_coordinates(hist_bins, edges_x, edges_y, edges_z):
    """
    hist_bins: boolean 3D histogram (Nx, Ny, Nz) or any shape
    edges_x, edges_y, edges_z: histogram bin edges
    """

    # 1. Reorder for skimage (Z, Y, X)
    volume = hist_bins.transpose(2, 1, 0)

    # 2. Skeleton
    skel = skeletonize(volume, method='lee')

    # 3. Get voxel indices
    z_idx, y_idx, x_idx = np.where(skel)

    # 4. Compute bin centers
    xs = 0.5 * (edges_x[:-1] + edges_x[1:])
    ys = 0.5 * (edges_y[:-1] + edges_y[1:])
    zs = 0.5 * (edges_z[:-1] + edges_z[1:])

    # 5. Convert voxel indices → real coordinates
    X = xs[x_idx]
    Y = ys[y_idx]
    Z = zs[z_idx]

    # Stack
    coords = np.vstack([X, Y, Z]).T

    return coords

def skeleton_voxel_coordinates2D(hist_bins, edges_x, edges_y):
    """
    hist_bins: boolean 2D histogram (Nx, Ny)
    edges_x, edges_y: histogram bin edges
    """

    # 1. Reorder for skimage (Y, X)
    volume = hist_bins.transpose(1, 0)

    # 2. Skeleton
    skel = skeletonize(volume, method='lee')

    # 3. Get voxel indices
    y_idx, x_idx = np.where(skel)

    # 4. Compute bin centers
    xs = 0.5 * (edges_x[:-1] + edges_x[1:])
    ys = 0.5 * (edges_y[:-1] + edges_y[1:])

    # 5. Convert voxel indices → real coordinates
    X = xs[x_idx]
    Y = ys[y_idx]

    # Stack
    coords = np.vstack([X, Y]).T

    return coords


def cubic_kernel_filter(df, kernel_size=5, alpha=0.3):
    """
    Charge-weighted cubic filter for voxelized 3D point clouds.

    df: DataFrame containing columns ['X','Y','Z','Q']
    kernel_size: odd integer (3, 5, 7, ...) defining cubic window
    alpha: contraction strength toward local charge centroid

    Returns: DataFrame with updated X,Y,Z and original Q
    """
    assert kernel_size % 2 == 1, "kernel_size must be an odd integer"

    # Extract point positions and charge
    points = df[['X', 'Y', 'Z']].to_numpy().astype(int)
    charge = df['Q'].to_numpy()

    radius = kernel_size // 2
    new_points = np.zeros_like(points)

    # Hash map: (x,y,z) → index
    index = {tuple(p): i for i, p in enumerate(points)}

    # Offsets inside cubic kernel
    offsets = [
        (dx, dy, dz)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
    ]

    for i, (x, y, z) in enumerate(points):
        neigh_coords = []
        neigh_charge = []

        # Collect neighbors in cube
        for dx, dy, dz in offsets:
            coord = (x + dx, y + dy, z + dz)
            if coord in index:
                j = index[coord]
                neigh_coords.append(points[j])
                neigh_charge.append(charge[j])

        neigh_coords = np.array(neigh_coords)
        neigh_charge = np.array(neigh_charge)

        # Charge-weighted centroid
        centroid = np.average(neigh_coords, weights=neigh_charge, axis=0)

        # Shrink point toward centroid
        new_points[i] = (1 - alpha) * points[i] + alpha * centroid

    # Build output dataframe with new coordinates + original Q
    df_out = df.copy()
    df_out[['X', 'Y', 'Z']] = new_points
    df_out['Q'] = charge        # ensure Q is included

    return df_out

def cubic_filter_with_anchor(df, kernel_size=5):
    """
    Apply cubic filter with an anchor rule to shrink points around high-charge regions.
    
    df: pandas DataFrame with columns ['X', 'Y', 'Z', 'Q']
        'X', 'Y', 'Z' represent the coordinates of the points
        'Q' is the charge at each point.
    
    kernel_size: size of the kernel used to determine the neighboring points
    """
    # Extract points and charge from DataFrame
    points = df[['X', 'Y', 'Z']].to_numpy()
    Q = df['Q'].to_numpy()
    
    # Initialize a new array to store updated points
    new_points = points.copy()
    
    # Iterate over each point in the dataset
    for i, (x, y, z) in enumerate(points):
        # Get the 5x5 neighborhood for the current point
        # Create a kernel window around the point (neighbors within the kernel size)
        neighbors = []
        charges = []
        
        for j, (x2, y2, z2) in enumerate(points):
            if i != j:  # Exclude the point itself
                distance = np.sqrt((x2 - x)**2 + (y2 - y)**2 + (z2 - z)**2)
                if distance <= kernel_size:
                    neighbors.append([x2, y2, z2])
                    charges.append(Q[j])
        
        # Apply the anchor rule: If the point has a higher charge, move it less
        # Compute the weighted average of the neighbors' positions
        if len(neighbors) > 0:
            total_charge = np.sum(charges)
            weight = np.array(charges) / total_charge  # Weighting based on charge
            
            # Calculate the weighted average position of the neighbors
            weighted_position = np.average(neighbors, axis=0, weights=weight)
            
            # Calculate movement (distance between current and new position)
            movement = weighted_position - [x, y, z]
            
            # Apply anchor rule: If the charge of the point is higher, reduce movement
            if Q[i] > np.max(charges):  # The point's charge is greater than its neighbors
                movement *= 0.2  # Reduce the movement (conservative)
            
            # Update the point's position
            new_points[i] = [x, y, z] + movement
    
    # Create a new DataFrame with the updated points
    df_new = df.copy()
    df_new[['X', 'Y', 'Z']] = new_points
    
    return df_new

def cubic_kernel_filter_adaptive_alpha(df, kernel_size=5, alpha_base=0.3, q_percentile=90):
    """
    Charge-weighted cubic filter using adaptive alpha.
    
    Parameters
    ----------
    df : DataFrame
        Must contain columns ['X','Y','Z','Q']
    kernel_size : odd integer (3,5,7,...)
        Size of cubic kernel window.
    alpha_base : float
        Maximum shrink strength for low-charge points.
    q_percentile : float
        Percentile used to define Q_ref for adaptive alpha scaling.
    
    Returns
    -------
    df_out : DataFrame
        Same dataframe but with updated X,Y,Z and original Q.
    """

    assert kernel_size % 2 == 1, "kernel_size must be an odd integer"

    # Extract data
    points = df[['X', 'Y', 'Z']].to_numpy().astype(int)
    charge = df['Q'].to_numpy()
    
    # Adaptive alpha reference charge (e.g., 90th percentile)
    Q_ref = np.percentile(charge, q_percentile)
    
    new_points = np.zeros_like(points)
    radius = kernel_size // 2

    # Hash map for fast neighbor lookup
    index = {tuple(p): i for i, p in enumerate(points)}

    # Precompute cubic kernel offsets
    offsets = [
        (dx, dy, dz)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
    ]

    for i, (x, y, z) in enumerate(points):
        neigh_coords = []
        neigh_charge = []

        # Collect neighbors inside kernel cube
        for dx, dy, dz in offsets:
            coord = (x + dx, y + dy, z + dz)
            if coord in index:
                j = index[coord]
                neigh_coords.append(points[j])
                neigh_charge.append(charge[j])

        neigh_coords = np.array(neigh_coords)
        neigh_charge = np.array(neigh_charge)

        # Charge-weighted centroid
        centroid = np.average(neigh_coords, weights=neigh_charge, axis=0)

        # ---------- Adaptive α ----------
        # Low Q → α ≈ α_base
        # High Q → α → small
        α_i = alpha_base * (Q_ref / (charge[i] + Q_ref))

        # Update point position
        new_points[i] = (1 - α_i) * points[i] + α_i * centroid

    # Build output dataframe
    df_out = df.copy()
    df_out[['X','Y','Z']] = new_points
    df_out['Q'] = charge

    return df_out
    
def kde_gradient_filter(df, kernel_size=5, bandwidth=1.0, alpha=0.2):
    """
    KDE-based gradient filter for voxelized 3D point clouds.
    
    df must contain columns: X, Y, Z, Q
    kernel_size: odd integer (5,7,...)
        KDE will be evaluated in a cube of side (kernel_size-1)
    bandwidth: KDE smoothing parameter
    alpha: step size for gradient movement
    """

    # Extract data
    points = df[['X','Y','Z']].to_numpy().astype(float)
    charge = df['Q'].to_numpy()

    # Build weighted KDE in 3D
    kde = gaussian_kde(points.T, weights=charge, bw_method=bandwidth)

    # Prepare sampling offsets
    radius = (kernel_size - 1) // 2
    offsets = np.array([
        [dx, dy, dz]
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
    ])

    # To compute gradient, we only need +/-1 offsets:
    grad_offsets = np.array([
        [ 1, 0, 0], [-1, 0, 0],
        [ 0, 1, 0], [ 0,-1, 0],
        [ 0, 0, 1], [ 0, 0,-1]
    ])

    new_points = np.zeros_like(points)

    # Process each point
    for i, p in enumerate(points):

        # Evaluate KDE at the six gradient offsets
        g_vals = kde((p + grad_offsets).T)

        gx = (g_vals[0] - g_vals[1]) / 2
        gy = (g_vals[2] - g_vals[3]) / 2
        gz = (g_vals[4] - g_vals[5]) / 2
        grad = np.array([gx, gy, gz])

        # Normalize
        norm = np.linalg.norm(grad)
        if norm < 1e-10:
            new_points[i] = p
        else:
            new_points[i] = p + alpha * grad / norm

    df_out = df.copy()
    df_out[['X','Y','Z']] = new_points
    return df_out
