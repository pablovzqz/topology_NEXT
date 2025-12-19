import matplotlib.pyplot as plt
import numpy as np
import itertools
import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio
from cycler import cycler
import sys
sys.path.append("../0nubbdata_LXe_Analysis/src/")

import clustering_fun as cf
import analysis_functions as af


def set_my_matplotlib_style():
    plt.rcParams.update({
        # --- General look ---
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "figure.autolayout": True,
        "savefig.bbox": "tight",

        # --- Fonts and sizes ---
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 12,           # Base font size
        "axes.titlesize": 16,      # Title size
        "axes.labelsize": 16,      # Axis label size
        "xtick.labelsize": 16,     # Tick label size
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 18,    # Figure title size
        "figure.dpi": 150,

        # --- Line and marker styles ---
        "lines.linewidth": 2,
        "lines.markersize": 6,

        # --- Colors ---
        "axes.prop_cycle": cycler('color', [
            '#1f77b4', '#ff7f0e', '#2ca02c',
            '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]),

        # --- Layout ---
        "figure.figsize": (6, 4),
        "axes.spines.top": False,
        "axes.spines.right": False,
    })



def plot_clusters_with_energy(df, highlight_points=None):
    """
    df: DataFrame with columns x, y, z, label, energy
    highlight_points: optional iterable of points (x, y, z) to overlay, e.g. [(p1x, p1y, p1z), (p2x, p2y, p2z)]
    Plots x vs y, z vs x, y vs z with cluster colors and legend showing cluster energy.
    """
    x = df['x']
    y = df['y']
    z = df['z']
    labels = df['label']
    
    # Compute total energy per cluster
    cluster_energy = df.groupby('label')['energy'].sum().to_dict()
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Get unique clusters for coloring and legend
    unique_labels = df['label'].unique()
    
    colors = plt.cm.tab20.colors  # up to 20 distinct colors
    color_map = {lbl: colors[i % len(colors)] for i, lbl in enumerate(unique_labels)}
    
    # x vs y
    for lbl in unique_labels:
        mask = df['label'] == lbl
        axs[0].scatter(x[mask], y[mask], s=20, color=color_map[lbl],
                       label=f"Cluster {lbl}: E={cluster_energy[lbl]:.3f}")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].legend(fontsize=8, loc='best')
    
    # z vs x
    for lbl in unique_labels:
        mask = df['label'] == lbl
        axs[1].scatter(z[mask], x[mask], s=20, color=color_map[lbl])
    axs[1].set_xlabel("z")
    axs[1].set_ylabel("x")
    
    # y vs z
    for lbl in unique_labels:
        mask = df['label'] == lbl
        axs[2].scatter(z[mask], y[mask], s=20, color=color_map[lbl])
    axs[2].set_xlabel("z")
    axs[2].set_ylabel("y")

    # Overlay highlighted points if provided
    if highlight_points is not None:
        # Normalize to iterable of (x, y, z)
        if hasattr(highlight_points, "__len__"):
            if len(highlight_points) == 3 and not hasattr(highlight_points[0], "__len__"):
                points_to_plot = [highlight_points]
            else:
                points_to_plot = list(highlight_points)
        else:
            raise TypeError("highlight_points must be an iterable of (x, y, z).")

        for idx, point in enumerate(points_to_plot):
            px, py, pz = point
            label = f"P{idx+1}"
            axs[0].scatter(px, py, s=80, color='red', marker='x', linewidths=2, label=label)
            axs[1].scatter(pz, px, s=80, color='red', marker='x', linewidths=2)
            axs[2].scatter(pz, py, s=80, color='red', marker='x', linewidths=2)

        # Keep legend readable by re-drawing
        handles, labels = axs[0].get_legend_handles_labels()
        axs[0].legend(handles, labels, fontsize=8, loc='best')
    
    plt.tight_layout()
    plt.show()



def plot_clusters_with_full_paht_mst(df):
    """
    Plots 3 projections (x–y, z–x, z–y) with all hits colored by cluster
    and overlays the full MST track path.
    """
    x, y, z = df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()
    labels = df["label"].to_numpy()
    points = np.column_stack((x, y, z))

    # Compute MST longest path
    path, max_dist = af.get_track_length_mst_V2(points)

    # Color clusters
    cluster_energy = df.groupby("label")["energy"].sum().to_dict()
    unique_labels = df["label"].unique()
    colors = plt.cm.tab20.colors
    color_map = {lbl: colors[i % len(colors)] for i, lbl in enumerate(unique_labels)}

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Scatter clusters
    for lbl in unique_labels:
        mask = labels == lbl
        axs[0].scatter(x[mask], y[mask], s=15, color=color_map[lbl])
        axs[1].scatter(z[mask], x[mask], s=15, color=color_map[lbl])
        axs[2].scatter(z[mask], y[mask], s=15, color=color_map[lbl])

    # Overlay MST path
    path_points = points[path]
    axs[0].plot(path_points[:, 0], path_points[:, 1], 'k-', lw=2, label=f"MST path ({max_dist:.1f} mm)")
    axs[1].plot(path_points[:, 2], path_points[:, 0], 'k-', lw=2)
    axs[2].plot(path_points[:, 2], path_points[:, 1], 'k-', lw=2)

    axs[0].set_xlabel("x"); axs[0].set_ylabel("y"); axs[0].legend()
    axs[1].set_xlabel("z"); axs[1].set_ylabel("x")
    axs[2].set_xlabel("z"); axs[2].set_ylabel("y")

    plt.suptitle("Event Topology with MST Track Path", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_clusters_with_full_paht_mst_3D(df):
    """
    Plots 3 projections (x–y, z–x, z–y) with all hits colored by cluster
    and overlays the full MST track path.
    """
    x, y, z = df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()
    labels = df["label"].to_numpy()
    points = np.column_stack((x, y, z))

    # Compute MST longest path
    path, max_dist = af.get_track_length_mst_V2(points)

    # Color clusters
    cluster_energy = df.groupby("label")["energy"].sum().to_dict()
    unique_labels = df["label"].unique()
    colors = plt.cm.tab20.colors
    color_map = {lbl: colors[i % len(colors)] for i, lbl in enumerate(unique_labels)}

    path_points = points[path]
    
    pio.renderers.default = "browser"   

    fig = go.Figure()
    
    # Original points
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(size=2, color='blue'),
        name='Original Points'
    ))
    
    # Spline path
    fig.add_trace(go.Scatter3d(
        x=path_points[:, 0],
        y=path_points[:, 1],
        z=path_points[:, 2],
        mode='lines',
        line=dict(color='red', width=4),
        name='Spline'
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=900,
        height=700,
        showlegend=True
    )
    
    fig.show()


def plot_event_with_pca(df):
    # --- Prepare data ---
    points = df[["x", "y", "z"]].to_numpy()
    x, y, z = points[:,0], points[:,1], points[:,2]
    
    # Compute PCA plane and ellipse
    pca, a, b, vecs2 = af.ellipse_axes_2D_plot(points)
    
    # 2D ellipse in PCA space
    theta = np.linspace(0, 2*np.pi, 200)
    ellipse_2d = np.vstack([2*a*np.cos(theta), 2*b*np.sin(theta)])  # shape (2,N)
    
    # Transform ellipse back to 3D
    ellipse_3d = pca.inverse_transform(ellipse_2d.T)
    
    # --- Plotly 3D ---
    pio.renderers.default = "browser"
    
    fig = go.Figure()
    
    # Original points
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=2, color="blue"),
        name="Points"
    ))
    
    # Ellipse embedded in 3D
    fig.add_trace(go.Scatter3d(
        x=ellipse_3d[:,0],
        y=ellipse_3d[:,1],
        z=ellipse_3d[:,2],
        mode="lines",
        line=dict(color="red", width=4),
        name="PCA Ellipse"
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        width=900,
        height=700,
        showlegend=True
    )
    
    fig.show()