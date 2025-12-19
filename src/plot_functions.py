import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Arc
import plotly.graph_objects as go
import plotly.io as pio

from cycler     import cycler
from contextlib import contextmanager


color_sequence = ("k", "m", "g", "b", "r",
                  "gray", "aqua", "gold", "lime", "purple",
                  "brown", "lawngreen", "tomato", "lightgray", "lightpink")

def auto_plot_style(overrides = dict()):
    plt.rcParams[ "figure.figsize"               ] = 10, 8
    plt.rcParams[   "font.size"                  ] = 20
    plt.rcParams[  "lines.markersize"            ] = 25
    plt.rcParams[  "lines.linewidth"             ] = 3
    plt.rcParams[  "patch.linewidth"             ] = 3
    plt.rcParams[   "axes.linewidth"             ] = 2
    plt.rcParams[   "grid.linewidth"             ] = 3
    plt.rcParams[   "grid.linestyle"             ] = "--"
    plt.rcParams[   "grid.alpha"                 ] = 0.5
    plt.rcParams["savefig.dpi"                   ] = 300
    plt.rcParams["savefig.bbox"                  ] = "tight"
    plt.rcParams[   "axes.formatter.use_mathtext"] = True
    plt.rcParams[   "axes.formatter.limits"      ] = (-3 ,4)
    plt.rcParams[  "xtick.major.size"            ] = 8
    plt.rcParams[  "ytick.major.size"            ] = 8
    plt.rcParams[  "xtick.minor.size"            ] = 5
    plt.rcParams[  "ytick.minor.size"            ] = 5
    plt.rcParams[   "axes.prop_cycle"            ] = cycler(color=color_sequence)
    plt.rcParams[  "image.cmap"                  ] = "gnuplot2"
    plt.rcParams.update(overrides)



@contextmanager
def temporary(name, new_value):
    old_value          = plt.rcParams[name]
    plt.rcParams[name] = new_value
    try    : yield
    finally: plt.rcParams[name] = old_value


def normhist(x, *args, normto=100, normfactor=None, **kwargs):
    if "histtype" not in kwargs:
        kwargs["histtype"] = "step"
    if normfactor is None:
        w = np.full(len(x), normto/len(x))
    else:
        w = np.full(len(x), normfactor)
    return plt.hist(x, *args, weights=w, **kwargs)


def normhist2d(x, y, *args, normto=100, normfactor=None, **kwargs):
    if normfactor is None:
        w = np.full(len(x), normto/len(x))
    else:
        w = np.full(len(x), normfactor)
    if "cmin" not in kwargs:
        kwargs["cmin"] = w[0]
    return plt.hist2d(x, y, *args, weights=w, **kwargs)


def plot_circular_sectors(sector_angle, radial_bins_per_sector, radius=480, center=(0, 0), dpi=180):
    """
    Plot the radial sector division used for the analysis.

    Parameters:
        sector_angle : float
            Angular width of each sector in degrees (e.g., 60 gives 6 sectors).
        radial_bins_per_sector : int
            Number of radial divisions (rings) per sector.
        radius : float
            Maximum radius of the circle.
        center : tuple
            Center of the circle (default is (0, 0)).
        dpi : int
            DPI for the figure.
    """
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    ax.set_aspect('equal')

    # Draw outer circle
    circle = plt.Circle(center, radius, fill=False, color='k', linewidth=2)
    ax.add_patch(circle)

    assert 360 % sector_angle == 0, "sector_angle must divide 360 evenly"
    n_sectors = 360 // sector_angle

    # Draw each sector
    for i in range(n_sectors):
        start_angle = i * sector_angle
        end_angle = (i + 1) * sector_angle
        color = color_sequence[i % len(color_sequence)]

        # Sector wedge
        wedge = Wedge(center, radius, start_angle, end_angle, 
                      facecolor=color, alpha=0.5, edgecolor='k')
        ax.add_patch(wedge)

        # Angle label
        mid_angle = (start_angle + end_angle) / 2
        label_radius = radius * 0.5
        x = label_radius * np.cos(np.radians(mid_angle))
        y = label_radius * np.sin(np.radians(mid_angle))
        ax.text(x, y, f'{start_angle:.0f}°-{end_angle:.0f}°', 
                ha='center', va='center', fontsize=12)

    # Radial lines
    for angle in np.arange(0, 360, sector_angle):
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        ax.plot([0, x], [0, y], 'k-', linewidth=1, alpha=0.5)

    # Radial arcs and labels in first sector only
    radial_bins = np.linspace(0, radius, radial_bins_per_sector + 1)
    for r in radial_bins[1:-1]:
        arc = Arc(center, 2 * r, 2 * r, angle=0,
                  theta1=0, theta2=sector_angle,
                  color='k', linewidth=1, alpha=0.7)
        ax.add_patch(arc)

        # Label inside the first sector
        label_angle = sector_angle / 6  # ~center-ish of first wedge
        x = r * np.cos(np.radians(label_angle))
        y = r * np.sin(np.radians(label_angle))
        ax.text(x, y, f'{r:.0f}', backgroundcolor='white',
                ha='center', va='center', fontsize=10)

    # Final plot formatting
    ax.set_xlim(-radius * 1.1, radius * 1.1)
    ax.set_ylim(-radius * 1.1, radius * 1.1)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    
    plt.tight_layout()
    plt.show()


def plot_3D_points_with_Q(df, primary_path_points=None, extreme_points=None, true_points=None, ellipse_size=None,title = None ):
    """
    Plots 3D points colored by Q and optionally overlays a primary path and transparent spheres
    around two extreme points.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'X', 'Y', 'Z', 'Q'.
    primary_path_points : np.ndarray, optional
        Array of shape (N,3) containing 3D points along the primary path.
    extreme_points : tuple of np.ndarray, optional
        Tuple (pt1, pt2) of 3D coordinates. If provided, draw spheres around them.
    sphere_radius : float
        Radius of the spheres around extreme points.
    """
    # Extract arrays
    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()
    z = df["Z"].to_numpy()
    Q = df["Q"].to_numpy()

    # Interactive plot
    pio.renderers.default = "browser"
    fig = go.Figure()

    # 3D scatter with Q
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=5, color=Q, colorscale='Viridis', colorbar=dict(title="Q")),
    ))

    # Plot primary path if provided
    if primary_path_points is not None:
        fig.add_trace(go.Scatter3d(
            x=primary_path_points[:, 0],
            y=primary_path_points[:, 1],
            z=primary_path_points[:, 2],
            mode='lines',
            line=dict(color='red', width=5),
        ))

    # Plot spheres and black markers if extreme points provided
    if extreme_points is not None:

        pt1, pt2 = extreme_points
        
        add_spheres_to_fig(fig, points=(pt1, pt2), a=ellipse_size[0], b=ellipse_size[1], colors=['orange','magenta'], opacity=0.2)

        # Black markers at extreme points
        fig.add_trace(go.Scatter3d(
            x=[pt1[0], pt2[0]],
            y=[pt1[1], pt2[1]],
            z=[pt1[2], pt2[2]],
            mode='markers',
            marker=dict(size=6, color='black', symbol='diamond'),
        ))
    
    # Plot spheres and black markers if extreme points provided
    if true_points is not None:

        pt1, pt2 = true_points
        
        # Black markers at extreme points
        fig.add_trace(go.Scatter3d(
            x=[pt1[0], pt2[0]],
            y=[pt1[1], pt2[1]],
            z=[pt1[2], pt2[2]],
            mode='markers',
            marker=dict(size=6, color='black', symbol='cross'),
        ))

    # Build a cube around all plotted points so axes share the same span
    coords = [np.column_stack((x, y, z))]
    if primary_path_points is not None:
        coords.append(np.asarray(primary_path_points))
    if extreme_points is not None:
        coords.append(np.vstack(extreme_points))
    if true_points is not None:
        coords.append(np.vstack(true_points))

    all_points = np.vstack(coords)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2
    max_range = float(np.max(maxs - mins))
    half_range = max_range / 2 if max_range > 0 else 1.0

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=[center[0] - half_range, center[0] + half_range]),
            yaxis=dict(range=[center[1] - half_range, center[1] + half_range]),
            zaxis=dict(range=[center[2] - half_range, center[2] + half_range]),
            aspectmode='cube',  # enforce equal scaling across axes
        ),
        width=1200,
        height=900,
        title=title
    )

    fig.show()


def plot_3D_points_with_Q_free_axes(df, primary_path_points=None, extreme_points=None, true_points=None, ellipse_size=None, title=None):
    """
    Copy of plot_3D_points_with_Q without enforcing equal axis ranges.
    """
    # Extract arrays
    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()
    z = df["Z"].to_numpy()
    Q = df["Q"].to_numpy()

    # Interactive plot
    pio.renderers.default = "browser"
    fig = go.Figure()

    # 3D scatter with Q
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=5, color=Q, colorscale='Viridis', colorbar=dict(title="Q")),
    ))

    # Plot primary path if provided
    if primary_path_points is not None:
        fig.add_trace(go.Scatter3d(
            x=primary_path_points[:, 0],
            y=primary_path_points[:, 1],
            z=primary_path_points[:, 2],
            mode='lines',
            line=dict(color='red', width=5),
        ))

    # Plot spheres and black markers if extreme points provided
    if extreme_points is not None:

        pt1, pt2 = extreme_points
        
        add_spheres_to_fig(fig, points=(pt1, pt2), a=ellipse_size[0], b=ellipse_size[1], colors=['orange','magenta'], opacity=0.2)

        # Black markers at extreme points
        fig.add_trace(go.Scatter3d(
            x=[pt1[0], pt2[0]],
            y=[pt1[1], pt2[1]],
            z=[pt1[2], pt2[2]],
            mode='markers',
            marker=dict(size=6, color='black', symbol='diamond'),
        ))
    
    # Plot spheres and black markers if extreme points provided
    if true_points is not None:

        pt1, pt2 = true_points
        
        # Black markers at extreme points
        fig.add_trace(go.Scatter3d(
            x=[pt1[0], pt2[0]],
            y=[pt1[1], pt2[1]],
            z=[pt1[2], pt2[2]],
            mode='markers',
            marker=dict(size=6, color='black', symbol='cross'),
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        width=1200,
        height=900,
        title=title
    )

    fig.show()



def add_spheres_to_fig(fig, points, a=0.5, b=0.5, colors=None, opacity=0.2):
    """
    Add smooth transparent ellipsoids at specified points to a Plotly 3D figure.
    
    Parameters
    ----------
    fig : go.Figure
        The existing figure to which ellipsoids will be added.
    points : list or array of shape (N,3)
        List of 3D points for ellipsoid centers.
    a : float
        Longer axis along Z direction.
    b : float
        Shorter axis along XY plane.
    colors : list of str
        Colors for each ellipsoid.
    opacity : float
        Transparency (0-1).
    """

    # Create spherical coordinates: phi (polar angle 0 to π) and theta (azimuthal 0 to 2π)
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2*np.pi, 40)
    phi, theta = np.meshgrid(phi, theta)
    
    # Generate ellipsoid surface coordinates (unit sphere scaled to ellipsoid)
    x_sphere = np.sin(phi) * np.cos(theta)
    y_sphere = np.sin(phi) * np.sin(theta)
    z_sphere = np.cos(phi)
    
    for idx, center in enumerate(points):
        # Translate ellipsoid to center position and scale by a (Z) and b (XY)
        x = center[0] + b * x_sphere
        y = center[1] + b * y_sphere
        z = center[2] + a * z_sphere
        
        # Use Surface for proper parametric surface rendering
        sphere_color = colors[idx] if colors else 'blue'
        fig.add_trace(go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=np.ones_like(z),  # Uniform surface
            colorscale=[[0, sphere_color], [1, sphere_color]],
            showscale=False,
            opacity=opacity,
            name=f'Sphere {idx+1}'
        ))


def plot_3D_points_with_ori_track(df, other_df=None, title=None):
    """
    Plots 3D points colored by Q from df and optionally adds points from another dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'X', 'Y', 'Z', 'Q'.
    other_df : pandas.DataFrame, optional
        DataFrame with columns 'x', 'y', 'z' to plot as additional points.
    title : str, optional
        Title for the plot.
    """
    # Extract arrays from main df
    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()
    z = df["Z"].to_numpy()
    Q = df["Q"].to_numpy()

    # Interactive plot
    pio.renderers.default = "browser"
    fig = go.Figure()

    # 3D scatter with Q
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=5, color=Q, colorscale='Viridis', colorbar=dict(title="Q")),
    ))

    # Plot points from other_df if provided
    if other_df is not None:
        x_other = other_df["x"].to_numpy()
        y_other = other_df["y"].to_numpy()
        z_other = other_df["z"].to_numpy()
        
        fig.add_trace(go.Scatter3d(
            x=x_other, y=y_other, z=z_other,
            mode='markers',
            marker=dict(size=5, color='red'),
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=1200,
        height=900,
        title=title
    )

    fig.show()