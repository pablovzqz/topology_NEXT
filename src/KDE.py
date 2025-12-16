import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity

from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter, maximum_filter

from mpl_toolkits.mplot3d import Axes3D




def kde_ridge_statistical(positions, energies, bins=60, bandwidth=None, smooth_sigma=2, axis_z=0, axis_y=1, max_jump=20):
    """
    Follows the ridge of maximum density in the Z-Y plane using Kernel Density Estimation (KDE).
    The ridge is traced by finding the maximum density point at each Z slice, with a penalty
    for large jumps in Y to ensure spatial continuity.
    """
    positions = np.asarray(positions)
    z = positions[:, axis_z]
    y = positions[:, axis_y]
    
    data = np.vstack([z, y]).T
    n, d = data.shape

    # Computing bandwidth for each individual case
    if bandwidth is None:

        sigma = np.mean(np.std(data, axis=0))
        bandwidth = sigma * n ** (-1. / (d + 4))
    
    # Computes the KDE -> get the probability of finding an electron in that position.
    kde = KernelDensity(bandwidth = bandwidth, kernel='gaussian')
    kde.fit(data, sample_weight = energies)
    
    # Create z bins and centers for the search at each z position
    z_bins = np.linspace(z.min(), z.max(), bins + 1)
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    
    # Same thing for Y
    y_min, y_max = y.min(), y.max()
    y_search = np.linspace(y_min, y_max, 200)
    
    y_ridge = []
    y_prev = None

    # Computes the maximum and penalizes disconnectivity
    
    for z_c in z_centers:
        grid_points = np.column_stack([np.full_like(y_search, z_c), y_search])
        densities = np.exp(kde.score_samples(grid_points))
        
        if y_prev is None:
            idx_max = np.argmax(densities)
        else:
            dist = np.abs(y_search - y_prev)
            weights = densities * np.exp(-dist**2 / (2 * max_jump**2))
            idx_max = np.argmax(weights)
        
        y_ridge.append(y_search[idx_max])
        y_prev = y_search[idx_max]
    
    # Smoothing
    y_ridge = gaussian_filter1d(y_ridge, sigma=smooth_sigma)
    
    return z_centers, y_ridge

def find_peaks_with_smoothing(positions, energies, bandwidth=None, 
                               grid_resolution=50,
                               smooth_sigma=1.5,
                               thr = 0.5):
    """
    Finds peaks in the KDE by applying additional Gaussian smoothing before peak detection.
    Useful if the KDE has too much noise.
    """        
    positions = np.asarray(positions)
    z, y = positions[:, 0], positions[:, 1]
    
    # KDE
    data = np.vstack([z, y]).T
    n, d = data.shape
    if bandwidth is None:
        sigma = np.mean(np.std(data, axis=0))
        bandwidth = sigma * n ** (-1. / (d + 4))
    
    kde = KernelDensity(bandwidth = bandwidth, kernel='gaussian')
    kde.fit(data, sample_weight= energies)
    
    z_grid = np.linspace(z.min(), z.max(), grid_resolution)
    y_grid = np.linspace(y.min(), y.max(), grid_resolution)

    Z, Y = np.meshgrid(z_grid, y_grid)
    grid_points = np.column_stack([Z.ravel(), Y.ravel()])
    
    densities = np.exp(kde.score_samples(grid_points))
    density_grid = densities.reshape(Z.shape)

    density_smooth = gaussian_filter(density_grid, sigma=smooth_sigma)
    
    local_max = maximum_filter(density_smooth, size=5) == density_smooth
    max_density = density_smooth.max()
    threshold = max_density * thr


    local_max = local_max & (density_smooth > threshold)
    
    # Extract peak information
    peaks_indices = np.argwhere(local_max)
    peaks = []
    
    for idx in peaks_indices:
        i, j = idx
        peaks.append({
            'z': Z[i, j],
            'y': Y[i, j],
            'density': density_smooth[i, j],
            'density_original': density_grid[i, j]
        })
    
    peaks = sorted(peaks, key=lambda x: x['density'], reverse=True)
    
    return peaks[:2], (Z, Y, density_grid, density_smooth)

def peak_finder_3D(positions, energies, bandwidth=None, grid_resolution=50, 
                   smooth_sigma=1.5, dimensions=3, thr = 0.5):
    """
    Applies the 2D peak finding function to multiple projections in a 3D space.
    
    Parameters:
    -----------
    positions : array-like, shape (n_points, dimensions)
        Positions of the points. First column is X (common),
        rest are Y, Z, etc.
    energies : array-like, shape (n_points,)
        Energies associated with each point         
    bandwidth : float, optional
        bandwidth for the KDE
    grid_resolution : int
        Resolution of the grid
    smooth_sigma : float
        Sigma for the Gaussian smoothing
    dimensions : int
        NNumber of dimensions (includes X)
        
    Returns:
    --------
    all_peaks : dict
        Dictionary with peaks for each projection
    """
    positions = np.asarray(positions)
    peak_pos = []
    
    all_peaks = {}
    projection_names = ['X', 'Y']  # Names for additional dimensions
    
    for i in range(dimensions - 1):
        # Create a 2D array with X (column 0) and the current dimension
        positions_2d = positions[:, [0, i + 1]]  # Selects X and the i-th dimension
        
        projection_name = f'Z{projection_names[i]}'
        
        print(f"Buscando picos en proyección {projection_name}...")
        
        # Call your existing function without modifying it
        peaks,  (Z, Y, density_grid, density_smooth) = find_peaks_with_smoothing(
            positions_2d, energies,
            bandwidth=bandwidth,
            grid_resolution=grid_resolution,
            smooth_sigma=smooth_sigma,
            thr = thr
        )
        
        # Sort peaks by the Z coordinate (from lowest to highest)
        if peaks is not None and len(peaks) > 0:
            peaks = sorted(peaks, key=lambda p: p['z'])
        
        peak_pos.append(peaks)
        all_peaks[projection_name] = peaks
        
        if peaks is not None:
            print(f"  → Encontrados {len(peaks)} picos en {projection_name}")
        else:
            print(f"  → No se encontraron picos en {projection_name}")
    
    return all_peaks, peak_pos

def visualize_peaks_3d(positions, energies, bandwidth=None):
    """
    Visualizes the peaks found in the 2D KDE of (z, y) positions in a 3D plot.
    """
    peaks, (Z, Y, density_grid, density_smooth) = find_peaks_with_smoothing(positions, energies, bandwidth, grid_resolution = 100)
    density_grid = density_smooth
    
    # Sort peaks by Z (from lowest to highest)
    peaks = sorted(peaks, key=lambda p: p['z'])
    
    z, y = positions[:, 0], positions[:, 1]
    data = np.vstack([z, y]).T
    n, d = data.shape
    
    if bandwidth is None:
        sigma = np.mean(np.std(data, axis=0))
        bandwidth = sigma * n ** (-1. / (d + 4))
    
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(data, sample_weight=energies)
    
    z_grid = np.linspace(z.min(), z.max(), 50)
    y_grid = np.linspace(y.min(), y.max(), 50)
    Z, Y = np.meshgrid(z_grid, y_grid)
    grid_points = np.column_stack([Z.ravel(), Y.ravel()])
    
    densities = np.exp(kde.score_samples(grid_points))
    density_grid = densities.reshape(Z.shape)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface plot
    surf = ax.plot_surface(Z, Y, density_grid, cmap='viridis', 
                          alpha=0.7, edgecolor='none')
    
    # Scatter plot of original points
    if peaks:
        for i, peak in enumerate(peaks[:2]):
            color = 'red' if i == 0 else 'cyan'
            ax.scatter([peak['z']], [peak['y']], [peak['density']], 
                      c=color, marker='o', s=50, edgecolors='black', 
                      linewidths=2, label=f"Peak {i+1} (ρ={peak['density']:.6f})", 
                      zorder=10)
    
    ax.set_xlabel('Z', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Densidad', fontsize=12)
    ax.set_title('KDE as a function of (z, y)', fontsize=14)
    ax.legend()
    # plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()
    
    for i, peak in enumerate(peaks[:2]):
        print(f"\nPico {i+1}:")
        print(f"  Z = {peak['z']:.2f}")
        print(f"  Y = {peak['y']:.2f}")
        print(f"  Density = {peak['density']:.8f}")
    
    return peaks
