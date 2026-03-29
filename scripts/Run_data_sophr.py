import tables
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import importlib as il

import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
import re

from sklearn.cluster import DBSCAN

import sys
sys.path.append("../src")

import clustering_fun as cf
import plotter as pl
import analysis_functions as af
import selector as sl
import MC_topology as MCtp
import Topology_functions as tf
import plot_functions as plf


if len(sys.argv) < 3:
    print("Usage: python Run_data_sophr.py <input_h5_file>")
    sys.exit(1)

with open(sys.argv[2], "r") as f:
    line = f.readline().strip()

# Split by comma and convert to integers/floats
type_str, Q_thr_frac, bubble_rad = line.split(",")
type = type_str.strip()
Q_thr_frac = float(Q_thr_frac)
bubble_rad = int(bubble_rad)

input_file = sys.argv[1]
# Extract the number before the first dot
match = re.search(r'(\d+)\.', input_file)
number = int(match.group(1))

df_hits = pd.read_hdf(input_file, key='hits')
df_hits = cf.clusterize_hits_sophr(df_hits)
event_list = df_hits['event'].unique()

vdrift = 1420 #mm/s
Dt_sq = 0.306 #mm^2/mm
Dl_sq = 0.020 #mm^2/mm
a = 14.55  # XY scale
b = 3.7  # Z scale

results = []

for ev in event_list:
    df_event = df_hits[(df_hits['event'] == ev) & (df_hits['cluster']>-1) ]

    sig_t = np.sqrt(Dt_sq * df_event['Z'].mean())
    sig_l = np.sqrt(Dl_sq * df_event['Z'].mean())   

    # Get the HE cluster
    cluster_E_sum = df_event.groupby('cluster')['E'].sum()
    best_cluster = cluster_E_sum.idxmax()
    df_counts = df_event[df_event['cluster'] == best_cluster]

    df_counts_scaled = df_counts.copy()
    df_counts_scaled['X'] = df_counts['X']/a
    df_counts_scaled['Y'] = df_counts['Y']/a
    df_counts_scaled['Z'] = df_counts['Z']/b

    sig_t = sig_t/a
    sig_l = sig_l/b

    Q_thr = Q_thr_frac*df_counts['Q'].max()

    df_counts_thr = df_counts_scaled[df_counts_scaled['Q'] > Q_thr ]
    df_counts_thr_filt = tf.kde_gradient_filter(df_counts_thr,kernel_size=11,bandwidth=.7,alpha=0.9)
    df_counts_thr_filt = tf.kde_gradient_filter(df_counts_thr,kernel_size=11,bandwidth=.7,alpha=0.9)

    primary_path_points = tf.compute_primary_path_fast(df_counts_thr_filt,100)

    smootherd_path = tf.reconstruct_path_ellipse(df_counts_thr_filt, primary_path_points, ellipse_size=(4,4))

    pt1 = smootherd_path[0]
    pt2 = smootherd_path[-1]

    ax1 = bubble_rad*sig_t
    ax2 = bubble_rad*sig_l

    Q1 = MCtp.sum_Q_in_ellipsoid(pt1, df_counts_scaled, ellipse_size=(ax2,ax1))
    Q2 = MCtp.sum_Q_in_ellipsoid(pt2, df_counts_scaled, ellipse_size=(ax2,ax1))
    print(Q1,Q2)

    results.append({
        "event_id": int(ev),
        "Q1": Q1,
        "Q2": Q2,
    })

df_results = pd.DataFrame(results)
def float2str_p(val):
    return str(val).replace('.', 'p')

filename = f"/Users/samuele/Documents/Postdoc/NEXT/NEXTTopologyAnalysis/analyzed/elec_diff_{type}_{float2str_p(Q_thr_frac)}_{bubble_rad}.csv"
df_results.to_csv(filename, index=False)
