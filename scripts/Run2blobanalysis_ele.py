import tables
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import importlib as il
from tqdm import tqdm


import sys
sys.path.append("../src/")

import clustering_fun as cf
import analysis_functions as af
import selector as sl
import MC_topology as MCtp
import Topology_functions as tf


# Read from file
params_file = sys.argv[1] 

with open(params_file, "r") as f:
    line = f.readline().strip()

# Split by comma and convert to integers/floats
type_str, Q_thr_frac, bubble_rad, first_ev, last_ev = line.split(",")
type = type_str.strip()
Q_thr_frac = float(Q_thr_frac)
bubble_rad = int(bubble_rad)
first_ev = int(first_ev)
last_ev = int(last_ev)

pressure = 15

print(type ,Q_thr_frac, bubble_rad, first_ev, last_ev)

filename = f"/scratch/torellis/ITACA/GXe_TrackAnalysis/data/GXe_{type}_{pressure}bar_100mum_step.next.h5"
print(filename)

file = tables.open_file(filename, mode="r") 

# Diffusion coefficient from here: 
#https://iopscience.iop.org/article/10.1088/1748-0221/14/08/P08009/pdf 
#Diff coefficient longitudinal

diff_coeff_L_45_PureXe = 4.3 / 10**2 # mm/sqrt(mm)
diff_coeff_L_30_PureXe = 5.1 / 10**2 # mm/sqrt(mm)
diff_coeff_L_15_PureXe = 7.3 / 10**2 #mm/sqrt(mm)

diff_coeff_L_dict = {
    45: diff_coeff_L_45_PureXe,
    30: diff_coeff_L_30_PureXe,
    15: diff_coeff_L_15_PureXe
}

diff_coeff_T_45_PureXe = 1.6 / 10**1 # mm/sqrt(mm)
diff_coeff_T_30_PureXe = 2.0 / 10**1 # mm/sqrt(mm)
diff_coeff_T_15_PureXe = 2.8 / 10**1 #mm/sqrt(mm)

diff_coeff_T_dict = {
    45: diff_coeff_T_45_PureXe,
    30: diff_coeff_T_30_PureXe,
    15: diff_coeff_T_15_PureXe
}

diff_coeff_L = diff_coeff_L_dict[pressure]

diff_coeff_ionT_PureXe = 0.031 #mm/sqrt(mm)
diff_coeff_ionL_PureXe = 0.031/4 #mm/sqrt(mm)

drift_velocity_45_PureXe_ions = 12.6 * 10**(-5) #mm/mus  for ions
drift_velocity_30_PureXe_ions = 20 * 10**(-5) #mm/mus   for ions
drift_velocity_15_PureXe_ions = 37.8 * 10**(-5) #mm/mus   for ions

############################################################
max_step = 0.1                                         #####<------------CHANGE
diff_coeff_L = diff_coeff_L_dict[pressure]             #####<------------CHANGE
diff_coeff_T = diff_coeff_T_dict[pressure]                  #####<------------CHANGE
drift_velocity = drift_velocity_15_PureXe_ions         #####<------------CHANGE
############################################################

# Read MC/hits table
hits_node = file.get_node("/MC/hits")
df_hits = pd.DataFrame.from_records(hits_node.read(), columns=hits_node.colnames)

# Read MC/particles table
particles_node = file.get_node("/MC/particles")
df_particles = pd.DataFrame.from_records(particles_node.read(), columns=particles_node.colnames)

#Select ionizing particles
df_epem = df_particles[sl.ionizing(df_particles)]
df_hits_epem = df_hits.merge(df_epem[['event_id', 'particle_id']], on=['event_id', 'particle_id'], how='inner')

df_hits_epem = df_hits_epem[ (df_hits_epem['event_id'] > first_ev) & (df_hits_epem['event_id'] < last_ev) ] 

########## make clusters and select higher energy track
df_hits_epem  = cf.cluster_kdtree(df_hits_epem, 2*max_step)
df_hits_epem_HET = sl.filter_HE_cluster(df_hits_epem)

########## rescale for positive diffusion
shift_z = df_hits_epem_HET['z'].min()
df_hits_epem_HET['z'] = df_hits_epem_HET['z'] - shift_z

########## Calculate the number of e- ion pairs
wval = 0.000015 #w-value from https://journals.aps.org/pra/abstract/10.1103/PhysRevA.12.1771
df_hits_epem_HET = af.compute_el_ion_pairs(df_hits_epem_HET,wval)

########## Get original track extremities
extremity_df_onubb = MCtp.build_extremities_df(df_hits_epem_HET)

results = []

for event_id, df in tqdm(df_hits_epem_HET.groupby("event_id"), total=df_hits_epem_HET['event_id'].nunique()):

    #get true endpoints
    true_p1 = extremity_df_onubb.loc[extremity_df_onubb['event_id']==event_id, 'p1'].iloc[0]
    true_p2 = extremity_df_onubb.loc[extremity_df_onubb['event_id']==event_id, 'p2'].iloc[0]

    #get the 3D map of diffused hits
    df_counts = MCtp.build_3d_counts_df_ele(df_hits_epem_HET[df_hits_epem_HET['event_id'] == event_id], diff_coeff_T, diff_coeff_L,2)

    #equalize space
    dx,dy,dz = tf.compute_min_axis_spacing(df_counts)
    df_counts['X'] = df_counts['X']/dx
    df_counts['Y'] = df_counts['Y']/dy
    df_counts['Z'] = df_counts['Z']/dz
    true_p1= true_p1 / np.array([dx, dy, dz])
    true_p2= true_p2 / np.array([dx, dy, dz])

    df_counts = tf.kde_gradient_filter(df_counts,kernel_size=11,bandwidth=.7,alpha=0.9)
    
    Q_thr = Q_thr_frac*df_counts['Q'].max()
    df_counts_thr = df_counts[df_counts['Q'] > Q_thr ]
    
    #get primary path
    try:
        primary_path_points = tf.compute_primary_path_fast(df_counts_thr,3)
    except:
        continue
        
    #smooth path
    smootherd_path = tf.reconstruct_path_ellipse(df_counts_thr, primary_path_points, ellipse_size=(2,2))
    
    #get extremities
    pt1 = smootherd_path[0]
    pt2 = smootherd_path[-1]

    sigmaxy = af.compute_sigma_tr(max(100,df_counts['Z'].mean()),diff_coeff_T)
    sigmaz = af.compute_sigma_lon(max(100,df_counts['Z'].mean()),diff_coeff_L)

    ax1 = bubble_rad*sigmaxy/dx
    ax2 = bubble_rad*sigmaz/dz    

    Q1 = MCtp.sum_Q_in_ellipsoid(pt1, df_counts, ellipse_size=(ax2,ax1))
    Q2 = MCtp.sum_Q_in_ellipsoid(pt2, df_counts, ellipse_size=(ax2,ax1))

    results.append({
        "event_id": event_id,
        "true_p1": true_p1,
        "pt1": pt1,
        "Q1": Q1,
        "true_p2": true_p2,
        "pt2": pt2,
        "Q2": Q2,
    })
    
df_results = pd.DataFrame(results)
def float2str_p(val):
    return str(val).replace('.', 'p')

filename = f"/scratch/torellis/ITACA/out_elediff_{pressure}bar/elec_diff_{type}_{float2str_p(Q_thr_frac)}_{bubble_rad}_{first_ev}_{last_ev}.csv"
df_results.to_csv(filename, index=False)




