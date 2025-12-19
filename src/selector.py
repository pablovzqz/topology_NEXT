def radiogenic_gammas(df_particles):
    
    return (df_particles['particle_name'] == b'gamma') & (df_particles['creator_proc'] == b'Radioactivation')

def ionizing(df_particles):
    
    return (df_particles['particle_name'] == b'e+') | (df_particles['particle_name'] == b'e-')

def filter_HE_cluster(df):
    """
    For each event, keep only hits belonging to the cluster (label)
    with the maximum total energy.
    Expects columns: ['event_id', 'label', 'energy'].
    """
    # Sum energy per (event_id, label)
    energy_sum = df.groupby(["event_id", "label"])["energy"].sum().reset_index()

    # Select the label with max energy per event
    max_labels = energy_sum.loc[energy_sum.groupby("event_id")["energy"].idxmax(), ["event_id", "label"]]

    # Keep only hits with those labels
    df_max = df.merge(max_labels, on=["event_id", "label"], how="inner")
    return df_max

def filter_in_ROI(df, Emin, Emax):
    """
    Select events with total energy between Emin and Emax (keV).
    """
    total_E = df.groupby("event_id")["energy"].sum()
    valid_ids = total_E[(total_E > Emin) & (total_E < Emax)].index
    return df[df["event_id"].isin(valid_ids)]

def selectXRay(df):
    return (df['particle_name'] == b'gamma') & (df['creator_proc'] == b'phot')

def selectkshell(df):
    return (df["kin_energy"]>0.028) & (df["kin_energy"]<0.036)

def selectlshell(df):
    return (df["kin_energy"]>0.0035) & (df["kin_energy"]<0.0048)

def selectkshell_exact(df):
    return  ( (df["kin_energy"]>0.029780) & (df["kin_energy"]<0.029782) ) | ( (df["kin_energy"]>0.029451) & (df["kin_energy"]<0.029453) ) | ( (df["kin_energy"]>0.033628) & (df["kin_energy"]<0.033630) ) | ( (df["kin_energy"]>0.033566) & (df["kin_energy"]<0.033568) ) | ( (df["kin_energy"]>0.034407) & (df["kin_energy"]<0.034409) ) | ( (df["kin_energy"]>0.034394) & (df["kin_energy"]<0.034396) )


def filter_hits_by_particles(df_selected, df_hits_cl):
    # Extract unique event IDs from df_selected
    selected_events = df_selected["event_id"].unique()

    # Filter df_hits_cl to keep only those events
    return df_hits_cl[df_hits_cl["event_id"].isin(selected_events)]