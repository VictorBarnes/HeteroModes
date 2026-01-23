#%%
import h5py
import numpy as np
import json
from heteromodes.utils import get_project_root

PROJ_DIR = get_project_root()

# Parameters
id = 55
n_nulls = 500
species = "human"

#%%
# Load heteromap labels
with open(f"{PROJ_DIR}/data/heteromaps/{species}/heteromaps_config.json", "r") as f:
    config = json.load(f)
    hetero_labels = {key: val["label"] for key, val in config.items()}

print(f"Found {len(hetero_labels)} heteromap labels: {hetero_labels}")

#%%
# Add "None" to the list for homogeneous model
hetero_labels["None"] = "Homogeneous"

# Initialize dictionaries to store results for all hmap_labels
all_results = {}

for hmap_label in hetero_labels:
    print(f"\nProcessing {hmap_label}...")
    
    # Load model results
    file = f"{PROJ_DIR}/results/human/model_rest/group/id-{id}/crossval/{hmap_label}/best_model.h5"
    
    try:
        with h5py.File(file, 'r') as f:
            edge_fc_model = np.mean(np.array(f['results']['edge_fc_corr']).flatten())
            node_fc_model = np.mean(np.array(f['results']['node_fc_corr']).flatten())
            fcd_model = np.mean(np.array(f['results']['fcd_ks']).flatten())
        
        print(f"  Model metrics - edge_fc: {edge_fc_model:.4f}, node_fc: {node_fc_model:.4f}, fcd: {fcd_model:.4f}")
    except Exception as e:
        print(f"  Could not load model file: {e}")
        continue
    
    # Skip null computation for "None" (homogeneous model)
    if hmap_label == "None":
        all_results[hmap_label] = {
            'edge_fc_model': edge_fc_model,
            'node_fc_model': node_fc_model,
            'fcd_model': fcd_model,
            'edge_fc_null': None,
            'node_fc_null': None,
            'fcd_null': None,
            'n_nulls_computed': 0
        }
        continue
    
    # Load null results
    edge_fc_null, node_fc_null, fcd_null = [], [], []
    
    # Loop through null files
    for i in range(n_nulls):
        file = f"{PROJ_DIR}/results/human/model_rest/group/id-{id}/crossval/{hmap_label}/nulls/null-{i}/best_model.h5"
        
        try:
            with h5py.File(file, 'r') as f:
                edge_fc_null.append(np.mean(np.array(f['results']['edge_fc_corr']).flatten()))
                node_fc_null.append(np.mean(np.array(f['results']['node_fc_corr']).flatten()))
                fcd_null.append(np.mean(np.array(f['results']['fcd_ks']).flatten()))
        except:
            # Silently skip missing files
            continue
    
    # Convert to numpy arrays
    edge_fc_null = np.array(edge_fc_null)
    node_fc_null = np.array(node_fc_null)
    fcd_null = np.array(fcd_null)
    
    print(f"  Loaded {len(edge_fc_null)} null samples")
    
    # Store results
    all_results[hmap_label] = {
        'edge_fc_model': edge_fc_model,
        'node_fc_model': node_fc_model,
        'fcd_model': fcd_model,
        'edge_fc_null': edge_fc_null,
        'node_fc_null': node_fc_null,
        'fcd_null': fcd_null,
        'n_nulls_computed': len(edge_fc_null)
    }

#%%
# Calculate p-values for each heterogeneous model
print("\n" + "="*80)
print("CALCULATING P-VALUES")
print("="*80)

# Get homogeneous model values
edge_fc_homo = all_results["None"]['edge_fc_model']
node_fc_homo = all_results["None"]['node_fc_model']
fcd_homo = all_results["None"]['fcd_model']
obj_homo = edge_fc_homo + node_fc_homo + 1 - fcd_homo

print(f"\nHomogeneous model (None):")
print(f"  edge_fc: {edge_fc_homo:.4f}")
print(f"  node_fc: {node_fc_homo:.4f}")
print(f"  fcd: {fcd_homo:.4f}")
print(f"  objective: {obj_homo:.4f}")

# Store p-values for each model
pvalues = {}

for hmap_label in hetero_labels:
    if hmap_label == "None":
        continue
    
    data = all_results.get(hmap_label)
    if data is None or data['edge_fc_null'] is None or len(data['edge_fc_null']) == 0:
        print(f"\n{hmap_label}: No null data available, skipping")
        continue
    
    print(f"\n{hmap_label}:")
    
    # Get model values
    edge_fc_hetero = data['edge_fc_model']
    node_fc_hetero = data['node_fc_model']
    fcd_hetero = data['fcd_model']
    obj_hetero = edge_fc_hetero + node_fc_hetero + 1 - fcd_hetero
    
    # Get null distributions
    edge_fc_null = data['edge_fc_null']
    node_fc_null = data['node_fc_null']
    fcd_null = data['fcd_null']
    obj_null = edge_fc_null + node_fc_null + 1 - fcd_null
    
    # Calculate differences relative to homogeneous model
    edge_fc_null_diff = edge_fc_null - edge_fc_homo
    node_fc_null_diff = node_fc_null - node_fc_homo
    fcd_null_diff = fcd_null - fcd_homo
    obj_null_diff = obj_null - obj_homo
    
    edge_fc_hetero_diff = edge_fc_hetero - edge_fc_homo
    node_fc_hetero_diff = node_fc_hetero - node_fc_homo
    fcd_hetero_diff = fcd_hetero - fcd_homo
    obj_hetero_diff = obj_hetero - obj_homo
    
    # Calculate p-values (one-tailed: proportion of nulls > observed)
    n_nulls_computed = len(edge_fc_null)
    p_edge = np.sum(edge_fc_null_diff > edge_fc_hetero_diff) / n_nulls_computed
    p_node = np.sum(node_fc_null_diff > node_fc_hetero_diff) / n_nulls_computed
    p_fcd = np.sum(fcd_null_diff > fcd_hetero_diff) / n_nulls_computed
    p_obj = np.sum(obj_null_diff > obj_hetero_diff) / n_nulls_computed
    
    # Significance markers
    def sig_marker(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''
    
    print(f"  Edge FC:   {edge_fc_hetero_diff:+.4f}  (p = {p_edge:.4f} {sig_marker(p_edge)})")
    print(f"  Node FC:   {node_fc_hetero_diff:+.4f}  (p = {p_node:.4f} {sig_marker(p_node)})")
    print(f"  FCD:       {fcd_hetero_diff:+.4f}  (p = {p_fcd:.4f} {sig_marker(p_fcd)})")
    print(f"  Objective: {obj_hetero_diff:+.4f}  (p = {p_obj:.4f} {sig_marker(p_obj)})")
    
    # Store p-values
    pvalues[hmap_label] = {
        'p_edge_fc': p_edge,
        'p_node_fc': p_node,
        'p_fcd': p_fcd,
        'p_obj': p_obj,
        'edge_fc_diff': edge_fc_hetero_diff,
        'node_fc_diff': node_fc_hetero_diff,
        'fcd_diff': fcd_hetero_diff,
        'obj_diff': obj_hetero_diff,
        'obj_model': obj_hetero
    }

print("\n" + "="*80)

#%%
# Save all results to h5 file
output_file = f"{PROJ_DIR}/results/human/model_rest/group/id-{id}/crossval/all_nulls_summary.h5"

print(f"\nSaving results to {output_file}")

with h5py.File(output_file, 'w') as f:
    # Store metadata
    f.attrs['id'] = id
    f.attrs['n_nulls_requested'] = n_nulls
    f.attrs['species'] = species
    f.attrs['hmap_labels'] = json.dumps(hetero_labels)
    
    # Create a group for each hmap_label
    for hmap_label, results in all_results.items():
        grp = f.create_group(hmap_label)
        
        # Store model metrics
        grp.create_dataset('edge_fc_model', data=results['edge_fc_model'])
        grp.create_dataset('node_fc_model', data=results['node_fc_model'])
        grp.create_dataset('fcd_model', data=results['fcd_model'])
        
        # Store null distributions (if they exist)
        if results['edge_fc_null'] is not None:
            grp.create_dataset('edge_fc_null', data=results['edge_fc_null'])
            grp.create_dataset('node_fc_null', data=results['node_fc_null'])
            grp.create_dataset('fcd_null', data=results['fcd_null'])
        
        grp.attrs['n_nulls_computed'] = results['n_nulls_computed']
        
        # Store p-values if available
        if hmap_label in pvalues:
            pval_data = pvalues[hmap_label]
            grp.create_dataset('p_edge_fc', data=pval_data['p_edge_fc'])
            grp.create_dataset('p_node_fc', data=pval_data['p_node_fc'])
            grp.create_dataset('p_fcd', data=pval_data['p_fcd'])
            grp.create_dataset('p_obj', data=pval_data['p_obj'])
            grp.create_dataset('edge_fc_diff', data=pval_data['edge_fc_diff'])
            grp.create_dataset('node_fc_diff', data=pval_data['node_fc_diff'])
            grp.create_dataset('fcd_diff', data=pval_data['fcd_diff'])
            grp.create_dataset('obj_diff', data=pval_data['obj_diff'])
            grp.create_dataset('obj_model', data=pval_data['obj_model'])

print("Done!")

#%%
# Verify saved data
print("\nVerifying saved data...")
with h5py.File(output_file, 'r') as f:
    print(f"File attributes: {dict(f.attrs)}")
    print(f"\nGroups (hetero_labels): {list(f.keys())}")
    
    for hmap_label in f.keys():
        print(f"\n{hmap_label}:")
        grp = f[hmap_label]
        print(f"  Datasets: {list(grp.keys())}")
        print(f"  Model edge_fc: {grp['edge_fc_model'][()]:.4f}")
        print(f"  Model node_fc: {grp['node_fc_model'][()]:.4f}")
        print(f"  Model fcd: {grp['fcd_model'][()]:.4f}")
        print(f"  N nulls computed: {grp.attrs['n_nulls_computed']}")
        if 'edge_fc_null' in grp:
            print(f"  Null array shape: {grp['edge_fc_null'].shape}")
        if 'p_edge_fc' in grp:
            print(f"  P-values: edge={grp['p_edge_fc'][()]:.4f}, node={grp['p_node_fc'][()]:.4f}, fcd={grp['p_fcd'][()]:.4f}, obj={grp['p_obj'][()]:.4f}")

#%%
