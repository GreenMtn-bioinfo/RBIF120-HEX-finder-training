import numpy as np
import math
from preparation_1 import structural_profiles, training_length
from profile_generator_3 import profiles_file_base_name, profiles_metadata_suffix
import json
import os


# INPUTS

# Establish paths for the profiles and profile metadata files
all_profiles_path = f'{structural_profiles}{profiles_file_base_name}.npy'
IDs_file_path = f'{structural_profiles}{profiles_file_base_name}{profiles_metadata_suffix}.npy'

# OUTPUTS

# Define the name of the train/test split and class label JASONs
split_json_path = f'{structural_profiles}partitioned_IDs.json'
new_labels_path = split_json_path.replace('_IDs.json', '_new_labels.npy')

# Script parameters
write_files = True # Safeguard to prevent accidental rerunning of this script and rewriting of train/test split (i.e. data leakage)
rng = np.random.default_rng(112263) # Set the seed for an RNG for consistent training/testing split for this data set
script_name = os.path.basename(__file__) # Used globally (not passed) in print statements to provide the name of this file (useful in of pipeline)


### Main functions:

# Utility functions for writing and reading JSONS
def write_json_pretty(object: dict, 
                      path: str,
                      verbose: bool = False):
    
    with open(path, mode='w', encoding='utf-8') as file:
        json.dump(object, file, ensure_ascii=False, indent=4)
        if verbose:
            print(f'write_json_pretty: Wrote {path}.')

def load_json(path: str,
              verbose: bool = False) -> dict:
    
    with open(path, mode='r') as file:
        jsn = json.load(file)
        if verbose:
            print(f'load_json: Loaded {path}.')
        return jsn

# Set aside the full training/validation/testing sets (DO NOT RERUN THIS FUNCTION WITH NEW RNG/SEED)
def train_test_split(catalogue_path: str = IDs_file_path, 
                     proportion_test: float = 0.2, 
                     split_file: str = split_json_path,
                     labels_file: str = new_labels_path, 
                     write_files: bool = True):
    
    # Prepare the relevant data structures (mem-map)
    IDs = np.load(catalogue_path, mmap_mode="r")
    
    # Loop through each sequence type and randomly select a number of IDs based on the training proportion
    seq_type_idx = 1 
    intr_type_idx = 3
    ids_idx = 0
    chroms_col = 4
    start_col = 5
    
    # Create a new numpy array in which the ID is parsed into the source chrom, start, and end coordinates
    print(f'{script_name}: Counting and preparing chromosomes/sources...')
    IDs_new = np.ndarray((IDs.shape[0], 7), dtype='<U40')
    for i in range(IDs_new.shape[0]):
        IDs_new[i,:chroms_col] = IDs[i,:chroms_col]
        label = str(IDs[i, ids_idx])
        label = label.split(':')
        IDs_new[i, chroms_col] = label[0]
        IDs_new[i, start_col:] = label[1].split('-')[0:2]
    chromosomes = list(set(IDs_new[:,chroms_col].tolist()))
    chromosomes.sort()
    
    sampled_test = []
    print(f'{script_name}: Beginning sampling...')
    for chromosome in chromosomes: # We split by sampling the last or first proportion_test of each chromosome to facilitate accurate sliding-window evaluation of performance in the test set
        
        # Sort the sampled profiles in a given chromosome by their start coordinates
        coord_starts = IDs_new[IDs_new[:, chroms_col] == chromosome, start_col].astype(np.int32)
        sort_indices = np.argsort(coord_starts[:])
        this_chrom_sorted = IDs_new[IDs_new[:, chroms_col] == chromosome, :chroms_col][sort_indices, :]
        
        # Account for the different types of introns sampled and update labels accordingly
        updated_seq_types = np.array([f'{str(row[seq_type_idx])}_{str(row[intr_type_idx])}' if str(row[intr_type_idx]) != '' else str(row[seq_type_idx]) for row in this_chrom_sorted[:,] ])
        seq_types = list(set(updated_seq_types))
        
        # Select a random region of the chromosome of size proportion_test
        chrom_length = this_chrom_sorted.shape[0]
        this_chrom_test_size = math.floor(chrom_length * proportion_test)
        n_splits = chrom_length // this_chrom_test_size
        splits = [slice(i * this_chrom_test_size, (i + 1) * this_chrom_test_size) if i != (n_splits - 1) else slice(i * this_chrom_test_size, chrom_length - 1) for i in range(n_splits)]
        this_chrom_test_slice = rng.choice(splits, size=1, replace=False).tolist()
        allowed_profiles = set(this_chrom_sorted[this_chrom_test_slice[0], ids_idx].tolist())
        
        # See which sequence types overlap with the randomly selected genomic region
        for seq_type in seq_types:
            print(f'{script_name}: Sampling {seq_type} from {chromosome}.')
            profile_subset = set(this_chrom_sorted[updated_seq_types == seq_type, ids_idx].tolist())
            selected_test_profiles = list(allowed_profiles.intersection(profile_subset))
            sampled_test = sampled_test + selected_test_profiles # profile_subset[this_class_test_slice[0]].tolist()
    
    # For convenience, also save and return the training/testing partition dictionary and class labels as .JSONs
    partitioned = {'train' : list(set(IDs[:, ids_idx].tolist()).difference(set(sampled_test))), 
                   'test' :  sampled_test}
    
    if write_files:
        print(f'{script_name}: Writing the files that track the data partition and labels...')
        write_json_pretty(partitioned, split_file, verbose=False)
        updated_seq_types = np.array([f'{str(row[seq_type_idx])}_{str(row[intr_type_idx])}' if str(row[intr_type_idx]) != '' else str(row[seq_type_idx]) for row in IDs[:,] ])
        np.save(labels_file, arr=updated_seq_types, allow_pickle=False)
    else:
        print(f'{script_name}: Finished. No files written.')

# Counts each class/sequence types and estimates their under-sampling factor relative to the human genome
def estimate_class_weights(assumptions: dict = {'human_genome_bp' : 3200000000,
                                                'percent_bp_exons' : 1.75 / 100, 
                                                'percent_bp_intergenic' : 75.0 / 100,
                                                'N_exons' : 202235,
                                                'N_genes' : 20000},
                           sample_seq_length = training_length + 1,
                           write_results: bool = False,
                           save_path: str = None,
                           verbose: bool = False) -> dict:
    
    assumptions['percent_bp_introns'] = 25.0 / 100 - assumptions['percent_bp_exons']
    
    # Load data on the training/testing split
    split = load_json(split_json_path, verbose=verbose)
    metadata = np.load(IDs_file_path, allow_pickle=verbose)
    new_labels = np.load(new_labels_path, allow_pickle=verbose)
    
    # Iterate through the sequence/sample types and tally how many of each are in the training and testing sets
    total_counts = np.zeros(len(set(new_labels)))
    proportions = []
    if verbose:
        print('estimate_class_weights(): counting each class in the data set...')
    for key in split:
        indices = np.argwhere(np.isin(metadata[:,0], split[key], assume_unique=True))
        labels = new_labels[indices]
        unique, counts = np.unique(labels, return_counts=True)
        if key == 'train': # In the name of preventing data leakage, only base sample weights on training set (later on)
            total_counts = np.add(total_counts, counts)
        proportions.append(np.array(counts)/np.sum(counts))
    
    # Prepare results for saving or return
    sample_proportions = np.array(total_counts)/np.sum(total_counts)
    sample_proportions = { label : float(sample_proportions[i]) for i, label in enumerate(unique.tolist()) }
    total_counts = { label : int(total_counts[i]) for i, label in enumerate(unique.tolist()) }
    
    # Sanity check, did we actually sample each class approximately evenly?
    err_sqd = np.sum(np.square(proportions[0] - proportions[1]))
    if verbose:
        print(f'estimate_class_weights(): training and testing class proportions have SSE of: {round(err_sqd, 4)}')
    
    ## Some back-of-the envelope high level math based on the available information on the human genome and my sampling numbers
    assumptions['exon_boundaries_bp'] = sample_seq_length * assumptions['N_exons'] * 2
    assumptions['exon-intron_bp'] = assumptions['exon_boundaries_bp'] / 2
    assumptions['intron-exon_bp'] = assumptions['exon_boundaries_bp'] / 2
    assumptions['control_exons_bp'] = assumptions['human_genome_bp']*assumptions['percent_bp_exons'] - (assumptions['N_exons'] * sample_seq_length)
    assumptions['control_introns_intron_bp'] = assumptions['human_genome_bp']*assumptions['percent_bp_introns'] - ((assumptions['N_exons'] - 1) * sample_seq_length)
    assumptions['control_introns_intergenic_bp'] = assumptions['human_genome_bp']*assumptions['percent_bp_intergenic'] - ((assumptions['N_genes'] - 1) * sample_seq_length + sample_seq_length)
    
    # Calculate how much each class has been under-sampled relative to the human genome
    undersampling_factors = { key : float(assumptions[f'{key}_bp'] / (total_counts[key] * sample_seq_length)) for key in total_counts }
    
    if write_results:
        write_json_pretty(undersampling_factors, f'{save_path}class_weights.json', verbose=verbose)
        write_json_pretty(total_counts, f'{save_path}class_counts.json', verbose=verbose)
        write_json_pretty(sample_proportions, f'{save_path}class_proportions.json', verbose=verbose)
    
    else:
        return total_counts, sample_proportions, undersampling_factors


### Call the functions defined above to split the data and write the files tracking the split and class membership
if __name__ == "__main__":
    
    train_test_split(write_files=write_files) # write_files is a safeguard to prevent split overwrite, change to True if splitting for the first time
    estimate_class_weights(write_results=write_files, save_path=structural_profiles, verbose=True)