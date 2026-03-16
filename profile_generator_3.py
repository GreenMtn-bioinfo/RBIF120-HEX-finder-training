import numpy as np
import pandas as pd
from npy_append_array import NpyAppendArray
import os
import re
import time
import math
import multiprocessing
import preparation_1 as prep # 1st script, required for important global variables across scripts

#### 3rd Script
# This script uses the parameters CSVs made from data supplied by Sharma et al., 2025 to calculate energy profiles for the sequences collected by the 2nd script.
# The profiles for each collection of sequences (e.g. sequence type such as intron-exon forward, control reverse, etc.) are saved in a .NPY file in 'output_dir'.
# Profiles are calculated as a moving average for each of the 28 parameters using a centered sliding window of tri or tetra-dinucleotide steps covering 27 nucleotides
# as described in Mishra et al. (2021), Sharma et al. (2023), and Sharma et al. (2025). Dataframes constructed from the parameters CSVs are used to translate each tri- or tetra-nucleotide step into the corresponding value per parameter.
# Variables 'all_seqs' and 'sample_size' (set in 'preparation.py') can be used to calculate profiles for all or a random subset of sequences. With 12 threads (2.2 GHz, ~16 GB RAM) it takes a little over 2 minutes to process ~40,000 sequences of length ~400.
# This is the longest step of the analysis and multiple cores are used to parallelize if possible, without parallelization entire data set can take on the order of hours or longer (RAM is not the limiting factor).


## Global variable assignments imported from 'preparation_1.py' script:

# Paths
annotation_path = prep.exon_annotation
selected_exon_seqs = prep.selected_exon_seqs
output_dir = prep.structural_profiles
param_table_paths = prep.param_table_paths

# Parameters
boundary_margin = prep.boundary_margin
window_len = prep.window_len
all_seqs = prep.all_seqs # Whether to calculate profiles for ALL sequences, or sample a random subset
sample_sizes = prep.sample_sizes # If sampling a subset, how many from each sequence sub-category (combo of intron-exon/exon-intron/control and forward/reverse) to randomly sample?
seq_len = boundary_margin * 2 + 1 + (window_len - 1)


## Local variables that are script-specific and/or hard-coded:

# Path-related

# INPUTS
strands = ['forward', 'reverse']
input_paths = {'control_exons' : f'{selected_exon_seqs}control_exons_',
               'control_introns' : f'{selected_exon_seqs}control_introns_',
               'intron-exon' : f'{selected_exon_seqs}intron-exon_boundaries_',
               'exon-intron' : f'{selected_exon_seqs}exon-intron_boundaries_'}

# OUTPUTS
profiles_file_base_name = 'all_profiles' # Name of the .NPY file the profiles are written to
profiles_metadata_suffix = '_IDs_labels' # this is appended to the base name above for the file containing class labels, IDs, strand, etc,

# Script parameter variables
threads = os.cpu_count() # Used by multiprocessing in profiles_batch()
placeholder = 'VOID' # Used as a place holder for partial steps at the end of reading frames, cannot contain the character 'N'
skip_seqs_w_Ns = True # Determines whether sequences containing Ns should be skipped (True), or handled via a sort of hacky average-based imputation (False)
script_name = os.path.basename(__file__) # Used globally (not passed) in print statements to provide the name of this file (useful in of pipeline)


### Main functions:

# Facilitates reading in the parameter table CSVs from Sharma et al. 2025 to a Pandas DataFrame
def prep_params_table(path_to_csv: str, 
                      incomplete_string: str = placeholder) -> pd.DataFrame:
    params_table = pd.read_csv(path_to_csv, index_col = 0).transpose()
    params_table[incomplete_string] = np.nan
    return params_table

# Splits a sting into consecutive substrings of split_length starting from a chosen location in the string (start_frame_idx)
def split_string(string: str, 
                 split_length: int, 
                 start_frame_idx: int = 0, 
                 incomplete_string: str = placeholder) -> list:
    frame = [string[(start_frame_idx + i):(start_frame_idx + i + split_length)] for i in range(0, len(string), split_length)]
    frame = [ incomplete_string if len(word) < split_length else word for word in frame ]
    return frame

# Calculates indices used by profile generation loop (in calculate_profile()) that are re-used for all sequences
# based on the length of the sequence, sliding window length (in bases), and length of the nucleotide step (di, tri, tetra, etc.)
def prep_slide_indices(step_length: int, 
                       seq_length: int = seq_len, 
                       window_length: int = window_len) -> tuple:
    window_n_nucleosteps = window_length - (step_length - 1) # number of N-nucleotide steps in the chosen window width
    max_frame_idx = math.floor(seq_length / step_length) # greatest index value needed in the reading frame coordination list
    frame_idx = np.arange((max_frame_idx + 1) * step_length) // step_length # list of indices that controls which positions are selected in each reading frame
    which_frame = list(range(step_length)) * math.ceil(seq_length / step_length) # list of indices that controls alternation between the reading frames
    final_seq_length = seq_length - (window_length - 1) # how many window nucleotides/positions do we want to end up with in the final profiles?
    return (frame_idx, which_frame, seq_length, window_n_nucleosteps, final_seq_length)

# Moves through the sequence one position at a time, calculating the average of each physicochemical parameter for a sliding window of 'window_length' nucleotides
# by splitting into all possible tri- or tetra-nucleotide steps (centered around the current base). In practice, for computational efficiency this is done by 
# splitting the entire sequence into N reading frames of N-nucleotide steps (offset by one base) and alternating between (and incrementing through) them. 
# NOTE: For reliability, please only use with window_lengths equal to 9, 15, 21, 27, 33, 39, etc. (i.e. odd multiples of 3). May or may not work outside of those use cases.
# NOTE: This function expects A, T, G, C, or N nucleotide symbols that are all uppercase (remove or convert lowercases/masking first)
def calculate_profile(seq: str, 
                      step_length: int, 
                      loop_indices: tuple, 
                      parameters: pd.DataFrame, 
                      window_length: int = window_len, 
                      N_skipping: bool = skip_seqs_w_Ns, 
                      debug: bool = False) -> tuple:
    
    if not (step_length in [3, 4]):
        Exception(f'{script_name}: calculate_profiles() only works with step_length equal to 3 or 4!')
    
    # Prepare reading frames and corresponding parameter values (for each position in each frame)
    frames = [ split_string(seq, step_length, start_idx) for start_idx in range(step_length) ]
    
    try: # this try except block handles nucleotide steps with Ns in them by:
        n_nucleo_frames = np.array([np.array(parameters[frame]) for frame in frames])
    except:
        if (N_skipping): # a) skipping sequences that contain N's or...
            return (None, False)
        else: # b) or averaging all possibilities, i.e. mean of profile for N replaced with A, T, G, and C in the step of interest.
            for frame in frames: # NOTE: This code is not ideal, particularly for more than one N per step, e.g. for cases like NNGC, NNNA, etc.
                for step in frame:
                    if 'N' in step:
                        print(f"{script_name}: N's detected in {step}.") # Make user aware of how the N falls within the steps
                        sum_of_options = np.zeros(parameters.shape[0])
                        for possibility in ['A', 'T', 'G', 'C']:
                            sum_of_options = np.add(sum_of_options, np.array(parameters[step.replace('N', possibility)]))
                        parameters[step] = sum_of_options / 4
            n_nucleo_frames = np.array([np.array(parameters[frame]) for frame in frames])
    
    if debug: # Useful to check that the windows are properly positioned (see loop below)
        frames = np.array(frames)
    
    # Loop through this sequence one base at a time and calculate moving average for sliding window (all parameters)
    (frame_idx, which_frame, seq_length, window_n_nucleosteps, final_seq_length) = loop_indices
    profiles = np.zeros((parameters.shape[0], seq_length - (window_length - 1)))
    for i in range(final_seq_length):
        if debug: # Useful to check that the windows are properly positioned
            print(frames[which_frame[i:(i + window_n_nucleosteps)], frame_idx[i:(i + window_n_nucleosteps)]]) 
        full_window = n_nucleo_frames[ which_frame[i:(i + window_n_nucleosteps)], :, frame_idx[i:(i + window_n_nucleosteps)] ].transpose()
        profiles[:, i] = np.mean(full_window, axis=1)
    
    return (profiles, True)

# Create a list of lists that holds important parameters and data used in the tri- and tetra-nucleotide-based profile calculation algorithm
# NOTE: this list is static/constant across all profiles so it is initialized here once for repeated use in all iterations
frame_resources = [ [step_length, prep_slide_indices(step_length), prep_params_table(param_table_paths[step_length])] for step_length in [3,4] ]

# Calculates the profiles using tri- and tetra-nucleotide steps for this sequence and concatenates them into one array
def calculate_multiframelength_profile(seq_item: tuple, 
                                       frame_resources: list = frame_resources) -> tuple:
    
    profiles = tuple(calculate_profile(seq=seq_item[0], step_length=frame[0], loop_indices=frame[1], parameters=frame[2]) for frame in frame_resources)
    
    if not any([profile[1] for profile in profiles]): # Was this sequence skipped due to the presence of N's?
        return (seq_item[1], None, False, seq_item[2])
    else:
        merged_profiles = np.concatenate([profile[0] for profile in profiles], axis=0)
        return (seq_item[1], merged_profiles, True, seq_item[2])

# Samples all or some of the sequences in a target FASTA, calculates the energy profiles for them, and saves resulting Numpy arrays to disk in .NPY files
# The Python library multiprocessing is used to divide the work over multiple CPU cores/threads for faster run times
# The following resources were helpful in figuring out how to use multiprocessing.Pool() and imap() properly:
# https://stackoverflow.com/questions/14723458/distribute-many-independent-expensive-operations-over-multiple-cores-in-python
# https://superfastpython.com/multiprocessing-pool-imap/
# https://superfastpython.com/multiprocessing-pool-python/
# https://stackoverflow.com/questions/53306927/chunksize-irrelevant-for-multiprocessing-pool-map-in-python
def profiles_batch(seqs_path: str, 
                   profiles_path: str, 
                   seq_type: str, 
                   strand: str,
                   all_seqs: bool = all_seqs, 
                   sample_size: int = None, 
                   metadata_suffix: str = profiles_metadata_suffix,
                   report_iter: int = 50000, 
                   RNG = None, 
                   save_cores: int = 0) -> int:
    
    with open(seqs_path, 'r') as file:
        lines = file.readlines()
    
    if seq_type == 'control_introns':
        with open(seqs_path.replace('.fasta', ''), mode='r') as file:
            intron_types = file.readlines()
            intron_types = [intron_type.split(' ') for intron_type in intron_types]
            intron_types = {intron_type[0] : intron_type[1].strip('\n') for intron_type in intron_types}
    
    num_seqs = int(len(lines)/2) # Assumes a FASTA format with seqs and headers taking one line each!
    sample_seq_idxs = range(0, num_seqs) if all_seqs else RNG.choice(list(range(0, num_seqs)), size=sample_size, replace=False)
    seq_items = []
    expr = re.compile(r'>(NC_\d*\.\d*:)(\d*-\d*)\((\+|-)\)')
    
    for seq_idx in sample_seq_idxs:
        seq_header = lines[seq_idx*2]
        seq = lines[seq_idx*2 + 1].strip('\n').upper()
        id_parse = re.match(expr, seq_header)
        base_id = f'{id_parse.group(1)}{id_parse.group(2)}'
        seq_id = f'{base_id}{"-f" if id_parse.group(3) == "+" else "-r"}'
        seq_items.append((seq, seq_id, '' if seq_type != 'control_introns' else intron_types[base_id])) # intron_types dict may be slow compared to list?
    seq_items = list(set(seq_items)) # This step eliminates relatively small number of duplicates that come from earlier calling from the feature annotation in Script 2
    N_completed = 0
    
    with multiprocessing.Pool(processes = (threads - save_cores)) as pool:
        for result in pool.imap(func = calculate_multiframelength_profile, iterable = seq_items, chunksize = (threads - save_cores)):
            if (result[2]):
                # Append to the file of profile arrays
                with NpyAppendArray(f'{profiles_path}.npy', delete_if_exists=False) as npy_file:
                    npy_file.append(np.reshape(result[1], shape=(1, *result[1].shape)))
                
                # Append to the file linking array indices and to seq IDs and class labels
                with NpyAppendArray(f'{profiles_path}{metadata_suffix}.npy', delete_if_exists=False) as npy_file:
                    npy_file.append(np.array([[result[0], seq_type, strand, result[3]]], dtype='<U40'))
                
                N_completed += 1
                
                if ((N_completed % report_iter) == 0):
                    print(f"{script_name}: {N_completed} profiles completed, {round(N_completed/len(seq_items)*100, 1)}% of this batch.")
            else:
                print(f"{script_name}: sequence {result[0]} skipped due to N's.")
    return N_completed

# Removes the previous .npy files from output directory (asks user first, just to be safe)
def ask_and_clear(directory_path: str) -> bool:
    
    if not len(os.listdir(directory_path)) == 0:
        
        response = input(f'{script_name}: Delete all existing .npy files in "{directory_path}" to prepare for new data?\n"y" = Yes/Continue\n"n" = No/Cancel\n')
        
        if response == 'y':
            print(f'{script_name}: Removing all existing .npy files...')
            for file in os.listdir(directory_path):
                os.remove(f'{directory_path}{file}')
            print(f'{script_name}: Done. Continuing on to generate new profiles.')
            return True
        else:
            print(f'{script_name}: You either selected "No/Cancel" or an invalid option, cancelling...')
            return False
        
    else:
        return True

# Runs the profile batch loop for each type of input file (e.g. combination of sequence type, strand, etc.)
def run_batches(input_paths: dict, 
                strands: list, output_dir: str,
                sample_all_seqs: bool, 
                sample_sizes: dict = sample_sizes, 
                output_name: str = profiles_file_base_name, 
                seed: int = 112263) -> None:
    
    files_cleared = ask_and_clear(output_dir)
    rng = np.random.default_rng(seed)
    
    if files_cleared:
        
        start_time = time.time()
        start_date_time =  time.ctime(int(time.time()))
        print(f'{script_name}: Started profile generation at {start_date_time}')
        n_profiles = 0
        
        for seq_type in input_paths:
            for strand in strands:
                n_batch = profiles_batch(f'{input_paths[seq_type]}{strand}.fasta', 
                                         f'{output_dir}{output_name}', 
                                         all_seqs=sample_all_seqs, 
                                         sample_size=sample_sizes[f'{seq_type}_{strand}'], 
                                         RNG=rng, 
                                         seq_type=seq_type, 
                                         strand=strand)
                n_profiles += n_batch
                print(f'{script_name}: Finished {strand} {seq_type} boundaries. {round((time.time() - start_time)/60, 2)} minutes elapsed for {n_profiles} profiles.')
        print(f'{script_name}: Finished all boundaries! Took total of {round((time.time() - start_time)/60, 2)} minutes to calculate {n_profiles} profiles.')
    else:
        return None


### Use functions defined above to generate all profiles:
if __name__ == '__main__':
    
    run_batches(input_paths, strands, output_dir, all_seqs)
