import multiprocessing
import numpy as np
from preparation_1 import structural_profiles
from split_data_4 import load_json, all_profiles_path, split_json_path, IDs_file_path
import time
import math
import os


# INPUTS

# # Load required data for accessing the training set
all_profiles = np.load(all_profiles_path, mmap_mode='r') # all parameter profiles
IDs = np.load(IDs_file_path, mmap_mode='r') # profile metadata

# OUTPUTS
means_path = f'{structural_profiles}z_norm_training_means.npy'
sdevs_path = f'{structural_profiles}z_norm_training_sdevs.npy'
mins_path = f'{structural_profiles}min_training_post_z.npy'
maxes_path = f'{structural_profiles}max_training_post_z.npy'

# Script parameters
cores = os.cpu_count() # How many cores (virtual or hardware) are available?
script_name = os.path.basename(__file__) # Used globally (not passed) in print statements to provide the name of this file (useful in of pipeline)


### Function definitions:

# Allows parallelization of the calculation of the mean (first sum all profile arrays in parallel, flatting the samples dimension)
def load_and_sum(args: tuple, 
                 arr_pointer = all_profiles) -> np.ndarray:
    indices = args[0]
    profiles_sum = np.sum(arr_pointer[indices,], axis=0)
    return np.reshape(profiles_sum, shape=profiles_sum.shape[1:])

# Takes a 1D array and copies it repeatedly along a new dimension
def extrude_1D(vector: np.ndarray, 
               n_columns: int) -> np.ndarray:
    
    if len(vector.shape) != 1:
        print(f'{script_name}: extrude() expects a 1D array!')
        return None
    column = vector.reshape((vector.shape[0], 1))
    return np.concatenate([column] * n_columns, axis=1)

# Calculates the mean from the aggregated sum of squared differences
def finalize_mean(profiles_sum: np.ndarray, 
                  profiles_count: int) -> np.ndarray:
    
    average = np.apply_along_axis(np.sum, axis=1, arr=profiles_sum) / (profiles_count * profiles_sum.shape[1])
    return extrude_1D(average, profiles_sum.shape[1])

# Calculates a metric or statistic for all the profiles in a selected set of IDs (e.g. training, validation, etc.)
def calculate_set_stat(multiproc_function, 
                       post_proc_function, 
                       metric_name: str,
                       indices: np.ndarray, 
                       additional_args: dict = {},
                       dataset = all_profiles, 
                       load_chunk: int = 25000, 
                       verbose: bool = False, 
                       notify: int = 20):
    
    # Create the slices that represent the chunks of the total array that will be processed by each worker
    N_profiles = len(indices)
    n_slices = math.ceil(N_profiles / load_chunk)
    remainder = N_profiles % load_chunk
    slices = [ slice(i * load_chunk, (i + 1) * load_chunk) if i < (n_slices - 1) else slice(i * load_chunk, i * load_chunk + remainder) for i in range(n_slices) ]
    args = [ (indices[slc].tolist(), additional_args) for slc in slices]
    
    # Create a multiprocessing pool and distribute/run the tasks, reporting on progress along the way
    N_batches = len(args)
    profiles_sum = np.zeros(shape=dataset.shape[1:])
    batches_finished = 0
    if verbose:
        print(f'{script_name}: starting the calculation of 28 parameter {metric_name}s across the selected set.')
    start = time.time()
    with multiprocessing.Pool(processes = (cores - 2)) as pool:
        for result in pool.imap(func = multiproc_function, iterable = args, chunksize=1):
            profiles_sum += result
            if verbose:
                batches_finished += 1
                if ((batches_finished % notify) == 0):
                    rate = (time.time() - start) / batches_finished
                    left = N_batches - batches_finished
                    print(f'{script_name}: completed aggregating {round(batches_finished/N_batches*100, 1)}% of profiles. ETC for this metric is {round((left * rate)/60, 2)} minutes.')
        pool.terminate()
    return post_proc_function(profiles_sum, N_profiles)

# Function allows parallelization of the sum of squared differences (for the standard deviation) across one dimension of the array
def load_and_sum_sq(args: tuple, arr_pointer = all_profiles) -> np.ndarray:
    indices = args[0]
    means = args[1]['means']
    profiles_diff_sq = np.square(arr_pointer[indices,] - means)
    profiles_sum_diff_sq = np.sum(profiles_diff_sq, axis=0)
    return np.reshape(profiles_sum_diff_sq, shape=profiles_sum_diff_sq.shape[1:])

# Calculates the standard deviation from the aggregated sum of squared differences
def finalize_sdev(profiles_sum_sq_diff: np.ndarray, profiles_count: int) -> np.ndarray:
    sum_of_sq = np.apply_along_axis(np.sum, axis=1, arr=profiles_sum_sq_diff) / ((profiles_count * profiles_sum_sq_diff.shape[1]) - 1)
    return extrude_1D(np.sqrt(sum_of_sq), profiles_sum_sq_diff.shape[1])

# Calculate the z-normalized values (each row/parameter normalized separately) given a compatible array of means and sdevs
def z_normalize(array: np.ndarray,
                means: np.ndarray,
                sdevs: np.ndarray) -> np.ndarray:
    
    normalized = (array - means) / sdevs
    return normalized.reshape((normalized.shape[0], normalized.shape[2], normalized.shape[3]))

# Calculate the min-max values (each row/parameter normalized separately) given a compatible array of mins and maxes
def min_max_norm(array: np.ndarray,
                 mins: np.ndarray,
                 maxes: np.ndarray,
                 bounds: tuple = (-1, 1)) -> np.ndarray:
    
    return (bounds[1] - bounds[0]) * (array - mins) / (maxes - mins) + bounds[0]

# Allows parallelization of the calculation of the min and max across an array larger than available RAM
def load_and_min_max(args: tuple, 
                     arr_pointer = all_profiles) -> np.ndarray:
    
    indices = args[0]
    normalize_first = args[1]['normalize_first']
    means_array = args[1]['means_array']
    sdevs_array = args[1]['sdevs_array']
    
    if normalize_first:
        array = z_normalize(arr_pointer[indices, :, :], means_array, sdevs_array)
    else:
        array = arr_pointer[indices, :, :]
    batch_min = np.min(array, axis=(0,2))
    batch_max = np.max(array, axis=(0,2))
    return (batch_min, batch_max)

# Calls load_and_min_max() to get the global min and max over the large array piecewise/parallelized
def calculate_min_max(multiproc_function, 
                      indices: np.ndarray, 
                      additional_args: dict = {},
                      dataset = all_profiles, 
                      load_chunk: int = 25000, 
                      verbose: bool = False, 
                      notify: int = 20):
    
    # Create the slices that represent the chunks of the total array that will be processed by each worker
    N_profiles = len(indices)
    n_slices = math.ceil(N_profiles / load_chunk)
    remainder = N_profiles % load_chunk
    slices = [ slice(i * load_chunk, (i + 1) * load_chunk) if i < (n_slices - 1) else slice(i * load_chunk, i * load_chunk + remainder) for i in range(n_slices) ]
    args = [ (indices[slc].tolist(), additional_args) for slc in slices]
    
    # Create a multiprocessing pool and distribute/run the tasks, reporting on progress along the way
    N_batches = len(args)
    batches_finished = 0
    if verbose:
        print(f'{script_name}: starting the calculation of the min and max of 28 parameter across the selected set.')
    start = time.time()
    first = True
    with multiprocessing.Pool(processes = (cores - 2)) as pool:
        for result in pool.imap(func = multiproc_function, iterable = args, chunksize=1):
            if first:
                global_min = result[0]
                global_max = result[1]
                first = False
            else:
                global_min = np.array([global_min, result[0]]).min(axis=0)
                global_max = np.array([global_max, result[1]]).max(axis=0)
                if verbose:
                    batches_finished += 1
                    if ((batches_finished % notify) == 0):
                        rate = (time.time() - start) / batches_finished
                        left = N_batches - batches_finished
                        print(f'{script_name}: completed aggregating {round(batches_finished/N_batches*100, 1)}% of profiles. ETC for this metric is {round((left * rate)/60, 2)} minutes.')
        pool.terminate()
    return extrude_1D(global_min, dataset.shape[-1]), extrude_1D(global_max, dataset.shape[-1])

def get_norm_params(indices,
                    save: bool = False,
                    sdevs_path=sdevs_path,
                    means_path=means_path,
                    mins_path=mins_path,
                    maxes_path=maxes_path):
    
    if indices.shape[0] < indices.shape[1]:
        indices = indices.transpose()
    
    # Calculate and save the mean of each parameter across all training samples
    training_means = calculate_set_stat(multiproc_function=load_and_sum, 
                                        post_proc_function=finalize_mean,
                                        indices=indices,
                                        metric_name='mean',
                                        verbose=True)
    
    if save:
        np.save(means_path, training_means, allow_pickle=False)
    
    # Calculate and save the standard deviation of each parameter across all training samples
    training_sdevs = calculate_set_stat(multiproc_function=load_and_sum_sq, 
                                        post_proc_function=finalize_sdev, 
                                        indices=indices,
                                        metric_name='standard deviation',
                                        additional_args={'means' : training_means},
                                        verbose=True)
    if save:
        np.save(sdevs_path, training_sdevs, allow_pickle=False)
    
    # Calculate and save the minimum and maximum of each parameter across all training samples (post z-norm)
    training_min_max_post_z = calculate_min_max(multiproc_function=load_and_min_max, 
                                                indices=indices,
                                                additional_args={'normalize_first' : True, 
                                                                 'means_array' : np.array([training_means]), 
                                                                 'sdevs_array' : np.array([training_sdevs])}, 
                                                verbose=True)
    if save:
        np.save(mins_path, training_min_max_post_z[0], allow_pickle=False)
        np.save(maxes_path, training_min_max_post_z[1], allow_pickle=False)
    else:
        return {'means' : training_means,
                'sdevs' : training_sdevs,
                'mins' : training_min_max_post_z[0],
                'maxes' : training_min_max_post_z[1]}



### Use functions defined above to calculate normalization parameters:
if __name__ == '__main__':
    
    partitioned = load_json(split_json_path) # train/test split
    train_set_indices = np.argwhere(np.isin(IDs[:,0], partitioned['train'], assume_unique=True))
    get_norm_params(train_set_indices, save=True)
