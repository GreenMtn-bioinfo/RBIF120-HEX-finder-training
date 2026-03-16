import numpy as np
import pandas as pd
import plotly.express as px
import preparation_1 as prep
import profile_generator_3 as profile_gen
import multiprocessing
import os

# Import project-wide variables from preparation_1.py
profiles_path = prep.structural_profiles
boundary_margin = prep.boundary_margin
parameter_tables = prep.param_table_paths
manifest_name = profile_gen.manifest_name

# Imports the physicochemical parameter tables to get the names of the parameters
def get_param_names(paths: dict = parameter_tables, prefixes: dict = {3 : 'tri ', 4 : 'tetra '}) -> list:
    param_names = []
    for key in paths:
        params_table = pd.read_csv(paths[key], index_col = 0)
        param_names = param_names + [ (prefixes[key] + name) for name in params_table.columns ]
    return param_names

# Z-normalizes a vector
def z_normalize(vector: np.ndarray) -> np.ndarray:
    return (vector - np.mean(vector)) / np.std(vector) 

# Facilitates multiprocessing
def mean_subproc(seq_item: tuple):
    seq_type = seq_item[0]
    manifest = seq_item[1]
    param_names = seq_item[2]
    strands = set(manifest['Strand'])
    profiles_sum = np.zeros((len(param_names), boundary_margin * 2 + 1), dtype = float)
    profiles_count = 0
    for strand in strands:
        profiles_set = np.load(f'{profiles_path}{seq_type}_{strand}.npz', allow_pickle=False)
        for profile in profiles_set.values():
            profiles_sum = np.add(profiles_sum, profile)
        profiles_count += len(profiles_set)
    # n_profiles[seq_type] = profiles_count
    # averages[seq_type] = profiles_sum / profiles_count
    return (seq_type, profiles_sum / profiles_count, profiles_count)

# Calculates the mean profile for all profiles of a given sequence type (i.e., exon start, exon end, or control/exon center)
def calculate_mean_profiles(seq_types: list = ['intron-exon', 'exon-intron', 'control']) -> tuple:
    manifest = pd.read_csv(f'{profiles_path}{manifest_name}')
    param_names = get_param_names()
    
    # Goes through all sequence types and calculate the mean profile (forward + reverse included)
    averages = {}
    n_profiles = {}
    # strands = set(manifest['Strand'])
    # for seq_type in seq_types:
    #     profiles_sum = np.zeros((len(param_names), boundary_margin * 2 + 1), dtype = float)
    #     profiles_count = 0
    #     for strand in strands:
    #         profiles_set = np.load(f'{profiles_path}{seq_type}_{strand}.npz', allow_pickle=False)
    #         for profile in profiles_set.values():
    #             profiles_sum = np.add(profiles_sum, profile)
    #         profiles_count += len(profiles_set)
    #     n_profiles[seq_type] = profiles_count
    #     averages[seq_type] = profiles_sum / profiles_count
    
    seq_items = [(seq_type, manifest, param_names) for seq_type in seq_types]
    with multiprocessing.Pool(processes = len(seq_types)) as pool:
        for result in pool.imap(func = mean_subproc, iterable = seq_items, chunksize = len(seq_types)):
            averages[result[0]] = result[1]
            n_profiles[result[0]] = result[2]
            print(f'Finished calculating {result[0]} averages.')
    
    # For each parameter, z-normalizes the mean profile across all 3 sequence types 
    # (i.e, concatenated, this ensures relatively small noise in the control mean is not unrealistically magnified relative to the exon start/end mean profiles)
    joined = np.concatenate([averages[seq_type] for seq_type in seq_types], axis=1)
    normalized = np.apply_along_axis(z_normalize, arr=joined, axis=1)
    min_y = np.min(normalized)
    max_y = np.max(normalized)
    
    # Prepare the relative position/nucleotide vector for visual clarity once plotted
    left_side = -np.array(range(1, boundary_margin + 1))[::-1]
    right_side = np.array(range(0, boundary_margin + 1))
    position_vector = np.concatenate((left_side, right_side))
    
    # Splits numpy array, converts to pandas data frames, adds position and seq_type columns, and re-concatenates for use with plotly express
    averages_normalized = {}
    seq_length = int(normalized.shape[1]/len(seq_types))
    for i, seq_type in enumerate(seq_types):
        averages_normalized[seq_type] = pd.DataFrame(normalized[:,(seq_length*(i)):(seq_length*(i + 1))]).transpose().set_axis(param_names, axis=1)
        averages_normalized[seq_type]['Position'] = position_vector
        averages_normalized[seq_type]['Type'] = seq_type
    averages_normalized = pd.concat([averages_normalized[seq_type] for seq_type in seq_types])
    
    return (averages_normalized, n_profiles, (min_y, max_y))


# Uses plotly express to create a faceted plot of the mean profile around exon beginnings, endings, and centers (for long exons i.e. controls)
def plot_means(mean_profiles: tuple, plot_size_px: tuple = (700, 750)):
    
    # Rename input objects for later use
    df = mean_profiles[0]
    n_profiles = mean_profiles[1]
    min_y = mean_profiles[2][0]
    max_y = mean_profiles[2][1]
    
    # Controls how a profile type is labeled on its respective plot
    plot_names_dict = { 'intron-exon' : 'Start',
                        'exon-intron' : 'End',
                        'control' : 'Center'}
    plot_names_dict = {key : plot_names_dict[key] + f' (N = {n_profiles[key]})' for key in plot_names_dict}
    
    # Create and format a multi-facet interactive plot of the mean profiles around all three sequence types using plotly express
    fig = px.line(df, x='Position', y=df.columns[:-2], facet_row="Type",
                  title =f'Normalized average parameter profiles around exon start, end, or center*', range_y=[min_y, max_y],
                  labels = { 'value' : 'Z-Normalized values',
                             'Position' : 'Relative position/nucleotide',
                             'variable' : 'Structural parameter'},
                  width=plot_size_px[0], height=plot_size_px[1])
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Type=", "")))
    fig.for_each_annotation(lambda a: a.update(text=plot_names_dict[a.text]))
    fig['layout']['yaxis']['title']['text']=''
    fig['layout']['yaxis3']['title']['text']=''
    fig.update_yaxes({'gridcolor': 'green', 'zerolinecolor': 'yellow'})
    fig.update_yaxes({'gridcolor': 'lightgrey', 'zerolinecolor': 'grey'})
    fig.update_xaxes({'gridcolor': 'lightgrey', 'zerolinecolor': 'grey'})
    # May be useful for tweaking/controlling plot subtitle positions
    # height_dict = { 'intron-exon' : 1.1,
    #                 'exon-intron' : 0.7,
    #                 'control' : 0.35}
    # fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Type=", ""), textangle = 0, yanchor='top', xanchor='center', x=0.5))
    # fig.for_each_annotation(lambda a: a.update(y=height_dict[a.text]))
    # print([annotation for annotation in fig.layout.annotations])
    fig.show()

if __name__ == "__main__":
    profile_means_tuple = calculate_mean_profiles()
    plot_means(mean_profiles=profile_means_tuple, plot_size_px=(700, 750))