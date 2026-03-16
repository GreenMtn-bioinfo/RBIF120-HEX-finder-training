import subprocess
import math
import sys
import os
import preparation_1 as prep # 1st script, required for important variables used across scripts
import re

#### 2nd Script
# This script creates files of genomic coordinates/regions for all sequences of interest and then uses 'samtools faidx' to get them.
# More specifically, it gets all exon boundaries (intron to exon, or vice verse) from the MANE Select exons that are large enough
# and far enough apart (as defined by 'min_boundary_distance'). It also gets sequences in the center of exons over 'exon_length_threshold'
# as controls for what a nonexistent signal (noise) looks like (this was an approach taken by Mishra et al., 2021)

# NOTE: This script is to be run from a Bash environment with samtools installed


## Variable assignments imported from 'preparation_1.py' script:

# Paths
annotation_path = prep.exon_annotation
selected_seqs = prep.selected_exon_seqs
reference_genome = prep.reference_genome

# Parameters
window_len = prep.window_len # odd number expected
boundary_margin = prep.boundary_margin # Bases of interest upstream and downstream of exon boundary
min_boundary_distance = prep.min_boundary_distance # What is the shortest intron/exon length allowed for consideration (boundaries too close = multiple signals in one "window")
exon_length_threshold = prep.exon_length_threshold # Only take midpoint sequences of exons longer than this as controls
intron_length_threshold = prep.intron_length_threshold
pre_pad = boundary_margin + math.floor(window_len/2) # How much extra front sequence is required to accommodate sliding window? (can change if algorithm in "profile_generator.py" is modified)
post_pad = pre_pad # How much extra end sequence is required to accommodate sliding window? (can change if algorithm in "profile_generator_3.py" is modified)
split_controls = prep.split_controls
training_length = prep.training_length
control_padding = prep.control_padding
max_exon_splits = prep.max_exon_splits
max_intron_splits = prep.max_intron_splits


## Local variables that are script-specific and/or hard-coded:

# Path-related

# INPUT
annotation_paths = {'forward' : f'{annotation_path}mane_select_forward_exons.gff',
                    'reverse' : f'{annotation_path}mane_select_reverse_exons.gff'}

# OUTPUT
exon_boundaries = {'intron-exon' : f'{selected_seqs}intron-exon_boundaries',
                   'exon-intron' : f'{selected_seqs}exon-intron_boundaries'}
introns_prefix = 'control_introns' # Used for the names of files generated pertaining to midpoints of exons > exon_length_threshold
exons_prefix = 'control_exons'

### 1) Functions for getting the exon boundary sequences for both strands:

# Gets the exon beginning or end boundaries from the previously compiled RefSeq annotation, filters by distance between exon boundaries
# https://www.biostars.org/p/175640/
def get_boundaries(annotation_path: str, 
                   forward_strand: bool, 
                   exon_start: bool, 
                   pre_pad: int = pre_pad, 
                   post_pad: int = post_pad) -> list:
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
        lines = [line.split('\t') for line in lines]
        regions = []
        # Next 4 lines are important! Interpretation of coordinates for exon boundaries from GFF3 file reverses for reverse strand!
        start_idx = 3 if forward_strand else 4
        end_idx = 4 if forward_strand else 3
        # Formulate the genomic coordinates for windows around exon boundaries of interest, taking strand/GFF3 conventions into account
        for i in range(0,len(lines)):
            exon_beginning = int(lines[i][start_idx])
            exon_end = int(lines[i][end_idx])
            exon_length = abs(exon_end - exon_beginning)
            if exon_length > min_boundary_distance and not (i == 0 or (i == len(lines) - 1)): # skips first exon of each chromosome out of convenience, acceptable in for this application as this represents a small portion of all of them
                previous_exon_end = int(lines[i-1][end_idx]) if forward_strand else int(lines[i+1][end_idx])
                next_exon_start = int(lines[i+1][start_idx]) if forward_strand else int(lines[i-1][start_idx])
                if exon_start and (abs(exon_beginning - previous_exon_end) > min_boundary_distance):
                    regions.append(f'{lines[i][0]}:{int(lines[i][start_idx])-pre_pad}-{int(lines[i][start_idx])+post_pad}\n')
                elif not exon_start and (abs(next_exon_start - exon_end) > min_boundary_distance):
                    regions.append(f'{lines[i][0]}:{int(lines[i][end_idx])-pre_pad}-{int(lines[i][end_idx])+post_pad}\n')
        return regions

# Writes the files that will be used by samtools faidx to grab sequences based on exon boundary
def write_boundary_files(annotation_paths: dict, 
                         exon_boundaries: str):
    for strand in annotation_paths:
        for boundary in exon_boundaries:
            boundaries = get_boundaries(annotation_path = annotation_paths[strand], forward_strand = (strand == 'forward'), exon_start = (boundary == 'intron-exon'))
            with open(exon_boundaries[boundary] + f'_{strand}', 'w') as file:
                file.writelines(boundaries)

# Checks whether the call to samtools succeeded
def check_exit(completed: subprocess.CompletedProcess):
    if completed.returncode != 0:
        print(f'{os.path.basename(__file__)}: Calling samtools failed, is samtools installed?')
        sys.exit(1)

# Calls samtools faidx to get the sequences around exon boundaries from the reference genome and write to FASTAs
def get_boundary_seqs(reference_genome: str, 
                      annotation_paths: dict, 
                      exon_boundaries: dict, 
                      max_line_len: int = boundary_margin*2 + window_len):
    for strand in annotation_paths:
        for boundary in exon_boundaries:
            strand_flag = '-i' if strand == "reverse" else ''
            command = f'samtools faidx -n {max_line_len} {reference_genome} -o {exon_boundaries[boundary] + "_" + strand + ".fasta"} -r {exon_boundaries[boundary] + "_" + strand} {strand_flag} --mark-strand sign'
            boundary_seqs = subprocess.run(command, shell=True)
            check_exit(boundary_seqs)


### 2) Functions for getting the midpoint sequences of controls (i.e. taken from exons or introns)

# Checks all exon regions in forward and reverse strand GFF3 files, prepared earlier for length >= threshold and returns center coordinates if true
def get_controls(annotation_paths: dict, 
                 length_threshold: int, 
                 max_control_splits, # can be dict or int
                 output_prefix: str, 
                 pre_pad: int = pre_pad, 
                 post_pad: int = post_pad, 
                 split_controls: bool = split_controls, 
                 introns: bool = False):
    gene_expr = re.compile(r'GeneID\:\d*')
    for strand in annotation_paths:
        regions = []
        with open(annotation_paths[strand], 'r') as file:
            lines = file.readlines()
            lines = [line.split('\t') for line in lines]
        for n, line in enumerate(lines):
            if introns and ((n == 0) or (n == len(lines) - 1)):
                continue
            elif introns:
                start = int(lines[n-1][4])
                end = int(line[3])
                gene_id_prev = re.search(gene_expr, lines[n-1][8]).group(0) ##
                gene_id_current = re.search(gene_expr, line[8]).group(0) ##
                if gene_id_prev == gene_id_current: ##
                    intron_type = 'intron' ##
                else: ##
                    intron_type = 'intergenic' ##
            else:
                start = int(line[3])
                end = int(line[4])
            length = end - start
            if length >= length_threshold:
                if split_controls:
                    n_splits = min(int(length // (training_length + window_len + control_padding)), (max_control_splits if not introns else max_control_splits[intron_type]))
                    if n_splits >= 2:
                        jump_dist = int(length // n_splits)
                        for i in range(1, n_splits):
                            if strand == 'forward':
                                regions.append(f'{line[0]}:{start + (jump_dist * i) - pre_pad}-{start + (jump_dist * i) + pre_pad}{"" if not introns else " " + intron_type}\n') #
                            else:
                                regions.append(f'{line[0]}:{end - (jump_dist * i) - pre_pad}-{end - (jump_dist * i) + pre_pad}{"" if not introns else " " + intron_type}\n') # 
                else:
                    exon_mid = start + math.floor(length/2)
                    regions.append(f'{line[0]}:{exon_mid - pre_pad}-{exon_mid + post_pad}{"" if not introns else " " + intron_type}\n') # 
        with open(f'{output_prefix}_{strand}', 'w') as file:
            file.writelines(regions)

# Calls samtools faidx to get the sequences based on a file full of regions of interest
def get_seqs(reference_genome: str, output_path: str, regions_path: str, reverse_strand: bool, max_line_len: int = boundary_margin*2 + window_len):
    strand_flag = '-i' if reverse_strand else ''
    command = f'samtools faidx -n {max_line_len} {reference_genome} -o {output_path} -r {regions_path} {strand_flag} --mark-strand sign'
    boundary_seqs = subprocess.run(command, shell=True)
    check_exit(boundary_seqs)

# Gets all sequences 
def get_control_seqs(annotation_paths: dict, output_prefix: str):
    for strand in annotation_paths:
        regions_file = f'{selected_seqs}{output_prefix}_{strand}'
        with open(regions_file, mode='r') as file:
            lines = file.readlines()
            lines = [line.split(' ') for line in lines]
        if len(lines[0]) != 1:
            regions_file = f'{regions_file}_amended'
            with open(regions_file, mode = 'w') as new_file:
                new_file.writelines([line[0] + '\n' for line in lines])
        get_seqs(reference_genome, regions_file.replace('_amended','') + '.fasta', regions_file, reverse_strand = (True if strand == "reverse" else False))


### Use functions defined above to retrieve and organize the sequences:
if __name__ == '__main__': # TODO: This script retrieves several duplicate seqs in each category, I think this is simply due to a few duplicate features retrieved from the reference annotation. These are filtered out in the next script anyways
    write_boundary_files(annotation_paths, exon_boundaries)
    get_boundary_seqs(reference_genome, annotation_paths, exon_boundaries)
    get_controls(annotation_paths, max_control_splits=max_exon_splits, introns = False, length_threshold=exon_length_threshold, output_prefix = f'{selected_seqs}{exons_prefix}')
    get_controls(annotation_paths, max_control_splits=max_intron_splits, introns = True, length_threshold=intron_length_threshold, output_prefix = f'{selected_seqs}{introns_prefix}')
    get_control_seqs(annotation_paths, exons_prefix)
    get_control_seqs(annotation_paths, introns_prefix)
