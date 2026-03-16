import subprocess
import os
import sys

#### 1st Script
# This script establishes and prepares everything for later steps, including:
# 1) Directory structure for the storage of input and output files
# 2) Important global variables/parameters used throughout the other scripts
# 3) GFF files with only the relevant subset of features from GRCh38 annotation included

# NOTE: This script expects Bash command environment with standard tools like grep and sort

# NOTE: For later scripts to work, the CSV files specified under the dictionary 'param_table_paths' must be present locally. 
# These CSVs come from the contents of supplemental files of molecular dynamics parameter estimates provided by Sharma et al., (2025).


## Variable assignments:

# Directory paths
exon_annotation = './1_Exon_Annotation/' # Where the files containing the extracted RefSeq exon annotation are stored
selected_exon_seqs = './2_Selected_Coords_Seqs/' # Intermediate files containing relevant genome coordinates and sequences of interest
structural_profiles = './3_Physicochemical_Profiles/' # Where the estimated B-DNA physicochemical profiles are stored once generated

# File paths
# energy_params = './EnergiesTable.csv' # Path to the CSV of all 31 parameters for all possible dinucleotide steps, made from supplemental Table S1.2 from Mishra et al., 2021
param_table_paths = { 3 : './trinucleo_sharma_et_al_2025_params.csv', # Path to CSV of 22 tri-nuncleotide-based physicochemical parameters calculated by Sharma et al., 2025
                      4 : './tetranucleo_sharma_et_al_2025_params.csv'} # Path to CSV of 6 tetra-nuncleotide-based physicochemical parameters calculated by Sharma et al., 2025
# It is assumed these files have been downloaded from NCBI and samtools faidx index has already been run to make the FAI:
reference_genome = './0_Reference_Genome/GCF_000001405.40_GRCh38.p14_genomic.fna' # location of reference genome FASTA (GRCh38)
reference_genome_annotation = './0_Reference_Genome/GCF_000001405.40_GRCh38.p14_genomic.gff' # location of a GFF annotation file for GRCh38

# Parameters 

# For exon boundary or exon midpoint sequence collection:
window_len = 27 # Length of the sliding window of dinucleotide steps used for the average energy profile calculations (odd number divisible by 3 expected)
boundary_padding = 1 # 5
training_length = 76 # True length is this + 1
boundary_margin = training_length // 2 # TODO: currently can only analyze new seqs add-hoc by changing this Bases of interest upstream and downstream of exon boundary (or chosen position)
min_boundary_distance = training_length + (window_len // 2) + boundary_padding # What is the shortest intron/exon length allowed between boundary sites for their consideration (boundaries too close = multiple signals in one "window")
control_padding = 5
exon_length_threshold = window_len*2 + control_padding*2 + training_length * 2 # boundary_margin*6 # Only take midpoint sequences of exons longer than this as controls
intron_length_threshold = exon_length_threshold

# For energy profiles calculation:
all_seqs = True # Whether to calculate profiles for ALL sequences, or sample a random subset using sample_sizes (all_seqs = True takes ~4 hours using 12 cores of a 2.2GHz processor w/ 32 GB RAM)
sample_sizes = {'control_exons_forward' : 27591,
                'control_exons_reverse' : 26469,
                'control_introns_forward' : 261214,
                'control_introns_reverse' : 254480,
                'exon-intron_forward' : 10000,
                'exon-intron_reverse' : 10000,
                'intron-exon_forward' : 10000,
                'intron-exon_reverse' : 10000}
split_controls = True # Whether or not to get two mid-exon sequences per center site (to the left and right of the center coordinate)
max_exon_splits = 60
max_intron_splits = {'intron' : 20, 'intergenic' : int(60 * 3)}

# This block is only run if the script is executed directly (not when imported as a module by other scripts)
if __name__ == "__main__":
    
    ### 1) Make directories that will be used later
    directories = [exon_annotation, selected_exon_seqs, structural_profiles]
    dir_check = [os.path.isdir(dir) for dir in directories] 
    if any(dir_check):
        print(f'{os.path.basename(__file__)}: One or more of the directories: ' + ', '.join(directories) + ' already exists. Please move or delete them and try again.')
        sys.exit(1)
    else:
        for dir in directories:
            os.mkdir(dir)
    
    ### 2) Run Bash commands to pull relevant RefSeq MANE Select features for forward and reverse strands out of the GRCh38 GFF into separate files
    
    ## First extract MANE select exon annotation for GRCh38
    command = f"cat {reference_genome_annotation} | grep -P '\tBestRefSeq\t' | grep -P 'tag=MANE\\sSelect' | grep -P '\texon\t' | grep '^NC_' > {exon_annotation}mane_select_exons.gff"
    boundary_annotation = subprocess.run(command, shell=True)
    
    ## Group by strand
    # https://www.biostars.org/p/297652/ was helpful for using sort properly here
    for strand in ['forward', 'reverse']:
        symbol = '+' if strand == 'forward' else '-'
        command = f"cat '{exon_annotation}mane_select_exons.gff' | grep -P '\t\\{symbol}\t' | sort -k1,1V -k4,4n -k5,5rn > {exon_annotation}mane_select_{strand}_exons.gff"
        boundary_annotation = subprocess.run(command, shell=True)
