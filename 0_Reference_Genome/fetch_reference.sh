#! /bin/bash

## Utility script provided in case the user does not already have a copy of human reference genome GRCh38.p14
## This script downloads the GRCh38.p14 genomic FASTA (.fna) and RefSeq annotation (.gff) files from the NCBI 
## FTP server, decompresses them, and indexes the FASTA file using samtools.
## Remember to check permissions and "chmod 700" or similar to this script before attempting to execute
## Remember to run in an environment with gzip, wget, and samtools installed (see provided environment.yml directory)

DATA_PATH=$1 # This script takes one argument, which is a path to a directory to which the files should be downloaded and prepared
FNA_path="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz"
GFF_path="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.gff.gz"

# If the user has not set the directory they want to download the files to, just use the directory this script lives in
if [ -z "${DATA_PATH}" ]; then
    DATA_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Prompt the user to continue and accept the space requirement
echo "Would you like to automatically download the GRCh38.p14 sequence and annotation files from NCBI and prepare them?"
read -p "This will take up ~3 GB of disk space post-decompression. (y/n) " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 0

# Fetch reference genome and annotation files
echo "Fetching FNA and GFF files from NCBI FTP server..."
wget -P "$DATA_PATH" "$FNA_path" && wget -P "$DATA_PATH" "$GFF_path"
last_status=$?

# If the files downloaded successfully, decompress the files and delete reaming .gz's
if [ $last_status -eq 0 ]; then
    echo "Files successfully downloaded! Decompressing them..."
    gunzip "$DATA_PATH"*.gz
    last_status=$?
else
    echo "One or more files could not be downloaded from the NCBI website. Quitting."
    exit 1
fi

# If the files unzipped, index the FASTA file
if [ $last_status -eq 0 ]; then
    echo "Indexing the FNA file using samtools..."
    samtools faidx "$DATA_PATH"*.fna
    last_status=$?
else
    echo "One or more files could not be unzipped, is gunzip installed and in your PATH? Quitting."
    exit 1
fi

# If the files unzipped, index the FASTA file
if [ $last_status -eq 0 ]; then
    echo "Success! You now have an indexed copy of GRCh38.p14 and the corresponding RefSeq annotation in $DATA_PATH"
else
    echo "FNA file could not be indexed, is samtools installed and in your PATH? Quitting."
    exit 1
fi