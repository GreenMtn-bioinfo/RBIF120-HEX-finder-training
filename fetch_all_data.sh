#!/bin/bash

### This script facilitates retrieving all data from remote sources to recreate the original research repository:
### 1) It will retrieve and prepare a copy of the GRCh38.p14 and corresponding reference annotation from NCBI
### 2) It will retrieve, unpack, and place the data files hosted on Zenodo
### The user will be prompted for confirmation for at each step, so they can pick and choose what is retrieved.

# Expected relative location and name of the reference files
REFERENCE_DIR="./0_Reference_Genome"
BASE_NAME="GCF_000001405.40_GRCh38.p14_genomic"
REFERENCE_FETCH="$REFERENCE_DIR/fetch_reference.sh"

# Path to the file that holds the paths of the files on Zenodo
ZENODO_FILES=".zenodo"
ZENODO_FETCH="fetch_zenodo_data.sh"

# Check for the presence of all reference files
if [ -f "$REFERENCE_DIR/$BASE_NAME.fna" ] && [ -f "$REFERENCE_DIR/$BASE_NAME.fna.fai" ] && [ -f "$REFERENCE_DIR/$BASE_NAME.gff" ]; then
  echo "All reference files for GRCh38.p14 already exist. Continuing on."
else
  echo "One or more files for GRCh38.p14 do not exist. Initiating reference retrieval."
  bash "$REFERENCE_FETCH"
  if [ $? -ne 0 ]; then
    echo "There appears to have been an issue while fetching and preparing the reference files!"
    echo "Please make sure the provided conda environment is active before running this script."
  else
    echo "Reference files successfully prepared, or user elected not to fetch them. Continuing on."
  fi
fi

# Read file paths into an array and set up boolean expression to check for each
mapfile -t files < "$ZENODO_FILES"
condition="true"
for file in "${files[@]}"; do
  condition="$condition && [ -f \"$file\" ]"
done

# Evaluate the all the conditions and initiate retrieval if any files are missing
if eval "$condition"; then
  echo "All files on Zenodo are already present in this repository. Finished!"
else
  echo "One or more files hosted on Zenodo are not present in this repository. Initiating retrieval."
  bash "$ZENODO_FETCH"
  if [ $? -ne 0 ]; then
    echo "There appears to have been an issue while fetching/unpacking the files from Zenodo!"
    echo "Please make sure you have an internet connection, as well as wget (or curl) and unzip installed."
  else
    echo "Zenodo files were successfully retrieved, or user elected not to fetch them. Finished!"
  fi
fi