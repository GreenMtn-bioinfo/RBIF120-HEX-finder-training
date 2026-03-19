#!/bin/bash

# Fetches supplementary data from Zenodo for this research archive on GitHub (https://github.com/GreenMtn-bioinfo/RBIF120-HEX-finder-training)
# This script points to the the "main" Zenodo record ID (19074687), which represents all versions
# and always resolves to the latest version: https://zenodo.org/records/19074687 (DOI: 10.5281/zenodo.19074687)

set -e

# Define Zenodo record ID and file names
ZENODO_RECORD_ID="19074687"
FILES=(
    "1_Exon_Annotation.zip"
    "2_Selected_Coords_Seqs.zip"
    "3_Physicochemical_Profiles.zip"
    "4_Sliding_Window.zip"
    "ChemEXIN_modified.zip"
)

# Offer the user a change to avoid download/extraction to the wrong directories
read -p "Please confirm: Are you are running this script from the ROOT directory of the repository? (y/n) " name
if [[ "$name" == "y" || "$name" == "yes" ]]; then
  echo "Starting data fetch from Zenodo record: $ZENODO_RECORD_ID..."
  echo "-------------------------------------------------"
else
  echo "Please rerun this script from the root directory of the repository."
  exit 1
fi

# Download and extract each file
for FILE in "${FILES[@]}"; do
    echo "Downloading $FILE..."
    
    curl -L -o "$FILE" "https://zenodo.org/records/${ZENODO_RECORD_ID}/files/${FILE}?download=1"
    
    echo "Extracting $FILE..."
    unzip -q -o "$FILE" -d $(echo "$FILE" | sed 's/.zip//')
    
    echo "Cleaning up $FILE..."
    rm -f "$FILE"
    
    echo "$FILE successfully processed."
    echo "-------------------------------------------------"
done

echo "All data successfully fetched and extracted into their respective directories in this repository."