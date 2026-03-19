#!/bin/bash

# Fetches supplementary data from Zenodo for this research archive on GitHub (https://github.com/GreenMtn-bioinfo/RBIF120-HEX-finder-training)
# This script points to the the latest Zenodo record ID (19114650), which will need to be updated if the record is updated
# The main record for all versions is 19074687: https://zenodo.org/records/19074687 (DOI: 10.5281/zenodo.19074687)

set -e

# Define Zenodo record ID and file names
ZENODO_RECORD_ID="19114650"
FILES=(
    "1_Exon_Annotation.zip"
    "2_Selected_Coords_Seqs.zip"
    "3_Physicochemical_Profiles.zip"
    "4_Sliding_Window.zip"
    "ChemEXIN_modified.zip"
)

# Offer the user a chance to avoid download/extraction to the wrong directories
read -p "Please confirm: Are you are running this script from the ROOT directory of the repository? (y/n) " name
if [[ "$name" == "y" || "$name" == "yes" ]]; then
  echo "Starting data fetch from Zenodo record: $ZENODO_RECORD_ID..."
  echo "-------------------------------------------------"
else
  echo "Please rerun this script from the root directory of the repository."
  exit 1
fi

# Checks if the user has wget or curl installed and uses whichever is available (if any)
download_file() {
    local URL="$1"
    local OUTPUT_NAME="$2"

    if command -v curl &> /dev/null; then
        curl -L -o "$OUTPUT_NAME" "$URL"
    elif command -v wget &> /dev/null; then
        wget -q --show-progress -O "$OUTPUT_NAME" "$URL"
    else
        echo "Error: Neither 'curl' nor 'wget' was found on your system."
        echo "Please install one of them to download the datasets."
        exit 1
    fi
}

# Download and extract each file
for FILE in "${FILES[@]}"; do
    echo "Downloading $FILE..."
    DOWNLOAD_URL="https://zenodo.org/records/${ZENODO_RECORD_ID}/files/${FILE}?download=1"
    download_file "$DOWNLOAD_URL" "$FILE"
    
    echo "Extracting $FILE..."
    unzip -q -o "$FILE"
    
    echo "Cleaning up $FILE..."
    rm -f "$FILE"
    
    echo "$FILE successfully processed."
    echo "-------------------------------------------------"
done

echo "All data successfully fetched and extracted into their respective directories in this repository."