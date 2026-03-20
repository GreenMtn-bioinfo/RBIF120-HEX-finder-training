#!/bin/bash

# Fetches supplementary data from Zenodo for this research archive on GitHub (https://github.com/GreenMtn-bioinfo/RBIF120-HEX-finder-training)
# This script points to the the latest Zenodo record ID (19114650), which will need to be updated if the record is updated
# The main record for all versions is 19074687: https://zenodo.org/records/19074687 (DOI: 10.5281/zenodo.19074687)

set -e

# Define Zenodo record ID and file names with sizes
ZENODO_RECORD_ID="19114650"
declare -A FILES_SIZES=(
  ["1_Exon_Annotation.zip"]="123 MB"
  ["2_Selected_Coords_Seqs.zip"]="1.1 GB"
  ["3_Physicochemical_Profiles.zip"]="4.2 GB"
  ["4_Sliding_Window.zip"]="5.6 GB"
  ["ChemEXIN_modified.zip"]="68 MB"
)
KEY_ORDER=("1_Exon_Annotation.zip" "2_Selected_Coords_Seqs.zip" "3_Physicochemical_Profiles.zip" "4_Sliding_Window.zip" "ChemEXIN_modified.zip")

# Offer the user a chance to avoid download/extraction to the wrong directories (or simply skip this step for their own reasons)
echo "This script will fetch data files from Zenodo, which will occupy a total of ~11 GB once decompressed."
echo "You will be given the chance to pick and choose specific subsets of the data on Zenodo."
echo "This script needs to be run within the ROOT directory of the repository or files will be misplaced."
read -p "Please confirm that you are running it properly and would like to proceed. (y/n) " confirm
if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
  echo "Starting data fetch from Zenodo record: $ZENODO_RECORD_ID..."
  echo "-------------------------------------------------"
else
  exit 0
fi

# Display the list of files with a numbering key system and their sizes
echo "The following files are available for download:"
echo "(approximate decompressed size reported for of each)"
i=1
for FILE in "${KEY_ORDER[@]}"; do
    echo "$i) $FILE (${FILES_SIZES[$FILE]})"
    ((i++))
done

# Prompt the user to download all files or specific files
read -p "Would you like to download all files (a) or specific files (s)? (a/s) " choice
if [[ $choice == [aA] ]]; then
    echo "You have chosen to download all files."
    FILES=("${KEY_ORDER[@]}")
elif [[ $choice == [sS] ]]; then
    echo "Please enter the numbers of the files you would like to download, separated by spaces:"
    read -a selected_files
    temp_files=()
    for num in "${selected_files[@]}"; do
        if (( num > 0 && num <= ${#KEY_ORDER[@]} )); then
            temp_files+=("${KEY_ORDER[num-1]}")
        else
            echo "Invalid file number: $num"
            exit 1
        fi
    done
    FILES=("${temp_files[@]}")
else
    echo "Invalid choice. Exiting."
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

# Download and extract each file that was selected by the user
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
echo "All selected data successfully fetched and extracted into their respective directories in this repository."
