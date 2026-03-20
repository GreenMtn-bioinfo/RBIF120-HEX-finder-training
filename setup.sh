#!/bin/bash

### This script automates the research repository setup for the user, including environment building, 
### environment configuration, and optional downloading of the externally hosted data files
### It should be run from within the root directory of the repository!

# Check what the user provided as the GPU qualifier string (first argument) and set the YAML file to use accordingly
GPU_TYPE=$(echo "$1" | tr '[:lower:]' '[:upper:]') # convert to uppercase
if [ "$GPU_TYPE" != "AMD" ]; then
    GPU_TYPE="NVIDIA"
fi

if [ "$GPU_TYPE" == "AMD" ]; then
    ENV_FILE_PATH="environment-amd.yml"
else
    ENV_FILE_PATH="environment.yml"
fi

# Build the conda environment from the chosen YAML
ENV_NAME=$(head -n 1 "$ENV_FILE_PATH" | cut -d ' ' -f 2)
echo "Building the $ENV_NAME conda environment (for $GPU_TYPE hardware)..."
conda env create -f "$ENV_FILE_PATH"

# Make sure the GPU libraries are properly accessible and limit GPU visibility to the first one
echo "Patching $GPU_TYPE GPU library paths and variables..."
eval "$(conda shell.bash hook)" # Initialize Conda for this subshell
if [ "$GPU_TYPE" == "AMD" ]; then
    conda run -n "$ENV_NAME" conda env config vars set LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH" HIP_VISIBLE_DEVICES="0" ROCR_VISIBLE_DEVICES="0"
else
    conda run -n "$ENV_NAME" conda env config vars set LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" CUDA_VISIBLE_DEVICES="0"
fi

# Register the Jupyter kernel
echo "Registering the newly created environment's Jupyter kernel..."
conda run -n "$ENV_NAME" python -m ipykernel install --user --name="$ENV_NAME" --display-name="Python ($ENV_NAME)"

# Final success message
echo "Setup complete for $GPU_TYPE-compatible environment! Run 'conda activate $ENV_NAME' to get started."

# Give the user a chance to automatically download the data files if they want
echo "Would you like to automatically download externally hosted data for this research compendium?"
read -n "You will be shown their decompressed sizes and given choices as to which files are downloaded and prepared. (y/n) " confirm
if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
    conda run -n "$ENV_NAME" "./fetch_all_data.sh"
else
    echo "You elected not to download the externally hosted data."
    echo "Remember to run 'conda activate $ENV_NAME' before executing any Python code in this repository."
fi