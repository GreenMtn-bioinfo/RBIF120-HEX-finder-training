# About this repository

This repository is a static research compendium associated with the preprint "Evaluation of a structure-based method for ab initio gene detection using deep learning" by Hummel and Estrada (DOI: [10.64898/2025.12.19.694709](https://doi.org/10.64898/2025.12.19.694709 )). It is currently **unsupported**. While the code is fully functional for reproducing the reported results, no further features or updates are planned.

That [preprint](https://doi.org/10.64898/2025.12.19.694709), and this supporting repository, encapsulate the final version of the work conducted during an 8-week independent research project in the Brandeis University Bioinformatics MS program (RBIF 120: Research Topics in Computational Biology). The models trained and evaluated during this project were later put into a command-line tool, called [Helix-EXon-finder](https://github.com/GreenMtn-bioinfo/Helix-EXon-finder). Check out that repository for a more organized and accessible way to explore the pipeline's current capabilities and core code base. Helix-EXon-finder ***is*** under active development and I hope to improve the prediction pipeline.

+ Please contact Jonathan Hummel (jhgmbioinfo@gmail.com) with any questions about the preprint or this repository.

# External Code and Models

Part of the work described in the [preprint](https://doi.org/10.64898/2025.12.19.694709) was a benchmarking analysis comparing the performance of the models trained during this project to a standard truth set (RefSeq MANE Select exons), as well as the predictions from [ChemEXIN](https://github.com/rnsharma478/ChemEXIN). ChemEXIN is the tool made by the authors whose research inspired this project, and this project represents a different implementation of the same underlying idea. For more details on their research, please see the publications by Mishra et al. in 2021 (DOI: [10.1093/nar/gkab098](https://doi.org/10.1093/nar/gkab098)), Sharma et al. 2023 (DOI: [10.1039/d2cp04820e](https://doi.org/10.1039/d2cp04820e)), and Sharma et al. in 2025 (DOI: [10.1039/d4mo00241e](https://doi.org/10.1039/d4mo00241e)). There are two components of this repository that were originally authored by the members of that research group:

1. [ChemEXIN_modified/](/ChemEXIN_modified/) contains a modified clone of the [ChemEXIN repository](https://github.com/rnsharma478/ChemEXIN). It is included in this repository to guarantee stability/reproducibility. I do ***not*** claim to be an author of the ChemEXIN repository, or its supporting data and model weights cloned here: all credit goes to Sharma et al. The ChemEXIN clone is redistributed here under its original [GPL v3 License](/ChemEXIN_modified/LICENSE), and with its original [README.md](/ChemEXIN_modified/README.md). The modifications I made are summarized in [MODIFICATIONS.md](/ChemEXIN_modified/MODIFICATIONS.md), so please read that file for more details. Modifications were necessary to run the benchmarking evaluation described in the pre-print, and only modifications necessary for that analysis were made. 

2. [trinucleo_sharma_et_al_2025_params.csv](/trinucleo_sharma_et_al_2025_params.csv) and [tetranucleo_sharma_et_al_2025_params.csv](/tetranucleo_sharma_et_al_2025_params.csv) contain the physicochemical parameter mappings from the [latest publication by Sharma et al.](https://doi.org/10.1039/d4mo00241e) These were used as inputs for my models and pipeline in this project. The values in these files are copies of the mappings by Sharma et al., which are publicly available as [supplemental data](https://www.rsc.org/suppdata/d4/mo/d4mo00241e/d4mo00241e1.xlsx) and within the cloned [ChemEXIN repository](https://github.com/rnsharma478/ChemEXIN/tree/master/param_files). I do ***not*** claim to be the original author of this data, and I did not modify the values of any data (only their storage format).

[**Human_Exon_Length_Distribution.csv**](/Human_Exon_Length_Distribution.csvv) contains data points estimated from Figure 1 in a publication by Mokry et al. in 2010 (DOI: [10.1093/nar/gkq072](https://doi.org/10.1093/nar/gkq072)). This was done using a tool that facilitates estimating data points from images of 2D plots. This approach is functionally analogous to using a ruler to estimate the relationship between axis units and absolute length units for a graph on paper, and then estimating the coordinates for each point via ruler measurements and conversion. I do not claim that this data is a high-fidelity recreation of the original data portrayed in Figure 1 of that paper, but I also do ***not*** claim to be the author that figure or its source data. This approach was a crude but quick stand-in for higher quality data on the distribution of exon lengths in the human genome at a point in the project where I was tight on time.

# Using this repository

While this repository is unsupported and was written without users in mind, it is hypothetically possible to run all of the Python code provided in the correct order and reproduce the results presented in the [preprint](https://doi.org/10.64898/2025.12.19.694709). If that is your goal, once setup is complete, please see the [Script and Notebook Overview](#script-and-notebook-overview) section for more context on each script and notebook and where they fall within this project. That said, this compendium already contains almost every intermediate and final data file, so re-running all of the code is not necessary.

## Pre-requisites

Before installing dependencies and attempting to run any Python code in this repository, you will need at least some of the following:

1) **A Linux or Unix-like system with Bash**
    + Required to run scripts that fetch the data hosted elsewhere, even if you do not plan on running any Python code.
    + A GUI and web browser will also be useful if you want to view or run the Jupyter notebooks.
    + This research project was conducted on Ubuntu 22.04 LTS running within WSL2 on Windows 10.

3) **Git**
    + Required to clone this repository.
    + If you do not already have it available in your command-line, please see the [Git](https://git-scm.com/install/linux) or [GitHub](https://github.com/git-guides/install-git) installation instructions for your specific Linux or Unix-like system.

2) **Miniconda or Conda**
    + Only necessary if you plan on running Python code from this repository.
    + Please see the official [quickstart Miniconda installation instructions for Linux](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2).

4) **NVIDIA GPU** with an **official proprietary NVIDIA driver** already properly installed on your system. 
    + ***This is only necessary if you specifically plan on running the model training code, which uses TensorFlow/Keras.***
    + If running "`nvidia-smi`" in your terminal shows a recent driver version and an accurate list of your GPU(s), you are probably good to go!
    + **For AMD GPUs**: It may be possible to run the model training code with an AMD GPU, but this is completely untested. I have provided an alternative environment file called [environment-amd.yml](environment-amd.yml), which should provide a starting point analogous to the NVIDIA-compatible environment setup in [environment.yml](environment.yml). That alternative file may work out of the box, but will likely require a slight modification to the version numbers on one or two lines to work with your specific hardware. Please see the official AMD documentation on [TensorFlow + ROCm installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/tensorflow-install.html) for more details on which versions of ROCm (and which hardware) are compatible with specific TensorFlow versions. Model training and testing was conducted using TensorFlow 2.19, so bear that in mind when adjusting version numbers in [environment-amd.yml](environment-amd.yml) to work with your hardware. You will need an open-source amdgpu kernel driver for this setup to work, rather than a legacy proprietary driver.

## Disk Space Considerations

The dependencies and data files involved in this research project take up a lot of disk space altogether. Below is some information on the required drive space to better inform your decisions when cloning and reproducing all or some of this repository. All file sizes listed below are for the *decompressed* or otherwise fully prepared form of the relevant file(s) on your drive:

+ **Clone of this GitHub repo:** ~613 MB

+ **Conda environment:** ~11 GB

+ **All files from [Zenodo.org](https://doi.org/10.5281/zenodo.19074687):** ~11 GB
    + `1_Exon_Annotation.zip`: 123 MB
    + `2_Selected_Coords_Seqs.zip`: 1.1 GB
    + `3_Physicochemical_Profiles.zip`: 4.2 GB
    + `4_Sliding_Window.zip`: 5.6 GB
    + `ChemEXIN_modified.zip`: 68 MB

+ **GRCh38.p14 files from [NCBI](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.40/):** ~ 3 GB

+ **3_Physicochemical_Profiles/all_profiles.npy:** ~90 GB
    + Not hosted on GitHub or Zenodo (must be created by running a Python script)
    + Holds a NumPy array of floating point numbers with a shape of (~5.6 M, 28, 77)

Based on the numbers above, to clone this repository, install all dependencies, and fetch/reproduce all data files, it would take **a total of ~116 GB**. 

Fortunately, all of these files are not strictly required to explore this research compendium. None of the data is technically necessary to understand the code. There is little reason to build the conda environment unless you plan on re-running or re-using the Python code. Also, having the structural profiles file is not necessary for understanding the project, so long as its purpose and shape are understood (see the [preprint](https://doi.org/10.64898/2025.12.19.694709)).

## Setup Instructions

There are two general ways to use this repository. The first is as a static record of the data files as they existed on my drive during the final days of the research project when the manuscript was being written (plus a few convenience scripts and context notes added later for you). The second is with an interest in rerunning or reusing the code to produce all or some of the results. If the former is your goal, skip the next section and go straight to [Retrieving the Missing Data](#retrieving-the-missing-data). If you are interested in running any Python code in this repository, then complete the steps in the next section and review the [Script and Notebook Overview](#script-and-notebook-overview).

### Code Dependencies Installation

You will need at least **~15 GB of free disk space** to complete all of the steps outlined in this section, please see [Disk Space Considerations](#disk-space-considerations) for more details.

Make the directory where you would like this repository to live and cd into it. For example:

```bash
mkdir ~/My_Git_Clones/
cd ~/My_Git_Clones/
```

Clone this repository and then cd into it:

```bash
git clone https://github.com/GreenMtn-bioinfo/RBIF120-HEX-finder-training
cd ./RBIF120-HEX-finder-training/
```

Run the setup script, which will automatically build and configure the conda environment (assuming NVIDIA hardware):
+ If you have an AMD GPU, you will first need to review and possibly edit [environment-amd.yml](environment-amd.yml) to match your hardware requirements before proceeding (see the "**For AMD GPUs**" note above under [Pre-requisites](#pre-requisites) for more details). After that is done, run "`bash setup.sh amd`" or "`bash setup.sh AMD`" instead of the command below.

```bash
bash setup.sh # This assumes NVIDIA hardware
```
+ As a reminder, GPU support will not matter unless you want to run the training code.

Activate the conda environment that was just created:

```bash
conda activate JH-RBIF120-project
```
This environment will always need to be active when running the Python code found in this repository!

#### Data Required by the Code

While most of the externally hosted data files are not necessary if you plan on rerunning the complete code from the start (as they are outputs of it), there are two exceptions:

1. The GRCh38.p14 files from [NCBI](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.40/) are necessary for the first several scripts, which prepare the training/testing data sets from genomic sequences.

2. The ChemEXIN model weights on [Zenodo.org](https://doi.org/10.5281/zenodo.19074687) are required if you plan on running [ChemEXIN_modified/main.py](/ChemEXIN_modified/main.py) or anything in [sliding_window_testing.ipynb](/sliding_window_testing.ipynb).

+ Please see [Script and Notebook Overview](#script-and-notebook-overview) for more details on the code-containing files mentioned above.

If you think you need one or more of the data sets listed above, then run:

```bash
bash fetch_all_data.sh # With the conda environment activated!
```
+ Read the prompts and choose to download and prepare the GRCh38.p14 files from NCBI (enter "y").

+ Read the prompts and choose to download files from Zenodo (enter "y"), then choose to download only *some* files (enter "s"), and select `ChemEXIN_modified.zip` (enter "5").

You are now good to go! See the [Script and Notebook Overview](#script-and-notebook-overview) section for information on the run order and purpose of each script and notebook.

#### About the Jupyter Notebooks

The conda environment built and configured in the previous section includes `ipykernel` and `jupyterlab` to facilitate opening the Jupyter notebooks and running code in them. All you need to do to open a notebook is:

1. Activate the conda environment:
    ```bash
    conda activate JH-RBIF120-project
    ```

2. And then run:
    ```bash
    jupyter lab notebook_of_interest.ipynb
    ```
    **OR**
    ```bash
    jupyter lab # Simply navigate to and open the notebook of interest using the JupyterLab UI once it loads in your browser.
    ```

***Make sure the "Python (JH-RBIF120-project)" kernel is selected before running any code in any notebook!***

The Jupyter notebooks in this repository are messier than I usually like to keep them. There are some one-off snippets of exploratory code that are not part of the main analysis, and there are blocks that expect only *some* of the previous blocks to be run. There is important code that is commented out. That said, each notebook still has a roughly linear flow from start to finish, and the cell output were left as-is. I have confirmed that key bits of the code run properly using the correct kernel and with the required data files in place, but proceed investigating these with tailored expectations!

### Retrieving the Missing Data

Due to the file and repository size limits on GitHub, files for this research compendium >25 MB were uploaded as a record to Zenodo.org (DOI: [10.5281/zenodo.19074687](https://www.doi.org/10.5281/zenodo.19074687)). If you would like to simply examine or use the exact contents of this research repository without having to re-run all of the code, you can run the command below to initiate retrieval of some or all missing data. 

 + The full ~11 GB conda environment setup in the previous section is not required to run this script. That said, you will need a Bash terminal with `wget` (or `curl`), `unzip`, and `gzip` already installed. Additionally, if you intend to fetch and prepare the GRCh38.p14 files from NCBI you will need `samtools` already installed for the indexing step.

```bash
bash fetch_all_data.sh
```

If you run this script, you will be prompted on which portions of the missing files you would like to download (and will be shown the total decompressed sizes you are agreeing to). You can pick and choose, for example, skipping the reference genome files from NCBI will still allow you to fetch all or some of the [files on Zenodo.org](https://www.doi.org/10.5281/zenodo.19074687). See [Disk Space Considerations](#disk-space-considerations) for more details.

#### Structural Profiles Set

The only file this script will not get you is `3_Physicochemical_Profiles/all_profiles.npy`. This file contains the actual structural profiles for the entire training and testing set. It is ~90 GB, so it was not hosted on GitHub or Zenodo. The only way to get this file is to:

1. Follow the steps in the [Code Dependencies Installation](#code-dependencies-installation) section and then ***activate the conda environment***:
    ```bash
    conda activate JH-RBIF120-project
    ```

2. You will need to rename the [3_Physicochemical_Profiles/](/3_Physicochemical_Profiles/) directory, as the profile generation script will not overwrite existing data in its intended output directory:
    ```bash
    mv 3_Physicochemical_Profiles/ 3_Physicochemical_Profiles_original/ # From within the repo's root directory!
    ```

3. Run the following from the root directory of the repo:

    + You will **~90 GB of free disk space** before proceeding, please see [Disk Space Considerations](#disk-space-considerations) for more details.

    ```bash
    python profile_generator_3.py # This will take time, see next bullet point.
    ```
    
    + Even with multithreading, this step is intensive and can take some time. For reference, it takes **~45 minutes** on a well-cooled desktop system with a 12-thread CPU (2.5 GHz nominal, ~4 GHz boosted multi-core) and 64 GB DDR5 RAM. 

# Script and Notebook Overview

In case you plan on running the code in this repository, or even if you just want a better understanding of how this project was conducted, below is an overview of each of the code-containing files in approximate run order:

    NOTE: Some of the scripts and notebooks listed below have logic to avoid overwriting existing data in their intended output directory. This means that, out-of-box, the data files included in this repo are a hindrance to running some of the code. If you really intend to re-run code from a particular script or notebook, you should first note the output directory listed for that file below (if any), and then consider renaming or deleting that directory prior to running the code.

1. [preparation_1.py](/preparation_1.py): Acts as a storage module for project-wide parameters and paths used by the other scripts, particularly the next two. When run directly, it pulls RefSeq MANE Select exon features from the GRCh38.p14 annotation GFF expected in [0_Reference_Genome/](/0_Reference_Genome/) and sorts them into two files by strand.
    + The output directory for this script is [1_Exon_Annotation/](/1_Exon_Annotation/).

2. [get_boundary_seqs_2.py](/get_boundary_seqs_2.py): Contrary to the naming, this script samples and organizes sequence of ALL classes (acceptor, donor, exon, intron) and sub-classes (intergenic or intragenic intron) based on the sampling parameters stored in the previous script. 
    + The output directory for this script is [2_Selected_Coords_Seqs/](/2_Selected_Coords_Seqs/).
    + To be clear, in this project and the preprint, the words "boundary" and "junction" were used interchangeably as a generic catch-all for the acceptor/donor splice sites, which the trained model aims to detect. In the code and preprint, the acceptor splice site is often referred to as an "exon start" or "intron-exon" boundary or junction. Likewise, the donor splice site is often referred to as an "exon end" or an "exon-intron" boundary or junction.

3. [profile_generator_3.py](/get_boundary_seqs_3.py): This script uses multithreading to perform the intensive task of calculating the moving averages for each physicochemical/structural parameter. This is done for every sequence sampled in the previous step that does not contain N's. This script generates the large file `3_Physicochemical_Profiles/all_profiles.npy` discussed at the end of the previous section, as well as the corresponding metadata files that track information like the sequence IDs and class membership.
    + The output directory for this script is [3_Physicochemical_Profiles/](/3_Physicochemical_Profiles/).
    + This script depends on the mappings from Sharma et al., 2025 (DOI: [10.1039/D4MO00241E](https://doi.org/10.1039/D4MO00241E)), which were shared publicly in their [supplemental data](https://www.rsc.org/suppdata/d4/mo/d4mo00241e/d4mo00241e1.xlsx) and in their [ChemEXIN repository](https://github.com/rnsharma478/ChemEXIN/tree/master/param_files). These mappings are based on molecular dynamics simulations, and link all possible tri- and tetra-nucleotides to estimated values for each structural parameter. The files [trinucleo_sharma_et_al_2025_params.csv](/trinucleo_sharma_et_al_2025_params.csv) and [tetranucleo_sharma_et_al_2025_params.csv](/tetranucleo_sharma_et_al_2025_params.csv) in this repository are simply reformatted copies of that same data. I do ***not*** claim to be the original author of this data. Please see [External Code and Models](#external-code-and-models) for more information.
    + The terms "structural parameter" and "physicochemical parameter" were used interchangeably in the code and preprint. Sharma et al. used "physicochemical parameter" in their papers, but I prefer "structural parameter".

+ [build_data_set.sh](/build_data_set.sh): A utility script originally written to automatically run the previous three scripts in order to generate the training/testing profiles set and all accompanying metadata files.

4. [split_data_4.py](/split_data_4.py): Performs the training/testing partition, which is based around randomly selected *regions* of each source chromosome, rather than complete, sequence-level randomization. This approach is important to enable various forms of testing (such as the "sliding window" testing seen in the preprint) within regions of each genome without fear of strong information leakage. Sequence homology means soft leakage is always possible, though this is less like over longer genomic distances, which the regions-based partitioning approach helps to maintain. This script also calculates the proportions of each class and sub-class in the training and testing subsets to make sure they are approximately evenly represented in either. It tallies the final numbers of each relevant class and subclass and uses those numbers, along with high-level knowledge about the human genome, to estimate class weights used during training.
    + The output directory for this script is again [3_Physicochemical_Profiles/](/3_Physicochemical_Profiles/).

5. [get_normalization_params_5.py](/get_normalization_params_5.py): Uses multithreading and batching to calculate four arrays of statistics from the training subset of the structural profiles. These four arrays were used to normalize profiles during model training and testing. The chosen normalization method involved z-normalization first, followed by min-max normalization. As such, this script first calculates the mean and standard deviation for each structural parameter across all structural profiles in the training set. It then applies z-normalization and calculates the post-z-normalization minimum and maximum value for each parameter across the training profiles. Because all of the training profiles did not fit into the available RAM, the typical (and convenient) NumPy array operations for the calculation of statistics like mean, standard deviation, minimum, and maximum could not be used on the entire array at once. I had to write my own implementation of these operations that iterated over smaller batches of structural profiles to calculate the four normalization statistic arrays for the entire training set.

6. [plot_average_profile.ipynb](/plot_average_profile.ipynb): Calculates the z-normalized average profile for each parameter across all profiles in the training set. In other words, the value of each structural profile at a particular position in the sequence/profile was normalized across all training profiles. The average profiles were then normalized using the mean and standard deviation arrays computed by the previous script. This notebook also handles the plotting of the average profiles, and it is where Figure 6 in the preprint was finalized.
    + [plot_average_profile.py](/plot_average_profile.py): This was the original script that the core code in the Jupyter notebook is based on, but the notebook was used to finalize the calculation and visualization of the average profiles.

7. [TCN_training.ipynb](/TCN_training.ipynb), [MBDA-Net_training.ipynb](/MBDA-Net_training.ipynb), and [BiLSTM_training.ipynb](/BiLSTM_training.ipynb): These are the notebooks from which cross-validation, training, and testing were run. Jupyter notebooks provided a convenient way to log the terminal output from TensorFlow/Keras during these steps. They depend on the following two modules:
    1. [keras_models.py](/keras_models.py): The Python module that uses Keras to define the three model architectures that were chosen and trained for this project.
    2. [keras_utility_classes.py](/keras_utility_classes.py): This module does the heavy-lifting for cross-validation and final training/testing. It contains definitions for custom classes that managed the execution of training and evaluation, creation of cross-validation folds, computation and logging of metrics, enforcement of stopping criteria, and other "epoch-related" operations. Perhaps most importantly, this is where the data generator class is defined, without which this project would not have been possible on my hardware. When the training data set is too large to fit on available RAM, it needs to prepared and fed to the model in an efficient batch-wise manner. The [NpyAppendArray](https://pypi.org/project/npy-append-array/) module, used by [profile_generator_3.py](/get_boundary_seqs_3.py), was important for ensuring the data generator did not become completely rate-limited by slow file reading speeds, while avoiding the need to have an unwieldy number of separate NPY files. NPZ files have read times that are far too slow for a model training data generator, and the same goes for many other common data storage formats in the Python data science ecosystem. NPY files are convenient and relatively snappy to read, but cannot be appended to directly, which really matters for large data sets when RAM-limited. That is where [NpyAppendArray](https://pypi.org/project/npy-append-array/) was a game-changer for this intensive and time-constrained project.
    + The Jupyter notebooks logged metrics and saved the best epoch weights to the following directories:
        + [TCN_best_cv](/TCN_best_cv/), [MBDA-Net_best_cv](/MBDA-Net_best_cv/), and [BiLSTM_best_cv](/BiLSTM_best_cv/) for the cross-validation runs.
        + [TCN_final](/TCN_final/), [MBDA-Net_final](/MBDA-Net_final/), and [BiLSTM_final](/BiLSTM_final/) for the final training and evaluation runs.

8. [results_plotting_aggregation_CV.ipynb](/results_plotting_aggregation_CV.ipynb): This is where the data from the cross-validations runs for the three models was aggregated and plotted. The supplemental figures in the [preprint](https://doi.org/10.64898/2025.12.19.694709) were creating using the code in this notebook.

9. [results_plotting_aggregation_FINAL.ipynb](/results_plotting_aggregation_FINAL.ipynb): This is where the data from the final training and evaluation for the three models was aggregated and plotted. Figure 7 in the [preprint](https://doi.org/10.64898/2025.12.19.694709) was created using the code in this notebook.

10. [sliding_window_testing.ipynb](/sliding_window_testing.ipynb): This is where the "sliding window" evaluation discussed in the preprint was conducted and visualized. This notebook handled the sampling of the three sets of testing sequences from the held-out regions, as well as running the entire prediction pipeline on each sequence from those sets. It calculated performance metrics for the three models over each sequence set for my models and ChemEXIN, using RefSeq MANE Select exons as truth features. Finally, it handled plotting of the results for my models and ChemEXIN, as well as displaying predictions form either pipeline next to RefSeq exons for specific sequences.
    + [ChemEXIN_modified/main.py](/ChemEXIN_modified/main.py): While the Jupyter notebook above handled most of the metric calculation and plotting, this script ran the ChemEXIN prediction pipeline and logged its predictions for all three sequence sets. This is a modified version of the [original "main.py" script](https://github.com/rnsharma478/ChemEXIN/blob/master/main.py). It technically needs to be run **BEFORE** any code in [sliding_window_testing.ipynb](/sliding_window_testing.ipynb) is run. Modification of the authors' original source code was necessary to automate the ChemEXIN tool for the benchmarking analysis. Please see [ChemEXIN_modified/MODIFICATIONS.md](/ChemEXIN_modified/MODIFICATIONS.md) for more details on the changes made. Please also see the [External Code and Models](#external-code-and-models) section for details on exactly which components of this repository I am not the original author of.

All files in this repository not listed above are either data files, or contain code that was written after-the-fact for the convenience of anyone potentially interested in cloning this research compendium and retrieving all of its data.