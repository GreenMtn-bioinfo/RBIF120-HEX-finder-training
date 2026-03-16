# ChemEXIN

An open-source, deep learning integrated, physicochemical property-based exon-intron boundary prediction method. It is based on a three-dimensional convolutional neural network (3D-CNN) architecture. Three organism-specific (_Homo sapiens_, _Mus musculus_, and _Caenorhabditis elegans_) models have been built and implemented in the final prediction pipeline. The universality of ChemEXIN lies in its ability to predict exon-intron boundaries across varying lengths of known genes (180 to ~2,500,000 bps) in the organisms under study.

Title: "Exon-Intron Boundary Detection Made Easy by Physicochemical Properties of DNA"

Authors: Dinesh Sharma, Danish Aslam, Kopal Sharma, Aditya Mittal, and B Jayaram

Note: ChemEXIN is a TensorFlow implementation.

Manuscript: Sharma, D., Aslam, D., Sharma, K., Mittal, A., & Jayaram, B. (2025). Exon–intron boundary detection made easy by physicochemical properties of DNA. Molecular Omics. 2025, 21, 226-239.

## Supported Operating System

It works as a command-line utility on Windows, Linux, and MacBook. It has been tested on Windows 10, Ubuntu 22.04, and macOS Sonoma (version 14) respectively.

## Installation

### Prerequisites
- **Conda**,  to create an environment for ChemEXIN across Windows, macOS, and Linux operating systems. This environment will include all the required dependencies. 
- Conda is available at https://conda.io/projects/conda/en/latest/user-guide/install/index.html.

### Setup (one-time setup)
**Clone the ChemEXIN project repository:**
```bash
 git clone https://github.com/rnsharma478/ChemEXIN.git
 ```
**Go to the project directory:**
```bash
 cd ChemEXIN
```
**Installing the virtual environment:**
```bash
conda env create -f ChemEXIN.yml
```
```bash
conda activate ChemEXIN
```
**Usage:**
```bash
cd ChemEXIN
```
```bash
python main.py
```
### Working Example

**STEP1:**
Considering that the user has successfully followed the Setup steps and the ChemEXIN directory is now available on the desired path, set the working directory to ChemEXIN (if it is not already there).
```bash
cd ChemEXIN
```
**STEP2:**
Activate the ChemEXIN environment and move to the ChemEXIN directory.
![image](https://github.com/rnsharma478/ChemEXIN/assets/112755171/fe1062a8-f8b8-4533-9b36-077305b8954c)

**STEP3:**

(A)	Check the contents of the directory.
![image](https://github.com/rnsharma478/ChemEXIN/assets/112755171/5c2bc5b1-0c1d-4fcc-9321-31dff3ac565a)

(B)	Run the prediction tool.
![image](https://github.com/rnsharma478/ChemEXIN/assets/112755171/d7093220-5209-42c5-9f1e-9521537e3783)

**STEP4:**

Provide the necessary inputs* when prompted.
![image](https://github.com/rnsharma478/ChemEXIN/assets/112755171/61a8da11-9486-4a14-80c2-966f2fe447cb)

*The text in white is the user input, in this case; "example.fasta" is the input file name containing the sequence in fasta format; "h" is the character input for the Homo sapiens model; "y" prompts the tool to save a single line file with the sequence named "example_single" with no header.*

**Note:** Single line file generation option is available only if the file contains a sequence spanning multiple lines. This option is an add-on utility that can be used for generating single lined sequences from multi-lined fasta files.

**STEP6:**

Users can track real-time progress following the processing statements displayed in the command prompt.
The displayed statements are as follows,

*STEP1/8.*

(A)	Checks the sequence length based on the length filter incorporated in the prediction pipeline.

`> For the input sequence “example.fasta”, the length = 10881 nucleotides.`

(B)	Sequence validity filter checks if all the characters in the sequence are valid DNA bases (A/T/G/C) or not.

`> The input sequence “example.fasta” has all the valid characters.`
![image](https://github.com/rnsharma478/ChemEXIN/assets/112755171/d18fee71-597e-49e8-ad14-6d0bb10e2863)

*STEP2/8.*

(A)	Converts the sequence into trinucleotide parameter profiles followed by its normalization.

`> Two “Checks” appear, one when the conversion ends and the other when normalization ends.`

(B)	 “SUCCESSFULLY COMPLETED :)” is prompted in case there are no errors.

*STEP3/8.*

(A)	Combines the individual trinucleotide numerical profiles into the major feature categories, i.e., BP-Axis, Backbone organization, Intra-BP organization, and the three energetics (Hydrogen bond energy, Solvation energy, and Stacking energy).

(B)	“SUCCESSFULLY COMPLETED :)” is prompted in case there are no errors.
 ![image](https://github.com/rnsharma478/ChemEXIN/assets/112755171/f42a6698-e6b3-4a28-9723-b4066a9661f6)

*STEP4/8.*

(A)	Converts the sequence into tetranucleotide parameter profiles followed by the normalization.

`> Two “Checks” appear, one when the conversion ends and the other when normalization ends.`

(B)	“SUCCESSFULLY COMPLETED :)” is prompted in case there are no errors.
![image](https://github.com/rnsharma478/ChemEXIN/assets/112755171/34ee0c4d-2a59-4d33-8687-2355fb52a8e4)

*STEP5/8.*

(A)	Combines the individual tetranucleotide numerical profiles into the major feature category i.e., Inter-BP organization.

(B)	Concatenates the combined tri- and tetra-nucleotide profiles together.

(C)	“SUCCESSFULLY COMPLETED :)” is prompted in case there are no errors.
![image](https://github.com/rnsharma478/ChemEXIN/assets/112755171/fc76ffdc-b4e4-483c-b111-8c16585a9fcf)

*STEP6/8.*

(A)	Creates the Final prediction data frame.

(B)	“SUCCESSFULLY COMPLETED :)” is prompted in case there are no errors.

*STEP7/8.*

(A)	Runs the user-selected 3D-CNN model (in this case, we chose “h”, so the *H. sapiens* prediction model is used).

(B)	“SUCCESSFULLY COMPLETED :)” is prompted in case there are no errors.
![image](https://github.com/rnsharma478/ChemEXIN/assets/112755171/2f62a0d1-e61d-4bda-9dd2-8c0eea13d9ad)

(C)	The probability threshold value prompt appears, the default is set at 0.75, if pressed enter without choosing a/A, b/B, or c/C, then predictions are automatically made on the 0.75 threshold. In this case, we proceeded with the default value by hitting ENTER.
![image](https://github.com/rnsharma478/ChemEXIN/assets/112755171/fb2b0cec-282a-451c-92d9-0ea41857153e)

*STEP8/8.*

(A)	Captures, refines and generates the position output.

`> Three “Checks” appear on all positions; capture, refining, and results generation.`

(B)	“SUCCESSFULLY COMPLETED :)” is prompted in case there are no errors.

(C)	Total Execution time is displayed for the entire run.

(D)	The prediction results are available in the results/ directory of the ChemEXIN.
![image](https://github.com/rnsharma478/ChemEXIN/assets/112755171/aa8e2c13-1ba2-446d-becc-367c7b5a1586)

**Output Description:**

In the output file, the user-specific metadata is available in the multi-line header; each line starts with a “#” character, followed by the predicted boundary windows information.

Metadata includes,

•	the user input parameters;

**Employed Model---> H. sapiens**

**Input File Name---> example**

•	the field description;

**S.No.---> Predicted boundary serial number.**

**Primary_ start---> Target boundary window start site.**

**Primary_end---> Target boundary window end site.**

**Secondary_start---> Extended boundary window start site.**

**Secondary_end---> Extended boundary window end site.**

•	The reliability threshold value description.

**Description:**
*The probabilities of exon-start (Class 1) and exon-end (Class 2) have been merged due to similar area under the receiver operating characteristic (AUROC) curves, as illustrated in Figure 4 of the main text. This similarity reflects the nearly identical biophysical profiles observed for acceptor and donor sites. Therefore, the predictions generated by our model undergo uniform processing procedures and incorporate multiple filters in the backend to enhance accuracy and reliability.*

The main filters include,

•	the sequence length filter;

Sequences must meet specific length criteria: they should be 180 nucleotides long or more but not exceed 2500000 nucleotides. This restriction is imposed because predictions are made on independent, non-overlapping windows of 50 nucleotides.

•	the terminal exon filter;

Specific precautions have been taken since exon-end sites could erroneously appear at the beginning of a sequence, or exon-start sites might be predicted towards the sequence's end due to our combined probabilities approach. Predictions within the sequence's initial window (0-50) are disregarded, as are any nucleotides beyond the last possible 50-nucleotide window from the input sequence.

•	the probability threshold filter;

(A)	The model searches for positions with the user-specified seeding threshold value (0.75 by default or 0.70/0.80/0.85). After creating a 50-length window using the predicted position, it designates this window as the target window and demarcates the start-/end- sites as the primary start and primary end. The model reports an extended window with the secondary start-/end- sites to refine the search scope and enhance the prediction reliability. This window spans –30 and +30, respectively, around the primary start and end sites.

(B)	If a subsequent position after a predicted seed position (predicted with 0.75 (default) or 0.70/0.80/0.85 threshold value) has a probability of boundary occurrence >=0.70, then the primary start-/end- sites are given for a window with respect to the position with a higher probability score, while the secondary start-/end- sites are given such that –30 upstream is taken for the first window’s start site, and +30 downstream is defined around the second window’s end site.

(C)	If a longer stretch comprising of three or more than three positions (each having probability >= 0.70) occurs (after the first seed threshold position), we can have two cases:

*(I) If the stretch has an even number of positions (including the first seed position), then for this stretch, n/2 and n+½ positions are considered (n=total number of positions in the stretch). 	Final processing is then applied as per step B (the primary start-/end- sites are reported for a window with respect to a position with higher probability, and the extended secondary window is created by moving -30 upstream from the first window’s start site and +30 nucleotides downstream from the second window’s end site).*

*(II) If the stretch has an odd number of positions, the median position is selected from the total number of positions (including the seed position). This gives a single position which undergoes further processing (to generate a window), the same as the single position processing explained in step A.*

**Note:** While the reliability threshold value is flexible, ranging from 0.70 to 0.85, it is advisable to consider factors such as over-representation and loss of information.
