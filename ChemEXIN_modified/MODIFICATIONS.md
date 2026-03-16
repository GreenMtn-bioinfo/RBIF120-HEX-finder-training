# About this repository clone

This sub-directory includes a *modified* version of cloned source code originally from the [ChemEXIN repository](https://github.com/rnsharma478/ChemEXIN) on GitHub.
+ The ChemEXIN repository is the culmination of work detailed in the journal article "Exon–intron boundary detection made easy by physicochemical properties of DNA" by Sharma et al. in 2025 (https://doi.org/10.1039/D4MO00241E).

The modifications were made by Jonathan Hummel (jhgmbioinfo@gmail.com) to facilitate the benchmarking of ChemEXIN against an independent implementation of the same underlying idea. This was one part of a final capstone project in the Bioinformatics MS program at Brandeis University.

**Modifications Summary**
+ The script [main.py](main.py) was modified to:
    1) Automatically use the model trained on human sequences/profiles without user input at the start
    2) Automatically use a probability threshold of 0.85 (to filter model predictions) without user input at the start
    3) Parse the user-specified FASTA for individual sequences and run a loop that calls the ChemEXIN prediction pipeline for each sequence, saving all predictions/output under a custom directory
+ Most other scripts in [src/](src/) were modified by removing print statements, which can really slow down a loop.
+ The code of the `prediction()` function was copied from [run_model.py](src/run_model.py) into [main.py](main.py).
    + Some of its code was moved outside of the function definition to avoid model loading on every loop iteration (i.e. for every sequence). This was necessary due to how TensorFlow loads models.
+ All changes, whether listed explicitly or not, were made with the objective of keeping the ChemEXIN prediction pipeline true to the repository version available at the time, while enabling automatic/repeated calling for a large set of test sequences stored in a single FASTA file.

In case this is unclear, I do **not** claim to be the original author of ChemEXIN, or a contributor to ChemEXIN or the underlying publication. The [GPL v3 license](LICENSE) chosen by the original authors of the cloned repository allows me to alter, re-use, and/or redistribute it, so long as the same license is maintained, major changes are declared, and credit to the original authors is given (all three of which have been done).

Please see the preprint "Evaluation of a structure-based method for ab initio gene detection using deep learning" by Hummel and Estrada for the results of the comparison, as well as more information on the capstone project (https://doi.org/10.64898/2025.12.19.694709).
