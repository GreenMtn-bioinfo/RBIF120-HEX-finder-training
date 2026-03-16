'''
This program file is part of "Exon-Intron Boundary Detection Made Easy by Physicochemical Properties of DNA" developed by the Computational Genomics group at Supercomputing Facility for Bioinformatics and Computational Biology (SCFBio), Kusuma School of Biological Sciences (KSBS), Indian Institute of Technology (IIT) Delhi, India.
Authors: Dinesh Sharma, Danish Aslam, Kopal Sharma, Aditya Mittal, B Jayaram.

Contact: run478@gmail.com, bjayaram@chemistry.iitd.ac.in 

Lab Webpage: www.scfbio-iitd.res.in
'''

### PLEASE SEE "MODIFICATIONS.md" IN THE ROOT DIRECTORY OF THIS CLONE FOR MORE DETAILS!!
### NOTE: This is modified source code from another repo called ChemEXIN (https://github.com/rnsharma478/ChemEXIN),
### which was the culmination of work that I was NOT involved in (https://doi.org/10.1039/D4MO00241E).
### This file, and others in the src/ directory of this repo clone, were modified by Jonathan Hummel for 
### a capstone research project in the Brandeis University GPS Bioinformatics Program that involved creating and
### evaluating my own implementation of the same underlying from the original authors (i.e. the researchers listed above).
### One component of that project was benchmarking my models against ChemEXIN, as well as known
### exons in the RefSeq MANE Select annotation set. All modifications made to the original
### source code of this repository were only made to silence print output and enable ChemEXIN to quickly and
### automatically make predictions for a larger number of sequences. This was was necessary for the benchmarking.
### I was very careful to avoid altering anything that could affect the performance of the core prediction pipeline.
### Please see the preprint "Evaluation of a structure-based method for ab initio gene detection using deep learning"
### for the final results of the capstone, as well as more context (https://doi.org/10.64898/2025.12.19.694709)


import time
import sys
import pandas as pd
import simple_colors as SC
from src import input_seq_check, combine_tri, combine_tetra, prediction_df, norm_tri, norm_tetra, preprocess, final_processing_one, final_processing_two,results # , run_model
import numpy as np
import os
from tensorflow.keras.models import load_model

# print(SC.cyan("\n\t\t\tChemEXIN: A Physicochemical Parameter-Based Exon-Intron Boundary Prediction Method developed by SCFBio, IIT Delhi."))
print(SC.magenta(f"Input your file name present in the sequence directory and hit ENTER:"))
file_name = input(SC.green(f"Waiting for user input: "))
file_name_base = file_name.split(".")[0]
os.mkdir(f'./sequence/{file_name_base}/')

# org = input(SC.magenta("\nSelect the organism and hit ENTER:")+SC.red("\n> h or H for H. sapiens")+SC.red("\n> m or M for M. musculus")+SC.red("\n> c or C for C. elegans")+SC.green("\nWaiting for user input: "))
# if org.upper() == "H" or org.upper() == "M" or org.upper() == "C":
#     print(SC.green(f"\nCarrying forward the analysis with the selected Organism."))
# else:
#     print(SC.red(f"The entered option doesn't correspond to a valid organism.\nRerun the analysis with the correct options."))
#     sys.exit()
org = 'H' # can be used to bypass the prompt

# print(SC.magenta(f"\nSelect the threshold value and hit ENTER else hit ENTER to proceed with default (0.75):"))
# prob = input(SC.red("> a or A for PROB: 0.70")+SC.red("\n> b or B for PROB: 0.80")+SC.red("\n> c or C for PROB: 0.85")+SC.green("\nWaiting for user input: "))
# if prob.upper() == "A":
#     prob=0.70
#     print(SC.green(f"Carrying forward the analysis with the selected threshold value: {prob}"))
# elif prob.upper() == "B":
#             prob=0.80
#             print(SC.green(f"Carrying forward the analysis with the selected threshold value: {prob}"))
# elif prob.upper() == "C":
#             prob=0.85
#             print(SC.green(f"Carrying forward the analysis with the selected threshold value: {prob}"))
# else:
#     prob = 0.75
#     print(SC.red(f"The entered option doesn't correspond to a valid threshold.\nCarrying the analysis with the default value (0.75)."))
prob = 0.85 # can be used to bypass the prompt


if org == "H":
    mod="H. sapiens"
    # print(SC.blue(f"\nSTEP 7/8: Running the") + SC.blue(" H. sapiens")+ SC.blue(f" prediction model."))
    model = load_model("models/3d_cnn_model_hg.h5")
elif org == "M":
    mod="M. musculus"
    # print(SC.blue(f"\nSTEP 7/8: Running the") + SC.blue(" M. musculus",'italic')+ SC.blue(f" prediction model."))
    model = load_model("models/3d_cnn_model_mg.h5")
else:
    mod="C. elegans"
    # print(SC.blue(f"\nSTEP 7/8: Running the") + SC.blue(" C. elegans",'italic')+ SC.blue(f" prediction model."))
    model = load_model("models/3d_cnn_model_eg.h5")


# This function needs to be defined here and the model built once for better performance in the loop
def prediction(df_final):

    test_data = pd.DataFrame.from_dict(df_final)
    X_test = test_data.values
    X_test = X_test.reshape(len(df_final), 50, 7, 1, 1)
    y_pred = model.predict(X_test, verbose=0) # JH: silenced the model during prediction
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    predicted_df ={}
    for i in range(len(y_pred_classes)):
        if y_pred_classes[i]==2:
            predicted_df[i]=[1,y_pred[i][0],y_pred[i][1] + y_pred[i][2]]
        else:
            predicted_df[i]=[y_pred_classes[i],y_pred[i][0],y_pred[i][1] + y_pred[i][2]]
    # print(SC.blue(f"STEP 7/8: SUCCESSFULLY COMPLETED :)"))
    return predicted_df


with open(f'./sequence/{file_name}', mode = 'r') as fasta:
    lines = fasta.readlines()
    lines = [line.strip('\n') for line in lines]
    seqs = lines[1::2]
    headers = [line.strip('>') for line in lines[::2] if line !='']
n_seqs = len(headers)
# processed_seq = preprocess.check_single(f"{('.').join(file_name)}")

report_iter = int(n_seqs*0.1) # report every 10% of sequences completed
start=time.time()
iter_start = start
per_iter = []

# output_input_check = input_seq_check.readsequencefile(processed_seq)
for i in range(n_seqs):
# if output_input_check:
    
    processed_seq = seqs[i].upper()
    
    if 'N' in processed_seq:
        print(f"{headers[i]} skipped due to one or more N's.")
        continue
    
    normalised_map_tri = norm_tri.calculateParameters(processed_seq)
    combine_dict_tri = combine_tri.combine_params(normalised_map_tri)
    normalised_map_tetra = norm_tetra.calculateParameters(processed_seq)
    all_param_combined_final = combine_tetra.combine_params(normalised_map_tetra,combine_dict_tri)
    final_df = prediction_df.non_over_50(all_param_combined_final)
    pred_results = prediction(final_df)

    positions_one = final_processing_one.final_process(pred_results,float(prob))
    # print(SC.green(f"	Check 1: PASSED ---> Positions captured successfully."))
    final_refined_pos = final_processing_two.process_pos(positions_one,pred_results)
    # print(SC.green(f"	Check 2: PASSED ---> Positions refined successfully."))
    result=results.filter_dict(final_refined_pos,headers[i],mod,prob)
    result.index = result.index+1
    result.index.name="S.No."
    result.to_csv(f"./sequence/{file_name_base}/{headers[i]}_results.csv", mode = "a")
    # print(SC.green(f"	Check 3: PASSED ---> Output saved successfully to results/{headers[i]}_results.csv"))
    # print(SC.blue("\nSTEP 8/8: SUCCESSFULLY COMPLETED :)"))
    if i != 0 and (i + 1) % report_iter == 0:
        percent_done = 100*(i+1)/n_seqs
        per_iter.append((time.time() - iter_start)/report_iter)
        mean_rate = np.mean(np.array(per_iter))
        etc = mean_rate * (n_seqs - (i + 1)) / 60
        print(f'Finished inference for {percent_done:.0f}% of sequences. Elapsed: {((time.time()-start)/60):.1f} min. ETC: {etc:.1f} min.')
        iter_start = time.time()
# else:
    # print("\nCANNOT PROCESS FURTHER !")
end=time.time()
print(SC.red(f"Total Execution time: {end-start} secs"))

