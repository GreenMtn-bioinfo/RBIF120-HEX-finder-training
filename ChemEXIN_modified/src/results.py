'''
This program file is part of "Exon-Intron Boundary Detection Made Easy by Physicochemical Properties of DNA" developed by the Computational Genomics group at Supercomputing Facility for Bioinformatics and Computational Biology (SCFBio), Kusuma School of Biological Sciences (KSBS), Indian Institute of Technology (IIT) Delhi, India.
Authors: Dinesh Sharma, Danish Aslam, Kopal Sharma, Aditya Mittal, B Jayaram.

Contact: run478@gmail.com, bjayaram@chemistry.iitd.ac.in

Lab Webpage: www.scfbio-iitd.res.in
'''

import pandas as pd
import datetime

def filter_dict(d,f_name,model,prob):
    keys_to_remove = []
    for key in range(1, len(d)):  
        difference = abs(d[key][0][0] - d[key-1][1][1])
        if difference <= 30:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del d[key]
    
    return process(d,f_name,model,prob)

def process(input_dict,file_name,model,prob):
    data_for_df = []

    for k, v in input_dict.items():
        first_l, second_l = v
        data_for_df.append((first_l[0], first_l[1], second_l[0], second_l[1]))

    df = pd.DataFrame(data_for_df, columns=['Primary_start', 'Primary_end', 'Secondary_start', 'Secondary_end'])
    with open(f"results/{file_name}_results.csv", "w") as top:
        top.write(f"#ChemEXIN OUTPUT FILE\n\n#ChemEXIN 1.0 Output generated on: {datetime.datetime.now()}\n\n#Input Parameters:\n#Employed Model---> {model}\n#Input File Name---> {file_name}\n\n#Field description:\n#S.No.---> Predicted boundary serial number.\n#Primary_start---> Target boundary window start site.\n#Primary_end---> Target boundary window end site.\n#Secondary_start---> Extended boundary window start site.\n#Secondary_end---> Extended boundary window end site.\n\n#Predicted Exon-intron boundaries at {prob} reliability threshold value.\n#Threshold value corresponds to the prediction probability of the boundary windows.\n\n")

    return df

