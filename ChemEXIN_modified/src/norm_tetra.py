'''
This program file is part of "Exon-Intron Boundary Detection Made Easy by Physicochemical Properties of DNA" developed by the Computational Genomics group at Supercomputing Facility for Bioinformatics and Computational Biology (SCFBio), Kusuma School of Biological Sciences (KSBS), Indian Institute of Technology (IIT) Delhi, India.
Authors: Dinesh Sharma, Danish Aslam, Kopal Sharma, Aditya Mittal, B Jayaram.

Contact: run478@gmail.com, bjayaram@chemistry.iitd.ac.in

Lab Webpage: www.scfbio-iitd.res.in
'''
import numpy as np
import pandas as pd
import simple_colors as SC

def assign_params(param_map,plist):
    param_map['l'].append(plist['l'])
    param_map['ma'].append(plist['ma'])
    param_map['n'].append(plist['n'])
    param_map['o'].append(plist['o'])
    param_map['p'].append(plist['p'])
    param_map['q'].append(plist['q'])
    return param_map

def calculateMovingAverages(param_map):
    moving_win_size = 25
    moving_param_map = {}
    for k, v in param_map.items():
        arr = np.array(v)
        weights = np.ones(moving_win_size) / moving_win_size
        moving_averages = np.convolve(arr, weights, mode='valid')
        moving_param_map[k] = moving_averages.tolist()
    # print(SC.green(f"	Check 1: PASSED ---> Computed moving averages."))
    return normalizeMovingAverages(moving_param_map)


def normalizeMovingAverages(moving_param_map):
    normalized_map = {}
    for k, arr in moving_param_map.items():
        arr = np.array(arr)  
        arr_min = arr.min()
        arr_max = arr.max()
        rang = arr_max - arr_min
        if rang == 0:  
            normalized_arr = np.zeros_like(arr)
        else:
            normalized_arr = (arr - arr_min) / rang

        normalized_map[k] = normalized_arr.tolist()
    # print(SC.green(f"	Check 2: PASSED ---> Normalised moving averages."))
    # print(SC.blue(f"STEP 4/8: SUCCESSFULLY COMPLETED :)"))
    return normalized_map


def calculateParameters(sequence):
	# print(SC.blue(f"\nSTEP 4/8: Converting sequence into normalised tetra-nucleotide parameter profiles."))
	param_map = {'l':[],'ma':[],'n':[],'o':[],'p':[],'q':[]}
	noofbases = len(sequence)
	dframe = pd.read_csv("param_files/tetramer.csv", index_col=0)
	if noofbases == 0:
		return
	trimotifs = []
	for m in range(noofbases - 3):
		trimotifs.append(sequence[m:m + 4])
	for motif in trimotifs:
		if (motif in dframe.columns):
			assign_params(param_map, dframe[motif])

	return calculateMovingAverages(param_map)
