'''
This program file is part of "Exon-Intron Boundary Detection Made Easy by Physicochemical Properties of DNA" developed by the Computational Genomics group at Supercomputing Facility for Bioinformatics and Computational Biology (SCFBio), Kusuma School of Biological Sciences (KSBS), Indian Institute of Technology (IIT) Delhi, India.
Authors: Dinesh Sharma, Danish Aslam, Kopal Sharma, Aditya Mittal, B Jayaram.

Contact: run478@gmail.com, bjayaram@chemistry.iitd.ac.in

Lab Webpage: www.scfbio-iitd.res.in
'''
import pandas as pd
import numpy as np
import simple_colors as SC
def assign_params(param_map,plist):
    param_map['a'].append(plist['a'])
    param_map['b'].append(plist['b'])
    param_map['c'].append(plist['c'])
    param_map['d'].append(plist['d'])
    param_map['f'].append(plist['f'])
    param_map['g'].append(plist['g'])
    param_map['h'].append(plist['h'])
    param_map['i'].append(plist['i'])
    param_map['j'].append(plist['j'])
    param_map['k'].append(plist['k'])
    param_map['t'].append(plist['t'])
    param_map['u'].append(plist['u'])
    param_map['v'].append(plist['v'])
    param_map['w'].append(plist['w'])
    param_map['x'].append(plist['x'])
    param_map['y'].append(plist['y'])
    param_map['z'].append(plist['z'])
    param_map['aa'].append(plist['aa'])
    param_map['ab'].append(plist['ab'])
    param_map['ac'].append(plist['ac'])
    param_map['ad'].append(plist['ad'])
    param_map['ae'].append(plist['ae'])
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
    # print(SC.blue(f"STEP 2/8: SUCCESSFULLY COMPLETED :)"))
    return normalized_map


def calculateParameters(sequence):
	# print(SC.blue(f"\nSTEP 2/8: Converting sequence into normalised tri-nucleotide parameter profiles.")) 
	param_map = {'a':[],'b':[],'c':[],'d':[],'f':[],'g':[],'h':[],'i':[],'j':[],'k':[],'t':[],'u':[],'v':[],'w':[],'x':[],'y':[],'z':[],'aa':[],'ab':[],'ac':[],'ad':[],'ae':[]}
	noofbases = len(sequence)
	dframe = pd.read_csv("param_files/trimer.csv", index_col=0)
	if noofbases == 0:
		return
	trimotifs = []
	for m in range(noofbases - 2):
		trimotifs.append(sequence[m:m + 3])
	for motif in trimotifs:
		if (motif in dframe.columns):
			assign_params(param_map, dframe[motif])

	return calculateMovingAverages(param_map)
