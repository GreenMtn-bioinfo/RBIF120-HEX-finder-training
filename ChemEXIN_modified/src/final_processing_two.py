'''
This program file is part of "Exon-Intron Boundary Detection Made Easy by Physicochemical Properties of DNA" developed by the Computational Genomics group at Supercomputing Facility for Bioinformatics and Computational Biology (SCFBio), Kusuma School of Biological Sciences (KSBS), Indian Institute of Technology (IIT) Delhi, India.
Authors: Dinesh Sharma, Danish Aslam, Kopal Sharma, Aditya Mittal, B Jayaram.

Contact: run478@gmail.com, bjayaram@chemistry.iitd.ac.in

Lab Webpage: www.scfbio-iitd.res.in
'''

import pandas as pd

def process_pos(pos,f_dic):
	for i in range(len(pos)):
		if len(pos[i]) >= 3:

			if len(pos[i])%2 == 0:
				f_index = len(pos[i])//2
				s_index = f_index + 1
				pos[i] = [pos[i][f_index-1], pos[i][s_index-1]]
			else:
				single_index = len(pos[i])//2 
				pos[i] = [pos[i][single_index]]
		else:
			continue
	final_data_positions={}
	for i in range(len(pos)):
		if len(pos[i])==1:
			exact_pos = [pos[i][0]*50, pos[i][0]*50+49]
			roi = [(pos[i][0]*50)-30, (pos[i][0]*50+49)+30]
		else:
			if f_dic[pos[i][0]][2] > f_dic[pos[i][1]][2]:
				exact_pos = [pos[i][0]*50, pos[i][0]*50+49]
				roi = [(pos[i][0]*50)-30, (pos[i][1]*50+49)+30]
			else:
				exact_pos = [pos[i][1]*50, pos[i][1]*50+49]
				roi = [(pos[i][0]*50)-30, (pos[i][1]*50+49)+30]
				
			
		final_data_positions[i] = [exact_pos,roi]
	return final_data_positions
	
