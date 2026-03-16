'''
This program file is part of "Exon-Intron Boundary Detection Made Easy by Physicochemical Properties of DNA" developed by the Computational Genomics group at Supercomputing Facility for Bioinformatics and Computational Biology (SCFBio), Kusuma School of Biological Sciences (KSBS), Indian Institute of Technology (IIT) Delhi, India.
Authors: Dinesh Sharma, Danish Aslam, Kopal Sharma, Aditya Mittal, B Jayaram.

Contact: run478@gmail.com, bjayaram@chemistry.iitd.ac.in  

Lab Webpage: www.scfbio-iitd.res.in
'''
import numpy as np
import simple_colors as SC


def combine_params(pmap_inter,comb_tri_map):
	# print(SC.blue(f"\nSTEP 5/8: Combining individual tetra-nucletoide numerical profiles into major feature category and concatenating tri- and tetra-nucleotide profiles together."))
	inter_array = np.zeros(len(pmap_inter['l']))
	for i in pmap_inter.keys():
		inter_array+=pmap_inter[i]
	processed_array = np.divide(inter_array,len(pmap_inter))
	comb_tri_map["inter"]=list(processed_array)
	# print(SC.blue(f"STEP 5/8: SUCCESSFULLY COMPLETED :)"))
	return comb_tri_map
