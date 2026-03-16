'''
This program file is part of "Exon-Intron Boundary Detection Made Easy by Physicochemical Properties of DNA" developed by the Computational Genomics group at Supercomputing Facility for Bioinformatics and Computational Biology (SCFBio), Kusuma School of Biological Sciences (KSBS), Indian Institute of Technology (IIT) Delhi, India.
Authors: Dinesh Sharma, Danish Aslam, Kopal Sharma, Aditya Mittal, B Jayaram.

Contact: run478@gmail.com, bjayaram@chemistry.iitd.ac.in

Lab Webpage: www.scfbio-iitd.res.in
'''
import simple_colors as SC
def non_over_50(df):
	# print(SC.blue(f"\nSTEP 6/8: Creating Final prediction dataframe."))
	non_over_dic = {}
	quot = len(df['bpaxis'])//50
	remain = len(df['bpaxis']) % 50
	for i in range(0,quot*50,50):
		non_over_dic[i]=df['bpaxis'][i:i+50]+df['intra'][i:i+50]+df['inter'][i:i+50]+df['bbone'][i:i+50]+df['hbond'][i:i+50]+df['stack'][i:i+50]+df['sol'][i:i+50]
	# print(SC.blue(f"STEP 6/8: SUCCESSFULLY COMPLETED :)"))
	return non_over_dic
			
