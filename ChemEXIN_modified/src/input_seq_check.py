'''
This program file is part of "Exon-Intron Boundary Detection Made Easy by Physicochemical Properties of DNA" developed by the Computational Genomics group at Supercomputing Facility for Bioinformatics and Computational Biology (SCFBio), Kusuma School of Biological Sciences (KSBS), Indian Institute of Technology (IIT) Delhi, India.
Authors: Dinesh Sharma, Danish Aslam, Kopal Sharma, Aditya Mittal, B Jayaram.

Contact: run478@gmail.com, bjayaram@chemistry.iitd.ac.in

Lab Webpage: www.scfbio-iitd.res.in
'''
import sys
import simple_colors as SC
def readsequencefile(cont):
	print(SC.blue("\nSTEP 1/8: Checking for the sequence length and sequence characters."))
	content=cont.strip()
	if len(content)>=180 and len(content)<= 2500000:
		print(SC.green(f"	Check 1: PASSED ---> The input sequence length >= 180 and <= 2500000 (input length = {len(content)})."))
		count=0
		for i in range(len(content)):
			if content[i] not in 'ATGC':
				count+=1
				print(SC.red(f"	Check 2: FAILED ---> Characters other than A, T, G and C are present in the sequence (at pos:{i+1})."))
				break
			else:
				count=0
	else:
		print(SC.red(f"	Check 1: FAILED ---> The input sequence length < 180 or is > 2500000 (input length = {len(content)})."))
		sys.exit()
	if count==0:
		print(SC.green(f"        Check 2: PASSED ---> All the input sequence characters are valid."))
		print(SC.blue(f"STEP 1/8: SUCCESSFULLY COMPLETED :)"))
		return content
	else:
		print(SC.red(f"STEP 1/8: FAILED :|")) 
		return("")
