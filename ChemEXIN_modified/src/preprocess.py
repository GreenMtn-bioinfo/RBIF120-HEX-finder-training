'''
This program file is part of "Exon-Intron Boundary Detection Made Easy by Physicochemical Properties of DNA" developed by the Computational Genomics group at Supercomputing Facility for Bioinformatics and Computational Biology (SCFBio), Kusuma School of Biological Sciences (KSBS), Indian Institute of Technology (IIT) Delhi, India.
Authors: Dinesh Sharma, Danish Aslam, Kopal Sharma, Aditya Mittal, B Jayaram.

Contact: run478@gmail.com, bjayaram@chemistry.iitd.ac.in

Lab Webpage: www.scfbio-iitd.res.in
'''

import sys
import simple_colors as SC

def check_single(seq_f):
	try:
		print(SC.blue("\nSequence pre-processing started"))
		with open(f"sequence/{seq_f}","r") as fasta:
			lines=fasta.readlines()
			full_seq=''
			if len(lines) > 1: 
				for line in lines:
					if '>' in line:
						continue
					else:
						full_seq=full_seq+line.strip()
				print(SC.magenta("\nThe input sequence file contains multiple lines.\nDo you want to retrieve the single line sequence file?(Y or y/N or n)"))
				response = input(SC.green("Waiting for user input: "))
				if response.upper()=="Y" or response.upper()=="YES":
					with open(f"results/{seq_f.split('.')[0]}_single","w") as out:
						out.write(full_seq)
					print(SC.yellow(f"\nFile saved to results/{seq_f.split('.')[0]}_single"))
					return str(full_seq)
				else:
					print(SC.yellow("Starting main analysis without saving single sequenece file"))
					print(SC.blue("pre-processing DONE :)"))
					return str(full_seq)
			elif len(lines)==1:
				print(SC.blue("pre-processing DONE :)"))
				return str(lines[0])
			else:
				print(SC.red("Input sequence File is Empty.\b Can not process further."))
				sys.exit()
	except:
		print(SC.red("The path is not valid or the input sequence file does not exist in sequence directory"))
		sys.exit()

