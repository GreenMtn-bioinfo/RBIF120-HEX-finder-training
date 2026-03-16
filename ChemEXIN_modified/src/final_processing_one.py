'''
This program file is part of "Exon-Intron Boundary Detection Made Easy by Physicochemical Properties of DNA" developed by the Computational Genomics group at Supercomputing Facility for Bioinformatics and Computational Biology (SCFBio), Kusuma School of Biological Sciences (KSBS), Indian Institute of Technology (IIT) Delhi, India.
Authors: Dinesh Sharma, Danish Aslam, Kopal Sharma, Aditya Mittal, B Jayaram.

Contact: run478@gmail.com, bjayaram@chemistry.iitd.ac.in

Lab Webpage: www.scfbio-iitd.res.in
'''

import pandas as pd
import numpy as np
import simple_colors as SC

def final_process(f_dic,prob):
    del f_dic[0]
    pos = []
    processed_keys = set() 

    for key in sorted(f_dic.keys()): 
        if key in processed_keys:  
            continue

        if f_dic[key][2] >= prob:
            c_stretch = [key]  
            processed_keys.add(key)  

            for next_key in range(key + 1, max(f_dic.keys()) + 1):
                if next_key in f_dic and f_dic[next_key][2] >= 0.70:
                    c_stretch.append(next_key)
                    processed_keys.add(next_key)  
                else:
                    break  

            pos.append(c_stretch)
    # print(SC.blue("\nSTEP 8/8: Exon-intron boundary capture, position refining and Output generation.")) 
    return(pos)
