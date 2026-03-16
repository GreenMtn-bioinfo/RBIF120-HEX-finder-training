'''
This program file is part of "Exon-Intron Boundary Detection Made Easy by Physicochemical Properties of DNA" developed by the Computational Genomics group at Supercomputing Facility for Bioinformatics and Computational Biology (SCFBio), Kusuma School of Biological Sciences (KSBS), Indian Institute of Technology (IIT) Delhi, India.
Authors: Dinesh Sharma, Danish Aslam, Kopal Sharma, Aditya Mittal, B Jayaram.

Contact: run478@gmail.com, bjayaram@chemistry.iitd.ac.in

Lab Webpage: www.scfbio-iitd.res.in
'''
import numpy as np
import simple_colors as SC



def combine_params(pmap):
    # print(SC.blue(f"\nSTEP 3/8: Combining individual tri-nucletoide numerical profiles into the major feature categories."))
    main_param =  ["bbone","bpaxis","intra","hbond","stack","sol"]
    main_param_alpha = [["t","u","v","w","x","y","z","aa","ab"],["a","b","c","d"],["f","g","h","i","j","k"],["ac"],["ad"],["ae"]]

    param_names= {"bbone":[],"bpaxis":[],"intra":[],"hbond":[],"stack":[],"sol":[]}
    for i in range(len(main_param)):
        param_names[main_param[i]]=np.zeros(len(pmap['a']))
        for j in main_param_alpha[i]:
            param_names[main_param[i]]+=np.array(pmap[j])
        param_names[main_param[i]]=np.divide(param_names[main_param[i]],len(main_param_alpha[i]))
    for i in  param_names.keys():
        param_names[i]=list(param_names[i][0:(len(param_names[i])-1)])
    # print(SC.blue(f"STEP 3/8: SUCCESSFULLY COMPLETED :)"))
    return  param_names
