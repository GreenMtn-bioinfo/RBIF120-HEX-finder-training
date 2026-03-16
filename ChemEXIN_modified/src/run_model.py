'''
This program file is part of "Exon-Intron Boundary Detection Made Easy by Physicochemical Properties of DNA" developed by the Computational Genomics group at Supercomputing Facility for Bioinformatics and Computational Biology (SCFBio), Kusuma School of Biological Sciences (KSBS), Indian Institute of Technology (IIT) Delhi, India.
Authors: Dinesh Sharma, Danish Aslam, Kopal Sharma, Aditya Mittal, B Jayaram.

Contact: run478@gmail.com, bjayaram@chemistry.iitd.ac.in.

Lab Webpage: www.scfbio-iitd.res.in
'''
import pandas as pd
import numpy as np
import simple_colors as SC
from tensorflow.keras.models import load_model

def prediction(df_final,name,organism):
	if organism == "H":
		mod="H. sapiens"
		# print(SC.blue(f"\nSTEP 7/8: Running the") + SC.blue(" H. sapiens")+ SC.blue(f" prediction model."))
		model = load_model("models/3d_cnn_model_hg.h5")
	elif organism == "M":
		mod="M. musculus"
		# print(SC.blue(f"\nSTEP 7/8: Running the") + SC.blue(" M. musculus",'italic')+ SC.blue(f" prediction model."))
		model = load_model("models/3d_cnn_model_mg.h5")
	else:
		mod="C. elegans"
		# print(SC.blue(f"\nSTEP 7/8: Running the") + SC.blue(" C. elegans",'italic')+ SC.blue(f" prediction model."))
		model = load_model("models/3d_cnn_model_eg.h5")
			
	test_data = pd.DataFrame.from_dict(df_final)
	X_test = test_data.values
	X_test = X_test.reshape(len(df_final), 50, 7, 1, 1)
	y_pred = model.predict(X_test, verbose=0) # JH silenced the model during prediction
	y_pred_classes = np.argmax(y_pred, axis=1)
	
	predicted_df ={}
	for i in range(len(y_pred_classes)):
		if y_pred_classes[i]==2:
			predicted_df[i]=[1,y_pred[i][0],y_pred[i][1] + y_pred[i][2]]
		else:
			predicted_df[i]=[y_pred_classes[i],y_pred[i][0],y_pred[i][1] + y_pred[i][2]]
	# print(SC.blue(f"STEP 7/8: SUCCESSFULLY COMPLETED :)"))
	return predicted_df,mod
