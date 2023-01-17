import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def nans(df):
	#get total row amount with nans and nans in every column
	print(df.shape[0]-df.dropna().shape[0],'rows with nans out of',df.shape[0])
	for c in df.columns:
		print(df[c].isna().sum(),'nans',c)

def hists(df,bins=30,figsize=(30,3),color=None,norm=False,alpha=0.9):
	plt.figure(figsize=figsize)
	if color is None:
		for i in range(len(df.columns)):
			 plt.subplot(1,len(df.columns),i+1)
			 plt.hist(df.iloc[:,i].values,bins=bins,density=norm)
			 plt.title(df.columns[i])
	else:
		options=color.unique()
		for i in range(len(df.columns)):
			 plt.subplot(1,len(df.columns),i+1)
			 plt.title(df.columns[i])
			 for opt in options:
				 plt.hist(df[color==opt].iloc[:,i].values,bins=bins,label=opt,density=norm,alpha=alpha)
			 plt.legend()

		

