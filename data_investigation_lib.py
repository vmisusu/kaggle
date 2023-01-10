import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def nans(df):
	#get total row amount with nans and nans in every column
	print(df.shape[0]-df.dropna().shape[0],'rows with nans out of',df.shape[0])
	for c in df.columns:
		print(df[c].isna().sum(),'nans',c)

def hists(df,bins=30,figsize=(30,3)):
	plt.figure(figsize=figsize)
	for i in range(len(df.columns)):
		 plt.subplot(1,len(df.columns),i+1)
		 plt.hist(df.iloc[:,i].values,bins=bins)
		 plt.title(df.columns[i])

