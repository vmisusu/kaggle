import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import openTSNE
import umap
from sklearn import decomposition

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

methods={}

def TSNE(df,color=None,figsize=(15,15),rewrite=False,alpha=0.7):
    if (not ("TSNE" in methods)) or rewrite or df.shape[0]!=methods["TSNE"].shape[0]:
        methods["TSNE"]=pd.DataFrame(openTSNE.TSNE(2).fit(df.values))
    plt.figure(figsize=figsize,alpha=alpha)
    if color is None:
        plt.scatter(methods["TSNE"][:,0],methods["TSNE"][:,1])
    else:
        options=color.unique()
        for opt in options:
            thisColor=methods["TSNE"][color==opt]
            plt.scatter(thisColor.iloc[:,0],thisColor.iloc[:,1],label=opt,alpha=alpha)
        plt.legend()
        
def UMAP(df,color=None,figsize=(15,15),rewrite=False,alpha=0.7):
    if (not ("UMAP" in methods)) or rewrite or df.shape[0]!=methods["UMAP"].shape[0]:
        methods["UMAP"]=pd.DataFrame(umap.UMAP(2).fit(df.values).transform(df.values)) #embedding_)
    plt.figure(figsize=figsize,alpha=alpha)
    if color is None:
        plt.scatter(methods["UMAP"][:,0],methods["UMAP"][:,1])
    else:
        options=color.unique()
        for opt in options:
            thisColor=methods["UMAP"][color==opt]
            plt.scatter(thisColor.iloc[:,0],thisColor.iloc[:,1],label=opt,alpha=alpha)
        plt.legend()
        
def PCA(df,color=None,figsize=(15,15),rewrite=False,alpha=0.7):
    if (not ("PCA" in methods)) or rewrite or df.shape[0]!=methods["PCA"].shape[0]:
        methods["PCA"]=pd.DataFrame(decomposition.PCA(2).fit_transform(df.values))
    plt.figure(figsize=figsize,alpha=alpha)
    if color is None:
        plt.scatter(methods["PCA"][:,0],methods["PCA"][:,1])
    else:
        options=color.unique()
        for opt in options:
            thisColor=methods["PCA"][color==opt]
            plt.scatter(thisColor.iloc[:,0],thisColor.iloc[:,1],label=opt,alpha=alpha)
        plt.legend()
