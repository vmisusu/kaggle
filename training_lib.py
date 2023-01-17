import numpy as np
import pandas as pd

def getParamConfigListRec(configsList,params,ind): #get all possible configs from dict. All lists in dict will be variated ([list] should be used for 1-option of list)
    pt=ind
    bFound=False
    for p in list(params.keys())[ind:]:
        pt+=1
        print(params[p],type(params[p]))
        if type(params[p])==list:
            bFound=True
            orig = params[p]
            for v in orig:
                params[p] = v
                getParamConfigListRec(configsList, params, pt)
            params[p] = orig
            break
    if not bFound:
        configsList.append(params.copy())

def getParamConfigList(params): #run recursion for getting all possible configs from dict
    configsList=[]
    paramsCopy=params.copy()
    getParamConfigListRec(configsList,paramsCopy,0)
    return configsList
