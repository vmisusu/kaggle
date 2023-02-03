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
    
    
###DATASET PREPARATION
###DATASET PREPARATION
###DATASET PREPARATION
###DATASET PREPARATION
###DATASET PREPARATION
###DATASET PREPARATION
###DATASET PREPARATION
###DATASET PREPARATION


#https://www.kaggle.com/competitions/playground-series-s3e2/discussion/377373 SMOTE

import category_encoders as ce #https://www.kaggle.com/competitions/playground-series-s3e2/discussion/377827

#toBins [('column1',bins1),('column2',bins2)] - turn each specified number fearure into <bin> separate values
#unite [['column1','column2','column3'],['column1','column5']] - unite columns, replace with mean target (y) order
#     should be good if rows are too few and feats are too many
#drop ['column1',2,3] - drop columns
#one-hot True - run get_dummies to turn any str feature into one-hot encoding
#norm - normalization, "minmax" - turn to 0..1

def col(df,column): #get column name for non-str column
    if type(column)!=str:
        return df.columns[column]
    return column

def prepareDataset(df, targetCol, datasetParams, dfTest=None, verb=0):
    #Prepare dataset, using customization parameters
    #also prepare test data (without y) if provided
    df_updated = df.copy() #copy original df not to spoil it
    test = not (dfTest is None)
    if test:
        dfTest_updated=dfTest.copy()
    params = datasetParams.copy()
    defaultParams = {"TO_BINS":[],"UNITE":[],"DROP":[],"KEEP":[],"ONE-HOT":False,"NORM":"None","ENCODING":[]}
    for p in defaultParams: #use default params for missing
        if not (p in params): params[p] = defaultParams[p]
    targetCol = col(df, targetCol)
    df_updated = df_updated.drop(targetCol, axis = 1) #drop target column, to add in the end
    if test and (targetCol in dfTest_updated.columns):
        dfTest_updated = dfTest_updated.drop(targetCol, axis = 1)
    for cols in [params["DROP"], params["KEEP"]]: #make num format for column names in parameters
        for i in range(len(cols)):
            if type(cols[i] == str):
                cols[i] = df.columns.get_loc(cols[i])
    #############################################
    for pair in params["TO_BINS"]: #slice these num feats by bins
        c = col(df_updated,pair[0])
        minVal, maxVal = df_updated[c].min(), df_updated[c].max()
        binSize = (maxVal-minVal) / pair[1]
        df_updated[c] = df_updated[c] - minVal
        df_updated[c] = df_updated[c].floordiv(binSize) * binSize
        if test:
            dfTest_updated[c] = dfTest_updated[c] - minVal
            dfTest_updated[c] = dfTest_updated[c].floordiv(binSize) * binSize
        if verb >= 2: #make histogram to check how consistent are values
            plt.figure()
            df_updated[c].hist()
    #############################################          
    for union in params["UNITE"]: #unite features
        cols = [] #get columns in one format
        newColumnName = "" #new column name from these columns
        for column in union:
            cols.append(df_updated.columns.get_loc(col(df_updated,column)))
            newColumnName += df_updated.columns[cols[-1]] + "_"
        newColumnName = newColumnName[:-1]
        newColumnInd = len(df_updated.columns)
        df_updated[newColumnName] = df_updated.iloc[:,0] #just add new column
        if test:
            dfTest_updated[newColumnName] = dfTest_updated.iloc[:,0] #just add new column

        uValues = {} #new unique values and mean target for them
        for row in range(df_updated.shape[0]):
            unionValue = ""
            for c in cols: #get new value from union
                unionValue += str(df_updated.iat[row,c])+"_"
            uValues[unionValue] = 0
            df_updated.iat[row,newColumnInd] = unionValue #put the new value to the new column
            
        if test: #same for submission or validation
            for row in range(dfTest_updated.shape[0]):
                unionValue=""
                for c in cols: #get new value from union
                    unionValue += str(dfTest_updated.iat[row,c])+"_"
                uValues[unionValue] = 0
                dfTest_updated.iat[row,newColumnInd] = unionValue #put the new value to the new column
                
        valList=[] #put mean targets in list to sort and get order (useless for decision trees)
        for val in uValues:
            newVal = df[df_updated[newColumnName] == val][targetCol].mean()
            uValues[val] = newVal
            valList.append((newVal,val))
        valList=sorted(valList)
        for p in valList:
            df_updated[newColumnName] = df_updated[newColumnName].replace(p[1],p[0])
            if test:
                dfTest_updated[newColumnName] = dfTest_updated[newColumnName].replace(p[1],p[0])
                
        if verb >= 2: #make histogram to check how consistent are values
            plt.figure()
            df_updated[newColumnName].hist()

        if verb>=1: #print count of new values, this shouldn't be too big for dataset
            print(newColumnName,len(uValues))
    #############################################        
    #perform columns drop
    df_updated = df_updated.drop(df_updated.columns[params["DROP"]], axis = 1)
    if test:
        dfTest_updated = dfTest_updated.drop(dfTest_updated.columns[params["DROP"]], axis = 1)
    #############################################        
    #perform columns keep
    if len(params["KEEP"]) > 0:
        df_updated = df_updated.iloc[:,params["KEEP"]]
        if test:
            dfTest_updated = dfTest_updated.iloc[:,params["KEEP"]]
    #############################################
    #perform encoding
    for encoding in params["ENCODING"]:
        if encoding[0] == "ORDINAL":
            print(encoding)
            #mapping=None
            #if len(encoding) > 2:
            #    mapping=encoding[2]
            encoder = ce.OrdinalEncoder(cols=[col(df_updated, c) for c in encoding[1]], return_df=True) #, mapping=mapping)
            df_updated = encoder.fit_transform(df_updated)
            if test:
                dfTest_updated = encoder.transform(dfTest_updated)
    #############################################    
    if params["ONE-HOT"]==True:
        df_updated = pd.get_dummies(df_updated)
        if test:
            dfTest_updated = pd.get_dummies(dfTest_updated)
    #############################################
    if params["NORM"]=="minmax":
        df_updated=(df_updated-df_updated.min())/(df_updated.max()-df_updated.min())
        if test:
            dfTest_updated=(dfTest_updated - df_updated.min()) / (df_updated.max() - df_updated.min())
        #sc = StandardScaler() #can be used later for something else
        #X[num_cols] = sc.fit_transform(X[num_cols])
    #############################################
    if verb>=3:
        display(df_updated.head(12))
    if not test:
        if verb >= 1:
            print(df_updated.columns)
        return (df_updated.values, df[targetCol].values)
    if test:
        if verb >= 1:
            print('train',df_updated.columns)
            print('val/sub',dfTest_updated.columns)
        return (df_updated.values, df[targetCol].values, dfTest_updated.values)
    
NEED_TEST = False #run test in case of RnD or param search
if NEED_TEST:
    datasetParams={"ENCODING":["ORDINAL",[1,3,"Gender"]]}
    X, Y = prepareDataset(df, df.columns[-1], datasetParams, verb=3)


###TRAINING FUNCTIONS
###TRAINING FUNCTIONS
###TRAINING FUNCTIONS
###TRAINING FUNCTIONS
###TRAINING FUNCTIONS
###TRAINING FUNCTIONS

##Grid run

#dirty

#getParamConfigList should be in training_lib
def runGridTest(df, datasetParams={}, trainingParams={}, libSpecificParams={}, kfolds=10, kfoldrepeats=3, stopRightAway=False, dfAdditional=None):
    datasetParamsList = getParamConfigList(datasetParams)
    trainingParamsList = getParamConfigList(trainingParams)
    libSpecificParamsList = getParamConfigList(libSpecificParams)
    print(len(datasetParamsList),'dataset params configs')
    print(len(trainingParamsList),'training params configs')
    print(len(libSpecificParamsList),'lib specific params configs')
    print(len(datasetParamsList)*len(trainingParamsList)*len(libSpecificParamsList),'all params configs')
    print('datasetParams',datasetParamsList)
    print('trainingParams',trainingParamsList)
    print('libSpecificParams',libSpecificParamsList)
    results=[]
    for datasetParamConfig in datasetParamsList:
        
        if dfAdditional is None:
            X, Y = prepareDataset(df, df.columns[-1], datasetParamConfig) #prepare dataset with this param configurations
        else:
            X, Y, X2 = prepareDataset(df, df.columns[-1], datasetParamConfig, dfAdditional) #prepare dataset with this param configurations
            #_, Y2 = prepareDataset(dfAdditional, dfAdditional.columns[-1], datasetParamConfig) #TODO - how to use transform from initial? Or should I make common? Probably its better to use contest dataset and not to adjust to orig
            Y2 = dfAdditional.iloc[:,-1].values
            #df, targetCol, datasetParams, dfTest=None, verb=0
        for trainingParamConfig in trainingParamsList:
            for libSpecificParamConfig in libSpecificParamsList:
                totalRes=0
                Losses=[]
                stf = RepeatedStratifiedKFold(n_splits=kfolds, n_repeats=kfoldrepeats, random_state=1) #make k-folds
                for train_ix, test_ix in tqdm(stf.split(X, Y), total = kfolds*kfoldrepeats):
                    if False:
                        df2=df.iloc[train_ix]
                        subDf=df.iloc[test_ix]
                        X_Train, Y_Train, X_Test = prepareDataset(df2, df.columns[-1], datasetParamConfig, subDf)
                        
                        Y_Pred=trainingParamConfig["FUNC"](X_Train, Y_Train, X_Test, subDf.iloc[:,-1].values, X_Test, trainingParamConfig, libSpecificParamConfig) #run training and get predictions
                    else:
                        if dfAdditional is None:
                            Y_Pred=trainingParamConfig["FUNC"](X[train_ix], Y[train_ix], X[test_ix], Y[test_ix], X[test_ix], trainingParamConfig, libSpecificParamConfig) #run training and get predictions
                        else:
                            Y_Pred=trainingParamConfig["FUNC"](np.vstack([X[train_ix],X2]), np.hstack([Y[train_ix],Y2]), X[test_ix], Y[test_ix], X[test_ix], trainingParamConfig, libSpecificParamConfig) #run training and get predictions
                    #auc=roc_auc_score(Y[test_ix], Y_Pred) #calculate AUC
                    #print(auc,'pre-auc')
                    #Y_Pred=fixPredsForAUC(Y_Pred)
                    
                    #auc=roc_auc_score(Y[test_ix], Y_Pred) #calculate AUC
                    if len(Y_Pred.shape)>1: #multiclass probs
                        Y_Pred2=Y_Pred.argmax(1)+3 #he cant handle probs
                    else: #regression
                        Y_Pred2=np.round(Y_Pred)
                    loss=cohen_kappa_score(Y[test_ix], Y_Pred2, weights='quadratic')
                    
                    #print(auc,'auc')
                    totalRes+=loss
                    Losses.append(loss)
                    predsFalse=[]
                    predsTrue=[]
                    for i in range(len(Y[test_ix])):
                        if Y[test_ix][i]==0:
                            predsFalse.append(Y_Pred[i])
                        else:
                            predsTrue.append(Y_Pred[i])
                    #plt.figure(figsize=(8,8))
                    #plt.hist(predsFalse,color='blue',bins=30)
                    #plt.hist(predsTrue*10,color='red',bins=30)
                    if stopRightAway:
                        break
                #plt.figure()
                #plt.hist(AUCs) #hist for AUCs in k-folds
                #plt.title(str(totalRes/kfolds)+" "+str(datasetParamConfig)+" "+str(trainingParamConfig))
                Losses=np.array(Losses)
                lossInfo="{:.4f}: ".format(totalRes/(kfolds*kfoldrepeats))+"{:.3f}".format(Losses.min())+"-{:.3f}".format(Losses.max())+" var: {:.5f}".format((Losses-Losses.mean()).var())
                print("kfolded loss",lossInfo,str(datasetParamConfig)+" "+str(trainingParamConfig)+" "+str(libSpecificParamConfig))
                results.append((totalRes/(kfolds*kfoldrepeats),str(datasetParamConfig)+" "+str(trainingParamConfig)+" "+str(libSpecificParamConfig)))
    return results
    
    
    
 ##Ensemble
 #https://www.kaggle.com/competitions/playground-series-s3e2/discussion/377732

def ensembleTrain(X,Y,X_val,Y_val,X_test,trainingParams,libSpecificParams):
    params=trainingParams.copy()
    defaultParams={"ALG":"", "SPLIT":0.5}
    for p in defaultParams:
        if not (p in params): params[p] = defaultParams[p]
    
    totalWeight=0
    
    #simple blending OR train
    #"ALG", "VAL"
    #if alg - make one new validation split (better stratified)
    if (params["ALG"] != ""):
        n2 = int(len(X) * params["SPLIT"]) #examples to train ensemble
        X2 = X[-n2:]
        Y2 = Y[-n2:]
        X1 = X[:-n2]
        Y1 = Y[:-n2]
        totalPreds_train = np.zeros(len(X2))
        totalPreds_test = np.zeros(len(X_test))
    else:
        totalPreds=np.zeros(len(X_test))
    
    for method in trainingParams["METHODS"]:
        #print(method)
        #print(X.shape,X_val.shape,X_test.shape)
        weight = 1
        if len(method) >= 2: weight = method[2]
        totalWeight += weight
        if (params["ALG"] == ""):
            Y_Pred = method[0]["FUNC"](X, Y, X_val, Y_val, X_test, method[0], method[1])
            Y_Pred -= Y_Pred.min() #normalize #TODO RANKING
            Y_Pred /= Y_Pred.max() #normalize
            #print(totalPreds.shape,Y_Pred.shape)
            totalPreds += Y_Pred * weight    
        else:
            Y_Pred_train = method[0]["FUNC"](X1, Y1, X2, Y2, X2, method[0], method[1])
            Y_Pred_test = method[0]["FUNC"](X2, Y2, X_val, Y_val, X_test, method[0], method[1])
            Y_Pred_test -= Y_Pred_train.min()
            Y_Pred_train -= Y_Pred_train.min()
            Y_Pred_test /= Y_Pred_train.max()
            Y_Pred_train /= Y_Pred_train.max()
            totalPreds_train += Y_Pred_train * weight
            totalPreds_test += Y_Pred_test * weight
        
        
    if params["ALG"] != "": #perform meta-training for ensembling
        #Y_Pred1 = totalPreds / totalWeight
        #print(X.shape,X2.shape,Y_Pred1.shape)
        totalPreds_train /= totalWeight
        totalPreds_test /= totalWeight
        #print(totalPreds_train.shape,totalPreds_test.shape)
        Y_Pred = params["ALG"](totalPreds_train.reshape(-1,1), Y2, [], Y_val, totalPreds_test.reshape(-1,1), trainingParams, libSpecificParams)
        return Y_Pred
    else:    
        return totalPreds / totalWeight
        
        
        
##catboost etc

#dirty

#catboost training functions
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv

def catBoostTrain(X,Y,X_val,Y_val,X_test,trainingParams,libSpecificParams):
    params=trainingParams.copy()
    defaultParams={"ITERS":300,"LOSS":"Poisson","SEED":1,"CLASSIFIER":False} #"st_weight":1,
    for p in defaultParams:
        if not (p in params):
            params[p] = defaultParams[p]
    if params["CLASSIFIER"]:
        cat = CatBoostClassifier(iterations=params["ITERS"], loss_function=params["LOSS"], use_best_model=True, thread_count=2, random_seed=params["SEED"],**libSpecificParams) #, hints="skip_train~false")#, use_best_model=True) #class_weights={0:1,1:params["st_weight"]}, 
    else:
        cat = CatBoostRegressor(iterations=params["ITERS"], loss_function=params["LOSS"], use_best_model=True, thread_count=2, random_seed=params["SEED"],**libSpecificParams) #, hints="skip_train~false")#, use_best_model=True) #class_weights={0:1,1:params["st_weight"]}, 
    cat_feats=[] #handle str and other features
    sample_weight=np.ones(len(Y))
    if ("CLASS_WEIGHT" in trainingParams) and (trainingParams["CLASS_WEIGHT"]>1):
        w=trainingParams["CLASS_WEIGHT"]
        for i in range(len(Y)):
            if Y[i]==1:
                sample_weight[i]=w
    for i in range(X.shape[1]):
        if type(X[0,i])==str:
            cat_feats.append(i)
    cat.fit(X, Y, cat_features=cat_feats, plot=False, verbose=False, eval_set=(X_val,Y_val), sample_weight=sample_weight) #train
    if params["CLASSIFIER"]:
        return cat.predict_proba(X_test)
    else:
        return cat.predict(X_test)

#random forest training

from sklearn.ensemble import RandomForestClassifier

def randomForestTrain(X,Y,X_val,Y_val,X_test,trainingParams,libSpecificParams):
    rfc = RandomForestClassifier(random_state=1,**libSpecificParams) #n_estimators=3,max_depth=2,
    rfc.fit(X, Y.reshape(-1))
    return rfc.predict_proba(X_test)[:,1]

#xgboost

import xgboost as xgb

def XGBTrain(X,Y,X_val,Y_val,X_test,trainingParams,libSpecificParams): #trainX,indTrainX,trainY,valX,indValX,params):
    model=xgb.XGBRegressor(**libSpecificParams)
    model.fit(X,Y)
    preds=model.predict(X_test)
    return preds

#lightgbm

import lightgbm as lgbm

def LGBMTrain(X,Y,X_val,Y_val,X_test,trainingParams,libSpecificParams):
    model=lgbm.LGBMRegressor(**libSpecificParams)
    model.fit(X,Y)
    preds=model.predict(X_test)
    return preds
    
#Testing development
NEED_TEST = False
if NEED_TEST:
    datasetParams={"ONE-HOT":True}
    X, Y = prepareDataset(df, datasetParams) #prepare dataset with all the params
    rfc = RandomForestClassifier(random_state=1)
    rfc.fit(X[:-1000], Y[:-1000].reshape(-1))
    Y_Pred=rfc.predict_proba(X[-1000:])[:,1]
    auc=roc_auc_score(Y[-1000:], Y_Pred) #cal
    
    
##Tensorflow

#dirty and stupid lib version

#tensorflow training
from tensorflow.python.keras import layers as L
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers
import numpy as np

import tensorflow as tf
print('tensorflow version',tf.__version__)

def buildDense(columns,params):
    #Build dense neural network model
    inputLayer=L.Input(columns,name="Input")
    X=inputLayer
    for layer in params["LAYERS"]:
        X=L.Dense(layer,activation=None,activity_regularizer=params["REGULARIZATION"])(X)
        #X=L.BatchNormalization()(X)
        if params["DROPOUT"]>0:
            X=L.Dropout(params["DROPOUT"])(X)
        X=L.Activation(activation=params["ACTIVATION"])(X)
    output=L.Dense(1,activation="tanh")(X)
    model = Model(inputs=[inputLayer], outputs=[output])
    return model

def tensorFlowTrain(X,Y,X_val,Y_val,X_test,trainingParams,libSpecificParams):
    params=trainingParams.copy()
    defaultParams={"MODEL":buildDense,"LAYERS":[64],"ACTIVATION":"relu","DROPOUT":0,"REGULARIZATION":None,"LOSS":"mse","ITERS":50,"OPTIMIZER":"Adam","LR":0.001,"VERB":0}
    for p in defaultParams:
        if not (p in params):
            params[p] = defaultParams[p]
    model=params["MODEL"](X.shape[1],params)
    model.compile(loss=params["LOSS"], optimizer=params["OPTIMIZER"])
    K.set_value(model.optimizer.learning_rate, params["LR"])
    hist=model.fit(X,Y,epochs=params["ITERS"],validation_split=0.05,verbose=params["VERB"])
    preds=model.predict(X_test)
    return preds

NEED_TEST = False
if NEED_TEST:
    pass #code to run this
    
###HELP
###HELP
###HELP

helps={}
helps[""] = "Main run:\ndatasetParams={}\ntrainingParams={\"FUNC\":catBoostTrain}\nlibSpecificParams={}\n\nres=runGridTest(df,datasetParams,trainingParams,libSpecificParams,10,3)\n\n\n===\nprepareDataset(df, targetCol, datasetParams, dfTest=None, verb=0)\n\n===\ncatBoostTrain\nrandomForestTrain\nXGBTrain\nLGBMTrain\n\nensembleTrain\n\ntensorFlowTrain"
helps["prepareDataset"]="prepareDataset(df, targetCol, datasetParams, dfTest=None, verb=0):\ndf - DataFrame with train and validation (but not submission/test)\ntargetCol - name of Y column,\ndatasetParams - (\"TO_BINS\",\"UNITE\",\"DROP\",\"KEEP\","\"ONE-HOT\"\"NORM\",\"ENCODING\") -dict of params (run help(\"datasetParams\") for more),\ndfTest - DataFrame to get Y_Pred (without Y_True) - feat transform is done from df stats,\nverb - verbose level"
help["datasetParams"] = "datasetParams - (\"TO_BINS\",\"UNITE\",\"DROP\",\"KEEP\","\"ONE-HOT\"\"NORM\",\"ENCODING\")\n\nTO_BINS - list of feats to cut into bins. Example - [[0,20],[\"Age\",15]]\n\
UNITE - list of lists of feats to unite and sub with mean-Y order (use drop for raw feats if needed). Example - [[\"Age\",\"HeartDecease\",\Gender\"],[\"Job\",\"Education\"]\n\
DROP - drop feats from df (after other calculations). Example - [0,\"Age\",4,\"Job\"]\n\
KEEP - drop everything, except specified feats. Example - [0,\"Age\",4,\"Job\"]\n\
ONE-HOT - if true - run make_dummies to turn ENUM or STRING feats into one-column for each option (after other calculations). Example - True\n\
NORM - normalize all feats (after other calculations). Options - \"minmax\". Example - "minmax"\n\
ENCODING - to be done"
help["prepareDataset.datasetParams"] = help["datasetParams"]
help["dataset"] = help["prepareDataset"]
help["trainingParams"] = "FUNC - catBoostTrain, randomForestTrain, XGBTrain, LGBMTrain, ensembleTrain, tensorFlowTrain......."
help["train"] = help["trainingParams"]
    
def help(stWhat = "", unfold = False):
	print(helps[stWhat])
	if unfold:
		for tip in helps:
			if (len(helps[tip]) > len(stWhat)) and (helps[tip][:len(stWhat)]==stWhat):
				print(helps[tip])
	print()
	
