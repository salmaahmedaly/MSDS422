import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import math
from sklearn.linear_model import LogisticRegression




def clean_df(file_path, remove_cols):
    df = pd.read_csv(file_path)
    df= df.drop(columns = remove_cols)

    return df

def split_df (df, train_cols, test_cols,test_size, rand_seed=1 ):
    x = df[train_cols]
    y = df[test_cols]
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_size, random_state=rand_seed)
    return x_train, x_test, y_train, y_test 


def plot_roc(name, fpr_train, tpr_train, fpr_test, tpr_test, roc_auc_train, roc_auc_test ):
    """
    plots ROC curve from both train and test datasets
    """
    plt.title(name)
    plt.plot(fpr_train, tpr_train, 'b', label = f'AUC TRAIN:{round(roc_auc_train,2)}')
    plt.plot(fpr_test, tpr_test, 'b', label = f'AUC TEST:{round(roc_auc_test,2)}', c='r')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def linear_reg_magic  (x_train, y_train, x_test, y_test, coef= False):
    #model
    all_vars_LR= LinearRegression()
    all_vars_LR = all_vars_LR.fit(x_train, y_train)
    #make predictions
    pred_train = all_vars_LR.predict(x_train)
    pred_test = all_vars_LR.predict(x_test)
    #MEAN
    mean_train = y_train.mean()
    mean_test = y_test.mean()
    #RMSE
    RMSE_train = math.sqrt( metrics.mean_squared_error( y_train, pred_train))
    RMSE_test = math.sqrt( metrics.mean_squared_error( y_test, pred_test))
    coef_dict = {}
    if coef:
        coef_dict['intercept'] = all_vars_LR.intercept_
        for coef, feat in zip(all_vars_LR.coef_,list(x_train.columns)):
            coef_dict[feat] = coef

    return mean_train, mean_test, RMSE_train, RMSE_test, coef_dict


def logistic_reg_magic(x_train, y_train, x_test, y_test, features):
    model = LogisticRegression()
    model = model.fit( x_train[features], y_train )

    #train
    pred_train = model.predict( x_train[features] )
    probs_train = model.predict_proba( x_train[features])

    
    acc_score_train = metrics.accuracy_score(y_train, pred_train)
    p1_train= probs_train[:,1]
    fpr_train, tpr_train, threshold_train = metrics.roc_curve( y_train,  p1_train)
    auc_train = metrics.auc(fpr_train,tpr_train)

    #test
    pred_test = model.predict( x_test[features] )
    probs_test = model.predict_proba( x_test[features])
    acc_score_test = metrics.accuracy_score(y_test, pred_test)
    p1_test= probs_test[:,1]
    fpr_test, tpr_test, threshold_test = metrics.roc_curve( y_test,  p1_test)
    auc_test = metrics.auc(fpr_test, tpr_test)

    return acc_score_train, acc_score_test, fpr_train, tpr_train, fpr_test, tpr_test, auc_train, auc_test

#Don's 
def getCoefLogit( MODEL, TRAIN_DATA ) :
    varNames = list( TRAIN_DATA.columns.values )
    coef_dict = {}
    coef_dict["INTERCEPT"] = MODEL.intercept_[0]
    for coef, feat in zip(MODEL.coef_[0],varNames):
        coef_dict[feat] = coef
    print("\nCRASH")
    print("---------")
    print("Total Variables: ", len( coef_dict ) )
    for i in coef_dict :
        print( i, " = ", coef_dict[i]  )


#Doug's 
def get_TF_ProbAccuracyScores( NAME, MODEL, X, Y ) :
    probs = MODEL.predict( X )
    pred_list = []
    for p in probs :
        pred_list.append( np.argmax( p ) )
    pred = np.array( pred_list )
    acc_score = metrics.accuracy_score(Y, pred)
    p1 = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve( Y, p1)
    auc = metrics.auc(fpr,tpr)
    return [NAME, acc_score, fpr, tpr, auc]

def print_Accuracy( TITLE, LIST ) :
    print( TITLE )
    print( "======" )
    for theResults in LIST :
        NAME = theResults[0]
        ACC = theResults[1]
        print( NAME, " = ", ACC )
    print( "------\n\n" )

    

def print_ROC_Curve( TITLE, LIST ) :
    pass
    fig = plt.figure(figsize=(6,4))
    plt.title( TITLE )
    for theResults in LIST :
        NAME = theResults[0]
        fpr = theResults[2]
        tpr = theResults[3]
        auc = theResults[4]
        theLabel = "AUC " + NAME + ' %0.2f' % auc
        plt.plot(fpr, tpr, label = theLabel )
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()