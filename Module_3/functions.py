import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import math



def clean_df(file_path, remove_cols):
    df = pd.read_csv(file_path)
    df= df.drop(columns = remove_cols)

    return df

def split_df (df, train_cols, test_cols,test_size, rand_seed=1 ):
    x = df[train_cols]
    y = df[test_cols]
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_size, random_state=rand_seed)
    return x_train, x_test, y_train, y_test 


def plot_roc(fpr_train, tpr_train, fpr_test, tpr_test, roc_auc_train, roc_auc_test ):
    """
    plots ROC curve from both train and test datasets
    """
    plt.title('TREE ROC CURVE')
    plt.plot(fpr_train, tpr_train, 'b', label = f'AUC TRAIN:{round(roc_auc_train,2)}')
    plt.plot(fpr_test, tpr_test, 'b', label = f'AUC TEST:{round(roc_auc_test,2)}', color= 'red')
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