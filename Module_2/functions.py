import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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