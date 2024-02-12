# import packages

#Data Structures
import pandas as pd
import numpy as np
import re
from IPython.display import display

#Sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score

#Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns



# Excldue columns showing dates
def exclude_date_fields(df):
    col_exclude = []
    for col in df.columns:
        if "date" in col:
            col_exclude += [col]

    df = df.drop(col_exclude, axis=1)
    print("Columns Excluded:", len(col_exclude))
    print(col_exclude)
    return df



# exclude columns with no variations
def exclude_single_value_fields(df):
    col_exclude = []
    for col in df.columns:

        if len(pd.unique(df[col][pd.notnull(df[col])])) == 1:
            col_exclude += [col]

    df = df.drop(col_exclude, axis=1)
    print("Columns Excluded:", len(col_exclude))
    print(col_exclude)
    return df



def exclude_fields_missing_values(df, threshold):
    col_null_percent = df.isnull().sum(axis=0) / df.shape[0]
    col_exclude = list(col_null_percent[col_null_percent > threshold].index)

    df = df.drop(col_exclude, axis=1)
    print("Columns Excluded:", len(col_exclude))
    print(col_exclude)
    return df



# show the percentage of null values in each column
def show_missing_value_percent(df, title):
    col_null_percent = df.isnull().sum(axis=0) / df.shape[0]
    #print(col_null_percent)
    plt.figure(figsize = (20,3))
    plt.bar(col_null_percent.index, col_null_percent.values)
    plt.xticks(rotation = 90)
    plt.ylim(bottom = 0, top = 1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig("plot" + "\\" + title + '.png')
    plt.show()



# show a few scatter plot made of PCA components
def PCA_scatter(n, Xtrain_pca, ytrain):
    index_0 = np.where(ytrain == 0)[0]
    index_1 = np.where(ytrain == 1)[0]
    plt.figure(figsize=(10, np.ceil(n/2) * 4))

    for i in range(n):
        plt.subplot(int(np.ceil(n/2)), 2, i + 1)
        plt.scatter(Xtrain_pca[index_0, i], Xtrain_pca[index_0, i + 1], label=0, c='red', s=3, alpha=0.2, marker='.')
        plt.scatter(Xtrain_pca[index_1, i], Xtrain_pca[index_1, i + 1], label=1, c='blue', s=3, alpha=0.1, marker='+')
        plt.title(" PCA component " + str(i) + " and " + str(i + 1))
        plt.legend(markerscale=5)

    plt.tight_layout()
    plt.savefig('plot\Scatter Plots of PCA Variables.png')
    plt.show()


def evaluate_classification_result(model_name, train_or_test, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    print("Prediction accuracy of ", model_name, " - ", train_or_test, ": ", np.round(accuracy, 4))

    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print("Prediction Sensitivity of ", model_name, " - ", train_or_test, ": ", np.round(sensitivity, 4))

    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])
    print("Prediction Precision of ", model_name, " - ", train_or_test, ": ", np.round(precision, 4))

    print("\n")
    print("Confusion Matrix of ", model_name, " - ", train_or_test, ": ")
    print(cm)




def  making_predictions(classifier, threshold, X, y):
    y_pred_proba = classifier.predict_proba(X)

    idx = np.where(y_pred_proba[:, 1] >= threshold)
    y_pred = np.zeros(y.shape)
    y_pred[idx] = 1

    return y_pred




def cap_outliers(array, k=3):
    upper_limit = array.mean() + k*array.std()
    lower_limit = array.mean() - k*array.std()
    array[array<lower_limit] = lower_limit
    array[array>upper_limit] = upper_limit
    return array




def cross_validation(classifier, classifier_name, Xtrain, ytrain, cv, threshold = None):

    for i, (train_index, test_index) in enumerate(cv.split(Xtrain)):
        #print(f"Fold {i}:")
        #print(f"  Train: index={train_index}     ", len(train_index))
        #print(f"  Test:  index={test_index}     ", len(test_index))


        Xtrain_cv = Xtrain[train_index, :]
        ytrain_cv = ytrain[train_index]
        Xtest_cv = Xtrain[test_index, :]
        ytest_cv = ytrain[test_index]

        model = classifier.fit(Xtrain_cv, ytrain_cv)

        if threshold == None:
            ypred_cv = model.predict(Xtest_cv)
        else:
            y_pred_proba_cv = classifier.predict_proba(Xtest_cv)

            idx = np.where(y_pred_proba_cv[:, 1] >= threshold)
            ypred_cv = np.zeros(ytest_cv.shape)
            ypred_cv[idx] = 1

        if i == 0:
            cm = confusion_matrix(ytest_cv, ypred_cv)
        else:
            cm += confusion_matrix(ytest_cv, ypred_cv)

        #print(confusion_matrix(ytest_cv, ypred_cv))

    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    TP = cm[1,1]
    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])

    print("Confusion Matrix of ",classifier_name)
    #print(cm)
    cm_df = pd.DataFrame(cm)
    columns = [('PREDICT', '0'), ('PREDICT', '1')]
    cm_df.columns = pd.MultiIndex.from_tuples(columns)
    index = [('TRUE', '0'), ('TRUE  ', '1')]
    cm_df.index = pd.MultiIndex.from_tuples(index)

    cm_df.to_csv("tables\Confusion Matrix of " + classifier_name + ".csv")

    display(cm_df)
    print("\n")

    result = pd.DataFrame({"Classifier": [classifier_name],
                           "True Negative": [TN],
                           "False Positive": [FP],
                           "False Negative": [FN],
                           "True Positive": [TP],
                           "Classification Accuracy": [accuracy],
                           "Sensitivity": [sensitivity],
                           "Precision": [precision]})
    return result