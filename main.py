#!/usr/bin/env python
# coding: utf-8

# In[]:
import matplotlib.pyplot as plt
import pandas as pd

# models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# preprocessing and predictions
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# metrics
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, precision_recall_curve, average_precision_score

# In[]:
# read data
df = pd.read_csv('malicioussites.csv')

# replace null values with 0 in CONTENT_LENGTH column and drop all other rows with null columns
df['CONTENT_LENGTH'] = df['CONTENT_LENGTH'].fillna(0)
df = df.dropna()

# drop columns
drop_cols = ['URL', 'SERVER', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE']
df = df.drop(drop_cols, axis=1)

# hot encode categorical columns
df = pd.get_dummies(df)

# splits data into 5 folds for cross validation - ensures that each sample has the same % of each class
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
# splits data into X and y
X = df.drop('Type', axis=1)
y = df['Type']

# In[]:
# puts metrics into a dictionary
def calc_scores(y_pred, y_pred_classes, y):
    scores = {}
    scores['confusion_matrix'] = confusion_matrix(y, y_pred_classes)
    scores['auc'] = roc_auc_score(y, y_pred)
    scores['f1'] = f1_score(y, y_pred_classes)
    scores['pr'] = average_precision_score(y, y_pred)
    return scores

# handles cross validation of data and returns metrics
def cross_val(forest=True, n_estimators=100, learning_rate=0.3):
    if forest:
        pipeline = Pipeline(steps=[('model', RandomForestClassifier(n_estimators=n_estimators, random_state=1))])
    else:
        pipeline = Pipeline(steps=[('model', XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=1))])
    
    y_pred = cross_val_predict(pipeline, X, y, cv=skf, method='predict_proba')
    
    
    y_pred = y_pred[:,1]
    y_pred_classes = [round(x) for x in y_pred]
    scores = calc_scores(y_pred, y_pred_classes, y)
    #return y_pred, y_pred_classes
    return scores, {'y_pred': y_pred, 'y_pred_classes': y_pred_classes}

# In[]:
# random forest models
forest_default = cross_val(forest=True)
forest_150 = cross_val(forest=True, n_estimators=150)
forest_250 = cross_val(forest=True, n_estimators=250)
forest_500 = cross_val(forest=True, n_estimators=500)

# xgb models
xgb_default = cross_val(forest=False)
xgb_0 = cross_val(forest=False, n_estimators=150, learning_rate=0.05)
xgb_1 = cross_val(forest=False, n_estimators=250, learning_rate=0.05)
xgb_2 = cross_val(forest=False, n_estimators=500, learning_rate=0.05)

# In[]:
# random forest scores
print('Default forest model:\n', forest_default[0])
print('150 n_estimators forest model:\n', forest_150[0])
print('250 n_estimators forest model:\n', forest_250[0])
print('500 n_estimators forest model:\n', forest_500[0])

# xgb scores
print('Default xgb model:\n', xgb_default[0])
print('xgb 150 n_estimators, 0.05 lr model:\n', xgb_0[0])
print('xgb 250 n_estimators, 0.05 lr model:\n', xgb_1[0])
print('xgb 500 n_estimators, 0.05 lr model:\n', xgb_2[0])

# In[]:
# plots confusion matrix
def make_confusion_matrix(model):
    plt.subplot(131)
    ax = sns.heatmap(model[0]['confusion_matrix'], annot=True, xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'], cbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Actual')

# plots roc curve
def make_roc_curve(model):
    plt.subplot(132)
    FPR, TPR, _ = roc_curve(y, model[1]['y_pred'])
    
    plt.annotate('AUC: {}'.format(model[0]['auc']), (0, 0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top')
    plt.annotate('F1 score: {}'.format(model[0]['f1']), (0, 0), (0, -60), xycoords='axes fraction', textcoords='offset points', va='top')

    plt.plot(FPR, TPR)
    # diagonal line
    plt.plot([0, 1], [0, 1], '--', color='black')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

# plots precision-recall curve
def make_pr_curve(model):
    plt.subplot(133)
    plt.annotate('Average PR score: {}'.format(model[0]['pr']), (0, 0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top')

    precision, recall, _ = precision_recall_curve(y, model[1]['y_pred'])
    plt.plot(recall, precision)
    plt.title('PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
# plotting function
def plot_model(model):
    plt.figure(figsize=(9, 3))

    make_confusion_matrix(model)
    make_roc_curve(model)
    make_pr_curve(model)

    plt.tight_layout()
    plt.show()

# In[]:
model = xgb_1
plot_model(model)