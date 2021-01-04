from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from sklearn.svm import SVC

def rand_sampling(x, var_hist):
    if np.isnan(x):
        rand_idx = np.random.choice(len(var_hist))
        x = var_hist.iloc[rand_idx][0]
    return x

def blank_2_num_samp(T1D_features):
    """

    :param T1D_features: Pandas series of T1D features
    :return: A pandas dataframe containing the "clean" features
    """
    T1D_features = T1D_features.replace(r'^\s*$', np.nan, regex=True) #replace blanks with NaN
    T1D_features = T1D_features.replace('No', 0, regex=True) #replace 'No' with 0
    T1D_features = T1D_features.replace('Yes', 1.0, regex=True) #replace 'Yes' with 1
    T1D_features = T1D_features.replace('Negative', 0, regex=True) #replace 'Negative' with 0
    T1D_features = T1D_features.replace('Positive', 1.0, regex=True) #replace 'Positive' with 1
    T1D_features = T1D_features.replace('Male', 2.0, regex=True) #replace 'Male' with 2
    T1D_features = T1D_features.replace('Female', 3.0, regex=True) #replace 'Female' with 3

    T1D_features_clean = pd.DataFrame()
    for key in T1D_features.keys():
        feat=[key]
        T1D_no_NaN=T1D_features[feat].loc[:].dropna() #no Nan's
        new = T1D_features[feat].applymap(lambda x: rand_sampling(x, T1D_no_NaN))
        new1 = pd.DataFrame.from_dict(new)
        T1D_features_clean[feat] = new1 
    return T1D_features_clean

def train_test(x_train, x_test, y_train, y_test, model):
    clf = model.fit(x_train, y_train)
    y_pred_val = clf.predict(x_test)
    ROC_log = roc_auc_score(y_test, y_pred_val)
    return ROC_log

def k_fold_CV(X, Y, linear, penalty, kernel, lmbda, n_splits, solver):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=10)
    m_x_train, m_x_val, m_y_train, m_y_val = train_test_split(X, Y, test_size =0.2, random_state = 5, stratify=Y)
    ROC_lamb = []
    for idx, lmb in enumerate(lmbda):
        C = 1/lmb
        if linear:
            model = LogisticRegression(random_state=5, penalty=penalty, C = C, max_iter=1000000, solver=solver)

        else:
            model = SVC(random_state=5, C = C, kernel = kernel, degree=3)
        
        print(model)

        with tqdm(total=n_splits, file=sys.stdout, position=0, leave=True) as pbar:
            h = 0 # index per split per lambda
            ROC = []
            for train_index, val_index in skf.split(m_x_train, m_y_train):
                clf = []
                pbar.set_description('%d/%d lambda values, processed folds' % ((1 + idx), len(lmbda)))
                pbar.update()
                #--------------------------Impelment your code here:-------------------------------------
                x_train_fold = m_x_train[train_index,:]
                y_train_fold = m_y_train[train_index]
                x_test_fold = m_x_train[val_index,:]
                y_test_fold = m_y_train[val_index]
                ROC = train_test(x_train_fold, x_test_fold, y_train_fold, y_test_fold, model) 
                #----------------------------------------------------------------------------------------
                h += 1
            ROC_lamb.append(np.mean(ROC))
            model = []
    return ROC_lamb

def best_estimator(x, y, model, n_splits):
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    
    if model == 'linear':
        lmbda = np.linspace(1e-5, 1, num=10)
        skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
#C = 1/best_lambda

        solver = 'liblinear'
        log_reg = LogisticRegression(random_state=5, C = 1/lmbda, max_iter=1000000, solver=solver)
        pipe = Pipeline(steps=[('logistic', log_reg)])
        clf = GridSearchCV(estimator=pipe, param_grid={'logistic__C': 1/lmbda, 'logistic__penalty': ['l1', 'l2']},
                           scoring=['accuracy','f1','precision','recall','roc_auc'], cv=skf,
                           refit='roc_auc', verbose=3, return_train_score=True)
        clf.fit(x, y)
        lin_best = clf.best_params_
        return lin_best
    
    if model == 'svm':
        lmbda = np.linspace(1e-5, 1, num=10)
        C = 1/lmbda
        svc = SVC(random_state=5, C = C, probability=True)
        skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)

        pipe = Pipeline(steps=[ ('svm', svc)])
        svm_nonlin = GridSearchCV(estimator=pipe,
                     param_grid={'svm__kernel':['rbf','poly'], 'svm__C':C, 'svm__degree':[3]},
                     scoring=['accuracy','f1','precision','recall','roc_auc'], 
                     cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
        svm_nonlin.fit(x, y)
        best_svm_nonlin = svm_nonlin.best_params_
    
        return best_svm_nonlin
