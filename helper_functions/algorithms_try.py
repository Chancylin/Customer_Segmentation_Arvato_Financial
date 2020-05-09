import xgboost as xgb

def xgb_train(X_train, y_train,
              X_valid, y_valid,
              X_total, y_total,
              param_dist, early_stopping_rounds=15, train_whole_data=False):

    xgb_clf = xgb.XGBClassifier(**param_dist)
    
    if not train_whole_data:

        xgb_clf.fit(X_train.values, y_train.values.ravel(),
                eval_set=[(X_train.values, y_train.values.ravel()), (X_valid.values, y_valid.values.ravel())],
                eval_metric='auc',
                early_stopping_rounds=early_stopping_rounds,
                verbose=False)

        print(xgb_clf.best_score, "\n", xgb_clf.best_iteration, "\n",  xgb_clf.best_ntree_limit)
        
    else:
        xgb_clf.fit(X_total.values, y_total.values.ravel(),
            eval_set=[(X_valid.values, y_valid.values.ravel())],
            eval_metric='auc',
            verbose=False)
        
    return xgb_clf


def xgb_train_generate_stacking(X_total, y_total, param_dist):
    
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=3, shuffle=False)
    
    X_total_new = X_total.copy()
    X_total_new["xgb_output"] = -1
    
    i = 1
    for train_index, test_index in skf.split(X_total, y_total):
        
        print("round: ", i )
        print(train_index)
        X_train, X_test = X_total.iloc[train_index,:], X_total.iloc[test_index,:]
        y_train, y_test = y_total.iloc[train_index,:], y_total.iloc[test_index,:]
        
        xgb_clf = xgb.XGBClassifier(**param_dist)

        xgb_clf.fit(X_train.values, y_train.values.ravel(), verbose=False)
        
        predict_y_proba_test = xgb_clf.predict_proba(X_test.values)
        X_total_new["xgb_output"].iloc[test_index] = predict_y_proba_test
        i = i + 1
        
    return X_total_new

# scoring function

from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

def get_accuracy_auc(model, X, ground_y):
    predict_y = model.predict(X)
    predict_y_proba = model.predict_proba(X)
    #
    accuracy_score = (predict_y == ground_y).sum()/len(predict_y)
    # ROC-AUC
    auc_score = roc_auc_score(ground_y, predict_y_proba[:,1])
    # PR_AUC
    pr_auc_score = average_precision_score(ground_y, predict_y_proba[:,1])
    #
    #print("accuracy score: {0:6.4f}, ROC-AUC score: {1:6.4f}, PR-AUC score: {2:6.4f}".\
    #  format(accuracy_score, auc_score, pr_auc_score))
    
    # Confusion matrix:
    #print("Confusion matrix [[TN, FP]\n[FN, TP]]:\n", confusion_matrix(ground_y, predict_y))
    
    return accuracy_score, auc_score, pr_auc_score

        
        