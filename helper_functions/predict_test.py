"""This module includes the handy functions to make the prediction on test set easier"""

def transform_test_data(mailout_test, max_abs_scaler_customer, pca_all):
    """applies the preprocessing and transformation on the test dataset"""

    # same preprocess

    # 1. ===============
    cols_to_drop = list(map(lambda x: "ALTER_KIND"+str(x), [1,2,3,4])) \
    + ["EXTSEL992", "CAMEO_DEU_2015"] \
    + ["D19_LETZTER_KAUF_BRANCHE", "EINGEFUEGT_AM", "OST_WEST_KZ"]

    mailout_test_clean = drop_columns(mailout_test, cols_to_drop)
    #mailout_test_clean = remove_row(mailout_test_clean)
    mailout_test_clean = miss_data_impu(mailout_test_clean)

    #mailout_test = None

    print(mailout_test_clean.shape)

    cols_X = [col for col in mailout_test_clean.columns.tolist()]

    mailout_test_clean_X = mailout_test_clean[cols_X]

    print(mailout_test_clean_X.shape)

    # 2. ===============
    # rescale
    mailout_test_X_scaled = max_abs_scaler_customer.transform(mailout_test_clean_X)
    # 3. ===============
    # PCA
    mailout_test_X_pca = pca_all.transform(mailout_test_X_scaled)
    
    return mailout_test_X_pca


def predict_test(clf, test_set_pca, testset_prediction, predict_result_file):
    """makes the prediction on the test dataset with the trained classifier"""

    predict_y_proba_test = clf.predict_proba(test_set_pca)
    testset_prediction["RESPONSE"] = predict_y_proba_test[:,1]
    
    testset_prediction.to_csv(predict_result_file, index=False)
    print("predict results saved in {0}".format(predict_result_file))
    
    print("how many positive predictions:", testset_prediction[testset_prediction["RESPONSE"]>=0.5].shape)
    
