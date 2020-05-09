import pandas as pd
import gc

def upsample_for_balance(mailout_train_X_pca, mailout_train_clean_Y, seed = 1234):
    
    mailout_train_data_clean = pd.concat([mailout_train_X_pca,mailout_train_clean_Y], axis=1)
    
    #upsample the positive instances

    mailout_train_data_neg = mailout_train_data_clean[mailout_train_data_clean["RESPONSE"] == 0]
    mailout_train_data_pos = mailout_train_data_clean[mailout_train_data_clean["RESPONSE"] == 1]

    print("Before upsample: negative v.s. positive")
    print(mailout_train_data_neg.shape, mailout_train_data_pos.shape)

    mailout_train_data_pos_up_sample = mailout_train_data_pos.sample(frac=8.0, random_state=seed, replace=True)

    print("After upsample: negative v.s. positive")
    print(mailout_train_data_neg.shape, mailout_train_data_pos_up_sample.shape)

    mailout_train_data_clean_new = pd.concat([mailout_train_data_neg, mailout_train_data_pos_up_sample], axis=0)

    mailout_train_data_clean_new = mailout_train_data_clean_new.sample(frac=1,  random_state=seed, replace=False)

    mailout_train_data_neg = mailout_train_data_pos = mailout_train_data_pos_up_sample = mailout_train_data_clean = None
    gc.collect()

    cols_X = [x for x in mailout_train_data_clean_new.columns.tolist() if x not in ["RESPONSE"]]

    mailout_train_X_pca_up_sample = mailout_train_data_clean_new[cols_X]

    mailout_train_Y_up_sample = mailout_train_data_clean_new[["RESPONSE"]]

    return mailout_train_X_pca_up_sample, mailout_train_Y_up_sample