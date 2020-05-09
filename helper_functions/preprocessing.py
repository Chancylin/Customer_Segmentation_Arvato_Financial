"""This module includes some basic functions to preprocess the data"""

import numpy as np
import matplotlib.pyplot as plt

def drop_columns(df, cols_to_drop):
    """Drops specific columns in a pandas dataframe"""
    #cols_to_drop = list(map(lambda x: "ALTER_KIND"+str(x), [1,2,3,4])) + ["EXTSEL992", "CAMEO_DEU_2015"]
    return df.drop(columns = cols_to_drop, axis=1)
    
def remove_row(df):
    """Removes rows with more than 80% missing value"""
    df["missing_value"] = df.isnull().sum(axis=1)/df.shape[1]
    # use 0.6 as threshold before, maybe too much information is lost
    df = df[df["missing_value"] < 0.8]
    return df.drop(columns = ["missing_value"], axis=1)

def miss_data_impu(df):
    """Impute the missing data, with -1"""
    df.fillna(-1, inplace=True)
    #
    df.loc[df["CAMEO_DEUG_2015"] == "X", "CAMEO_DEUG_2015"] = -1
    #
    df.loc[df["CAMEO_INTL_2015"] == "XX", "CAMEO_INTL_2015"] = -1
    # convert them to int
    return df.astype({"KK_KUNDENTYP": 'float', "CAMEO_DEUG_2015": 'float', "CAMEO_INTL_2015": 'float'}).\
    astype({"KK_KUNDENTYP": 'int', "CAMEO_DEUG_2015": 'int', "CAMEO_INTL_2015": 'int'})
    
def miss_data_check_row(df):
    """plot the histogram of missing value percentage for rows"""
    # all a column to indicate the number of missing values
    df["missing_value"] = df.isnull().sum(axis=1)/df.shape[1]
    
    #print('row with null values:' )
    #print('\n'.join('{:6.2f}'.format(val) for val in col_nan.values))
    print("-"*10)
    print('Row null values histogram:\n')
    ax = df["missing_value"].hist(bins=np.arange(11)*0.1)
    ax.set_xlabel("Missing value percentage")
    ax.set_ylabel("Numbers of rows")
    df.drop(columns = ["missing_value"], axis=1, inplace=True)
    
def miss_data_check_col(df):
    """plot the histogram of missing value percentage for rows

    Args:
        df: pandas dataframe

    Returns:
        a pandas series with information of missing value distribution acroos columns

    """
    # pandas.core.series.Series
    col_nan = df.isnull().sum()/df.shape[0]
    print('Columns with null values:' )
    #print('\n'.join('{:6.2f}'.format(val) for val in col_nan.values))
    print("-"*10)
    print('Columns null values histogram:\n')
    col_nan.hist(bins=np.arange(11)*0.1)
    return col_nan

def col_hist_plot(df, col_name, **kw):
    """plot the histogram of unique values for a specific column"""
    #ax=plt.subplots(figsize=(6,3))
    #print("unique valus in colmn: ", customers["KK_KUNDENTYP"].unique())
    
    # get data by column_name and display a histogram
    if "bins" in kw:
        df[col_name].hist(bins = bins)
    else:
        df[col_name].hist()
        
    title = "Histogram of " + col_name
    plt.title(title, fontsize=12)
    plt.show()

