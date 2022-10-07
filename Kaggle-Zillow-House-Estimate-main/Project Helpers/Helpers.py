import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import missingno as msno
from sklearn.feature_selection import chi2
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor


"""
A set of helper functions for the Kaggle House Price Prediction Competition.
"""

def read_dataframe(path, analyze = False):
    '''
    Reads from a CSV file and generates descriptive stats of a dataframe.
    
    Args:
        df (dataframe): the dataframe to be analyzed
    
    Returns:
        None
    '''
    df = pd.read_csv(path)
    
    if analyze:
        print("\n\n'\033[1m'")
        print("*"*30)
        print("Data Report")
        print("*"*30)

        print("Number of rows::",df.shape[0])
        print("Number of columns::",df.shape[1])
        print("\n")

        print("Column Names::",df.columns.values.tolist())
        print("\n")

        print("Column Data Types::\n",df.dtypes)
        print("\n")

        print("Columns with Missing Values::",df.columns[df.isnull().any()].tolist())
        print("\n")

        print("Number of rows with Missing Values::",len(pd.isnull(df).any(1).to_numpy().nonzero()[0].tolist()))
        print("\n")

        print("Sample Indices with missing data::",pd.isnull(df).any(1).to_numpy().nonzero()[0].tolist()[0:5])
        print("\n")

        print("General Stats::")
        print(df.info())
        print("\n")

        print('Summery Stats(Numerical): ')
        print(df.describe())
        print('\n')


        print('Summery Stats(Objects): ')
        print(df.astype('object').describe().transpose())
        print('\n')

        print("Dataframe Sample Rows::")
        display(df.head(5))

        print('Visualizing Missing Values...')
        msno.matrix(df,labels=True)
        plt.show()
    return df
    

def extract_cat_features(df):
    '''
    Extract categorical features from a Dataframe.
    In this case, categorical features are of type object.
    
    Args: 
        df(Dataframe)
    
    Returns: 
        num_features(Dataframe)
    '''
    # Extract columns from the Dataframe
    column_list = df.columns.tolist()
    cat_features = []
    
    # Iterate through the column list and extract categorical features
    for column in column_list:
        if df[column].dtype =='object':
            cat_features.append(column)
    return cat_features

def extract_num_features(df):
    '''
    Extract numerical features from a Dataframe.
    In this case, Numerical features are of type int64 or float64.
    
    Args: 
        df(Dataframe)
    
    Returns: 
        num_features(List)
    '''
    # Extract columns from the Dataframe
    column_list = df.columns.tolist()
    num_features = []
    
    # Iterate through the column list and extract numerical features
    for column in column_list:
        if df[column].dtype =='int64':
            num_features.append(column)
            
        if df[column].dtype == 'float64':
                num_features.append(column)
            
    return num_features

def find_high_correlation(df):
    '''
    Find highly correlated features.
    High Correlation is defined as having an absolute value higher than 0.8.
    
    Args: 
        df(Dataframe)
    
    Returns: 
        Dataframe
    '''
    # Extract numerical features and create a baseline correlation matrix
    num_features = extract_num_features(df)
    correlation = abs(df[num_features].corr())
    correlation_matrix = df[num_features].corr()
    
    # Filter features that dont have an absolute correlation higher than 0.75
    high_correlation_features = df[num_features].iloc[:,np.where(correlation_matrix[(correlation > 0.8) & (correlation < 1)].any())[0]]
    return high_correlation_features
    
            
def correlation_heatmap(df):
    '''
    Plot a seaborn heatmap and print correlation.
    
    Args: 
        df(Dataframe)
    
    Returns: 
        Correlation Heatmap
    '''
    high_correlation_features = find_high_correlation(df)
    
    # Plot the correlation heatmap and Dataframe
    sns.heatmap(high_correlation_features.corr())
    plt.title('Heat Map of Highly Correlated Features', size = 18)
    plt.show()
    return high_correlation_features.corr()  

def get_pairs(df):
    """
    Grabs the combinations of pairs of column names
    
    Input
    -----
    df: {pandas DataFrame array}
        input data

    Output
    ------
    pairs: {set}
           a set whose elements are tuples which have all pairs of
           columns in a DataFrame. 
    """
    pairs = set()
    cols = df.columns
    for i in range(len(df.columns)):
        for j in range(i + 1):
            pairs.add((cols[i], cols[j]))
    return pairs

def top_abs_corrs(df, threshold=.8):
    """
    Returns a dataframe with pairwise correlations between features that
    meet the creiteria of absolute value of correlation >= threshold
    
    Input
    -----
    df: {pandas DataFrame array}
        input data
    threshold:{float}
        parameter used to as the threshold

    Output
    ------
    top_corrs: {DataFrame}
           data with selected features
    """
    num_features = extract_num_features(df)
    corrs = df[num_features].corr().abs().unstack()
    pairs = get_pairs(df[num_features])
    top_corrs = (corrs.drop(labels=pairs)
                     .reset_index()
                     .rename(columns={0:'correlation', 
                                      'level_0':'feature_1', 
                                      'level_1':'feature_2'})
                     .sort_values(by='correlation', ascending=False))
    top_corrs = top_corrs[top_corrs['correlation'] >= threshold].reset_index(drop=True)
    return top_corrs

def find_duplicated_columns(df):
    """
    This function finds duplicated columns in a 
    pandas DataFrame and returns a dictionary with 
    keys as column names and value as list of its 
    duplicated column names.
    
    It also returns a list of just duplicated columns

    Input
    -----
    df: {pandas DataFrame}

    Output
    ------
    duplicated_features: {dictionary}
    dictionary with all columns as key and list of duplicated columns as value. 
    List is empty if that column doesn't have duplicates
    
    duplicate_of: {list}
    List of all columns that are duplicates of some other column
    """
    duplicated_features = {}
    duplicate_of = []
    for i in range(0, len(df.columns)): 
        feat_1 = df.columns[i]
        if feat_1 not in duplicate_of:
            duplicated_features[feat_1] = []
            for feat_2 in df.columns[i + 1:]:
                if df[feat_1].equals(df[feat_2]):
                    duplicated_features[feat_1].append(feat_2)
                    duplicate_of.append(feat_2)
    return duplicated_features, duplicate_of

def x_y_split(df):
    y = df.loc[:,'SalePrice']
    x = df.drop('SalePrice', axis = 1)
    return x, y

def plot_feature(df, feature_name):
    '''
    Plot the feature's distribution(Histogram) and its correlation with SalePrice(Scatter plot)
    
    Args:
        df(DataFrame)
        feature_name(str)
    
    Returns:
        Histogram and scatter plot
    '''
    # if the feature is continuous 
    if len(df[feature_name].value_counts()) >= 100:
    
        fig = plt.figure(figsize = (15,4))
        sns.histplot(df[feature_name], kde = True)
        plt.xlabel(feature_name, size = 16)
        plt.ylabel('Count', size = 16)
        plt.title(feature_name + ' Distribution', size = 16)
        plt.show()

        fig = plt.figure(figsize = (15,4))
        sns.scatterplot(data = df, x = feature_name, y = 'SalePrice')
        plt.xlabel(feature_name, size = 16)
        plt.ylabel('SalePrice', size = 16)
        plt.title(feature_name +' vs Sale Price', size = 16)
        plt.show()
        
        correlation_to_price = df[feature_name].corr(other=df['SalePrice'])
        print('\nThe Correlation between ' + feature_name + ' and SalePrice is: ', round(correlation_to_price,4))

        fig = plt.figure(figsize = (15,4))
        sns.boxplot(data = df, y = feature_name, x = 'OverallQual')
        plt.ylabel(feature_name, size = 16)
        plt.xlabel('OverallQual', size = 16)
        plt.title(feature_name +' vs Overall Quality', size = 16)
        plt.show()

        correlation_to_quality = df[feature_name].corr(other=df['OverallQual'])
        print('\nThe Correlation between ' + feature_name + ' and Overall Quality is: ', round(correlation_to_quality,4))
        
    # if the feature is discrete 
    if len(df[feature_name].value_counts()) < 100:
        fig = plt.figure(figsize = (15,4))
        sns.countplot(x = df[feature_name])
        plt.xlabel(feature_name, size = 16)
        plt.ylabel('Count', size = 16)
        plt.title(feature_name + ' Distribution', size = 16)
        plt.show()

        fig = plt.figure(figsize = (15,4))
        sns.boxplot(data = df, x = feature_name, y = 'SalePrice')
        plt.xlabel(feature_name, size = 16)
        plt.ylabel('SalePrice', size = 16)
        plt.title(feature_name +' vs Sale Price', size = 16)
        plt.show()
        
        correlation_to_price = df[feature_name].corr(other=df['SalePrice'])
        print('\nThe Correlation between ' + feature_name + ' and SalePrice is: ', round(correlation_to_price,4))

        fig = plt.figure(figsize = (15,4))
        sns.barplot(data = df, y = feature_name, x = 'OverallQual')
        plt.ylabel(feature_name, size = 16)
        plt.xlabel('OverallQual', size = 16)
        plt.title(feature_name +' vs Overall Quality', size = 16)
        plt.show()

        correlation_to_quality = df[feature_name].corr(other=df['OverallQual'])
        print('\nThe Correlation between ' + feature_name + ' and Overall Quality is: ', round(correlation_to_quality,4))

    
def missing_nonMissing_split(df, feature_name):
    
    # Spliting the data to two Dataframes: one where LotFrontage is missing, the other where it is not 
    missing_feature_df = df[df[feature_name].isna()]
    not_missing_feature_df = df[df[feature_name].isna() == False]
    return missing_feature_df, not_missing_feature_df
    
    
    
    
def analyze_feature(df, feature_name):
    print("\n\n")
    print("*"*30)
    print("Feature Report")
    print("*"*30)
    
    plot_feature(df, feature_name)
    
    missing_feature_df, not_missing_feature_df = missing_nonMissing_split(df, feature_name)
    
    fig = plt.figure(figsize = (15,4))
    ax = sns.kdeplot(missing_feature_df['SalePrice'], color = 'b', shade = True, label = 'Missing')
    ax = sns.kdeplot(not_missing_feature_df['SalePrice'], color = 'r', shade = True, label = 'Not Missing')

    plt.title('Sale Price Distribution - Missing and Not Missing ' + feature_name, size = 16)
    plt.legend(['Missing ' + feature_name,'Not Missing ' + feature_name])
    plt.show()

    fig = plt.figure(figsize = (15,4))
    ax = sns.kdeplot(missing_feature_df['OverallQual'], color = 'b', shade = True, label = 'Missing')
    ax = sns.kdeplot(not_missing_feature_df['OverallQual'], color = 'r', shade = True, label = 'Not Missing')

    plt.title('Overall Quality Distribution - Missing and Not Missing ' + feature_name, size = 16)
    plt.legend(['Missing ' + feature_name,'Not Missing ' + feature_name])
    plt.show()

    amount_of_missing_values = df[feature_name].isna().sum()
    print('There are',amount_of_missing_values, ' Missing Values in ' + feature_name)
    
    avg_SalePrice_overall =  round(not_missing_feature_df['SalePrice'].mean(),2)
    std_SalePrice_overall =  round(not_missing_feature_df['SalePrice'].std(),2)
    print('The Average House with ' + feature_name + ' costs {}$ with a Standard Deviation of {}'.format(avg_SalePrice_overall,std_SalePrice_overall))
    
    avg_SalePrice_missing_feature_name = round(missing_feature_df['SalePrice'].mean(),2)
    std_SalePrice_missing_feature_name = round(missing_feature_df['SalePrice'].std(),2)
    print('The Average House with missing ' + feature_name + ' costs {}$ with a Standard Deviation of {}$'.format(avg_SalePrice_missing_feature_name,std_SalePrice_missing_feature_name))
    
    avg_OverallQual_overall =  round(not_missing_feature_df['OverallQual'].mean(),2)
    std_OverallQual_overall =  round(not_missing_feature_df['OverallQual'].std(),2)
    print('The Average House with ' + feature_name + ' has an overall quality of {} with a Standard Deviation of {}'.format(avg_OverallQual_overall,std_OverallQual_overall))
    
    avg_OverallQual_missing_feature_name = round(missing_feature_df['OverallQual'].mean(),2)
    std_OverallQual_missing_feature_name =  round(missing_feature_df['OverallQual'].std(),2)
    print('The Average House with missing ' + feature_name + ' has an overall quality of {} with a Standard Deviation of {}'.format(avg_OverallQual_missing_feature_name,std_OverallQual_missing_feature_name))

    print('\nSummary Table')
    summary = pd.DataFrame([avg_SalePrice_overall,std_SalePrice_overall], columns = ['Sale Price - Non Missing ' + feature_name], index = ['Average','Standard Deviation'])
    summary['Sale Price - Missing ' + feature_name] = avg_SalePrice_missing_feature_name, std_SalePrice_missing_feature_name
    summary['Overall Quality - Non Missing ' + feature_name] = avg_OverallQual_overall, std_OverallQual_overall
    summary['Overall Quality - Missing ' + feature_name] = avg_OverallQual_missing_feature_name, std_OverallQual_missing_feature_name
    return summary

def f_score(df):
    """
    This function implements the anova f_value feature selection (existing method for classification in scikit-learn),
    where f_score = sum((ni/(c-1))*(mean_i - mean)^2)/((1/(n - c))*sum((ni-1)*std_i^2))

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y : {numpy array},shape (n_samples,)
        input class labels

    Output
    ------
    F: {numpy array}, shape (n_features,)
        f-score for each feature
    """
    x, y = x_y_split(df)
    
    num_features = extract_num_features(df)
    cat_features = extract_cat_features(df)
    
    F_numeric, pval = f_regression(x[num_features], y)
    F_category, pval = f_classif(x[cat_features], y)
    
    return F_numeric, F_category


def feature_ranking(F):
    """
    Rank features in descending order according to f-score, the higher the f-score, the more important the feature is
    """
    idx = np.argsort(F)
    return idx[::-1]


def minus_1_imputer(df, features_to_impute):
    imputer = SimpleImputer(missing_values = np.nan, strategy='constant', fill_value = -1)
    df.loc[:,features_to_impute] = imputer.fit_transform(df.loc[:,features_to_impute])
    return df

def most_frequent_imputer(df, features_to_impute):
    imputer = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
    df.loc[:,features_to_impute] = imputer.fit_transform(df.loc[:,features_to_impute])
    return df

def variance_plot(numerical_df):
    
    threshold_list  = [0 ,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    number_of_features_dropped = []
    
    for i in threshold_list:
        selector = VarianceThreshold(threshold=i)
        features_to_keep = selector.fit_transform(numerical_df)
        number_of_features_dropped.append(numerical_df.shape[1] - features_to_keep.shape[1])
        
    plt.plot(threshold_list, number_of_features_dropped, c = 'r')
    plt.xlabel('Variance Threshold', size = 16)
    plt.ylabel('Amount of Features to Drop', size = 16)
    plt.show()
    
    
    
def constantish_features(df, threshold):
    """
    This function finds constant-ish features in a 
    pandas DataFrame and returns a list with the column names
    
    It also returns a list of just duplicated columns

    Input
    -----
    df: {pandas DataFrame}
    the DataFrame you want to find constant-ish features in
    
    threshold: {float}
    all features with value greater than threshold are considered
    to be constanti

    Output
    ------
    constant_ish_feat: {list}
    List of all columns that are constant-ish
    """
    constant_ish_feat = []
    for feature in df.columns:
        prop_of_col = (df[feature].value_counts() / np.float(
            len(df))).sort_values(ascending=False).values[0]
        if prop_of_col >= threshold:
            constant_ish_feat.append(feature)
    return constant_ish_feat

def find_duplicated_columns(df):
    """
    This function finds duplicated columns in a 
    pandas DataFrame and returns a dictionary with 
    keys as column names and value as list of its 
    duplicated column names.
    
    It also returns a list of just duplicated columns

    Input
    -----
    df: {pandas DataFrame}

    Output
    ------
    duplicated_features: {dictionary}
    dictionary with all columns as key and list of duplicated columns as value. 
    List is empty if that column doesn't have duplicates
    
    duplicate_of: {list}
    List of all columns that are duplicates of some other column
    """
    duplicated_features = {}
    duplicate_of = []
    for i in range(0, len(df.columns)): 
        feat_1 = df.columns[i]
        if feat_1 not in duplicate_of:
            duplicated_features[feat_1] = []
            for feat_2 in df.columns[i + 1:]:
                if df[feat_1].equals(df[feat_2]):
                    duplicated_features[feat_1].append(feat_2)
                    duplicate_of.append(feat_2)
    if len(duplicate_of) == 0:
        print('No Duplicate Features')
    else:
        
        return duplicated_features, duplicate_of
    
def who_to_drop(corr_df, numerical_df, target_variable):

    features_to_drop_pairs = []
    
    for i in range(corr_df.shape[0]):
        feature1 = corr_df.iloc[i][0]
        feature2 = corr_df.iloc[i][1]
        
        x1 = np.array(numerical_df.loc[:,feature1].fillna(0)).reshape(-1,1)
        x2 = np.array(numerical_df.loc[:,feature2].fillna(0)).reshape(-1,1)
        model =  RandomForestRegressor()
        
        model.fit(x1, target_variable)
        predictions1 = model.predict(x1)
        r_squared_x1 = r2_score(target_variable, predictions1)
        
        model.fit(x2, target_variable)
        predictions2 = model.predict(x2)
        r_squared_x2 = r2_score(target_variable, predictions2)
        
        if r_squared_x1 < r_squared_x2:
            features_to_drop_pairs.append(feature1)
        else:
            features_to_drop_pairs.append(feature2)
            
    return features_to_drop_pairs


class ChiSquare:
    from Helpers import extract_cat_features
    
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.colX = None
        
    
    def TestIndependence(self, colX, colY, alpha=0.1):
        self.colX=colX
        self.colY = colY
        X = self.df.loc[:, self.colX]
        Y = self.df.loc[:, self.colY]
        contingency_table = pd.crosstab(Y,X, margins=True) 
        dfObserved = contingency_table.iloc[:-1,:-1].values
        chi2, p, dof, expected = stats.chi2_contingency(dfObserved, correction=False)
        self.p = p
        self.chi2 = chi2
        return colX, p
    
    def features_to_drop(self, colY, alpha=0.05):
        
        cat_features = extract_cat_features(self.df)
        self.colY = colY
        features_to_drop = []
        for feature in cat_features:
            if self.TestIndependence(colX = feature, colY = 'SalePrice')[1] > alpha:
                features_to_drop.append(self.TestIndependence(colX = feature, colY = 'SalePrice'))
        
        # Function to sort the list of tuples by its second item
        def Sort_Tuple(tup):

            # getting length of list of tuples
            lst = len(tup)
            for i in range(0, lst):

                for j in range(0, lst-i-1):
                    if (tup[j][1] > tup[j + 1][1]):
                        temp = tup[j]
                        tup[j]= tup[j + 1]
                        tup[j + 1]= temp
            return tup
        
        features_to_drop = Sort_Tuple(features_to_drop)
        return features_to_drop
        
