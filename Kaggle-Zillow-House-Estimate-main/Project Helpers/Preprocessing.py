import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from Helpers import extract_cat_features, extract_num_features, minus_1_imputer, most_frequent_imputer

class preprocessor:
    
    def __init__(self, cols_to_filter = None, cols_to_impute_minus_1 = None):
        
        self.cols_to_filter = cols_to_filter
        self.cols_to_impute_minus_1 = cols_to_impute_minus_1
        self.was_fit = False
        
    def fit(self, x, y=None):
        self.was_fit = True
        # filter
        x_new = x.drop(self.cols_to_filter, axis=1)
        
        # impute
        x_new = minus_1_imputer(x_new, self.cols_to_impute_minus_1)
        x_new = most_frequent_imputer(x_new)
             
        # dummy code
        self.categorical_features = extract_cat_features(x_new)
        dummied = pd.get_dummies(x_new, columns = self.categorical_features, dummy_na=True, drop_first = True)
        self.colnames = dummied.columns
        del dummied
        
        return self
        
    def transform(self, x, y=None):
        if not self.was_fit:
            raise Error("need to fit preprocessor first")
            
        # filter   
        x_new = x.drop(self.cols_to_filter, axis = 1)
        
        # impute
        x_new = minus_1_imputer(x_new, self.cols_to_impute_minus_1)
        x_new = most_frequent_imputer(x_new)
    
        # dummy code
        x_new = pd.get_dummies(x_new, columns=self.categorical_features, dummy_na=True, drop_first = True)
        newcols = set(self.colnames) - set(x_new.columns)
        for x in newcols:
            x_new[x] = 0
            
        x_new = x_new[self.colnames]  
        return x_new
    
    def fit_transform(self, x, y=None):
        """fit and transform wrapper method, used for sklearn pipeline"""

        return self.fit(x).transform(x)
        
        
        
        
        
        
        
        
 
