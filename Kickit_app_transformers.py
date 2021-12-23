from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class Kickit_dataframe_preprocessor (BaseEstimator, TransformerMixin):
    """ 
    This is transformer class that inherits from the BaseEstimator,
    and TransformerMixin of Sklearn library. 
    The class have been modified to provide to instances 
    all the preprocessing capacity specific to the  kick dataset
    It is compatible with sklearn Pipeline and provide insights as messages when ..... 
    
    parameters =             X (required)  : the dataframe
                  choosen_axis (optional)  : a subset of features of the previous dataframe
   
   mother class = Sklearn.Base.BaseEstimator , TransformerMixin
    """

    def fit(self, X, choosen_axis = None, y=None):
        return  self  
        
    
    def transform(self, X, choosen_axis = None , y=None):

        # replacing all the '?' by null values
        X= X.replace(['?'], np.nan )
        
        # selecting the features to keep
        if choosen_axis == None :
            pass
        else : 
            X = X[choosen_axis]
        return  X
    
    
    
class Kickit_dataframe_balancer (BaseEstimator, TransformerMixin):
    """ 
    This is transformer class that inherits from the BaseEstimator,
    and TransformerMixin of Sklearn library. 
    The class have been modified to provide to instances 
    capacity  to balance the  kick dataset
    It is compatible with sklearn Pipeline and provide insights as messages when ..... 
    parameters =             X (required)  : the dataframe
    mother class = Sklearn.Base.BaseEstimator , TransformerMixin
    """
    def fit(self, X, y=None):
        return  self  

    def transform(self, X, y=None):
        # Count how many targets are 1 (meaning that the car is a kick)
        num_one_targets = X['IsBadBuy'].sum()
        # Set a counter for targets that are 0 
        zero_targets_counter = 0
        row_list2 =[]
        # We want to create a "balanced" dataset, so we will have to remove some input/target pairs

        for i in X.index.astype('int64'):
            if (X.loc[i,'IsBadBuy']) == 0:
                zero_targets_counter += 1
                if zero_targets_counter > num_one_targets:
                    row_list2.append(i)
        X = X.drop(index=row_list2,axis= 0)
        #resetting the index
        X.index = range(len(X.index))
        return  X
    
    
    
class Kickit_frequency_encoder (BaseEstimator, TransformerMixin):
    def fit(self, X, catCols = ["Model", "Make" ,"Transmission"],  y=None):
        return  self  
        
    def transform(self, X, catCols = ["Model", "Make" ,"Transmission"], y=None):   
        for col in X[catCols]:
            df_frequency_map = X[col].value_counts().to_dict()
            X[col] = X[col].map(df_frequency_map)
        return  X  
    
    
    
    
    
class Kickit_weight_of_evidence_encoder (BaseEstimator, TransformerMixin):
    def fit(self, X, catCols = ["Model", "Make" ,"Transmission"],  y=None):
        return  self  
        
    def transform(self, X, catCols = ["Model", "Make" ,"Transmission"], y=None):   
        for col in X[catCols]:
            df_frequency_map = X[col].value_counts().to_dict()
            X[col] = X[col].map(df_frequency_map)
        return  X  
    