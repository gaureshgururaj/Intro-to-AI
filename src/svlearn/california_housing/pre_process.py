#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#  
#   Use is limited to the duration and purpose of the training at SupportVectors.
#  
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------
#
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

#-----------------------------LOG TRANSFORM-------------------------------------------------------#
# returns log of an array of positive floating numbers
def log_transform(arr):
    """Takes an array and performs its element wise logarithm

    Args:
        arr (_type_): A numpy array

    Returns:
        _type_: another numpy array containing the logarithm of elements of passed in array
    """
    return np.log(arr)
#-----------------------------LOG TRANSFORM-------------------------------------------------------#

#-----------------------------IDENTITY TRANSFORM--------------------------------------------------#
# return without any transformation
def identify_fn(x):
    """An identity transformation

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return x
#-----------------------------IDENTITY TRANSFORM--------------------------------------------------#

#-----------------------------DROP NA ROWS--------------------------------------------------------#
# drop na rows from a dataframe
def drop_na(x):
    """Drops any rows containing NA from the passed in pandas dataframe

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return x.dropna()
#-----------------------------DROP NA ROWS--------------------------------------------------------#

#-----------------------------PIPELINE CREATION---------------------------------------------------#
# create and return pipeline of transformations
def create_pipeline(categorical_cols, log_cols, numeric_cols) -> Pipeline:
    """Creates a preprocessing pipeline for the california housing dataset

    Args:
        categorical_cols (_type_): _description_
        log_cols (_type_): _description_
        numeric_cols (_type_): _description_

    Returns:
        Pipeline: _description_
    """
    return Pipeline([
        ('column_transforms', ColumnTransformer
            (transformers=
                [
                    ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
                    ('log', Pipeline(steps=[
                        ('log_transform', FunctionTransformer(log_transform)),
                        #('scaler', StandardScaler())]), log_cols), ## We can replace below line with this line
                        ('identity_transform', FunctionTransformer(identify_fn))]), log_cols),
                    #('standard_scale', StandardScaler(), numeric_cols) ## We can replace below line with this line
                    ('identity_transform', FunctionTransformer(identify_fn), numeric_cols)
                ]
            )
        )
    ])
#-----------------------------PIPELINE CREATION---------------------------------------------------#
#------------------------------PRE-PROCESS DATA---------------------------------------------------#  
## Preprocess the data 
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """The preprocessing function for a given experiment for the california housing dataset

    Args:
        data (pd.DataFrame): California Housing Dataset raw incoming
    Returns:
        pd.DataFrame: The preprocessed dataframe
    """
    data = data.dropna()
    
    # Define columns for the various transformations
    categorical_cols = ['ocean_proximity']
    log_cols = ['total_bedrooms', 'population', 'total_rooms', 'households', 'median_income', 'median_house_value']
    numeric_cols = ['housing_median_age', 'longitude', 'latitude']   
    
    pipeline = create_pipeline(categorical_cols, log_cols, numeric_cols)
    
    x_transformed = pipeline.fit_transform(X=data)
        
    # Get the column names after transformation for the categoricals
    cat_transformed_cols = list(pipeline.named_steps['column_transforms'].named_transformers_['one_hot'].get_feature_names_out(categorical_cols))

    # Combine all column names in the same order as transformations done
    transformed_cols = cat_transformed_cols + log_cols + numeric_cols

    # Convert the transformed array back to a DataFrame
    x_df = pd.DataFrame(x_transformed, columns=transformed_cols)
    x_df = x_df.rename(columns={'median_house_value': 'y_target'})
    
    return x_df
    
#------------------------------PRE-PROCESS DATA---------------------------------------------------# 