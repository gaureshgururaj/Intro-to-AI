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
    return np.log(arr)
#-----------------------------LOG TRANSFORM-------------------------------------------------------#

#-----------------------------IDENTITY TRANSFORM--------------------------------------------------#
# return without any transformation
def identify_fn(x):
    return x
#-----------------------------IDENTITY TRANSFORM--------------------------------------------------#

#-----------------------------DROP NA ROWS--------------------------------------------------------#
# drop na rows from a dataframe
def drop_na(x):
    return x.dropna()
#-----------------------------DROP NA ROWS--------------------------------------------------------#

#-----------------------------PIPELINE CREATION---------------------------------------------------#
# create and return pipeline of transformations
def create_pipeline(categorical_cols, log_cols, numeric_cols) -> Pipeline:
    return Pipeline([
        ('column_transforms', ColumnTransformer
            (transformers=
                [
                    ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
                    ('log', Pipeline(steps=[
                        ('log_transform', FunctionTransformer(log_transform)),
                        ('identity_transform', FunctionTransformer(identify_fn))]), log_cols),
                    ('identity_transform', FunctionTransformer(identify_fn), numeric_cols)
                ]
            )
        )
    ])
#-----------------------------PIPELINE CREATION---------------------------------------------------#

#------------------------------PRE-PROCESS DATA---------------------------------------------------#  
## Preprocess the data
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame: 
    """Preprocess the AirBnB Data

    Args:
        data (pd.DataFrame): incoming data

    Returns:
        pd.DataFrame: preprocessed data
    """

    data.columns = data.columns.str.strip()
    
    # Drop irrelevant columns
    columns_to_drop = ['neighbourhood', 'house_rules', 'license', 'neighbourhood group']
    data = data.drop(columns=columns_to_drop)

    data = data.dropna()

    # Check for missing host_name and handle it
    if 'host_name' in data.columns:
        data['host_name'] = data['host_name'].fillna('Unknown').astype(str)
    else:
        raise ValueError("'host_name' column is missing from the data.")

    # Convert data types
    data['price'] = data['price'].str.replace('$', '').str.replace(',', '').astype(float)
    data['service fee'] = data['service fee'].str.replace('$', '').str.replace(',', '').astype(float)
    data['last review'] = pd.to_datetime(data['last review'], errors='coerce')

    # Feature engineering
    data['last_review_year'] = data['last review'].dt.year
    data['host_name_length'] = data['host_name'].str.len()
    
    # Define columns for transformations
    categorical_cols = ['room_type', 'cancellation_policy']
    log_cols = ['price', 'service fee']
    numeric_cols = ['latitude', 'longitude', 'minimum nights', 'number of reviews', 'reviews per month', 'calculated host listings count', 'availability 365', 'last_review_year', 'host_name_length']
    
    pipeline = create_pipeline(categorical_cols, log_cols, numeric_cols)
    
    x_transformed = pipeline.fit_transform(X=data)

    cat_transformed_cols = list(pipeline.named_steps['column_transforms'].named_transformers_['one_hot'].get_feature_names_out(categorical_cols))

    transformed_cols = cat_transformed_cols + log_cols + numeric_cols

    x_df = pd.DataFrame(x_transformed, columns=transformed_cols)
    x_df = x_df.rename(columns={'price': 'y_target'})

    return x_df

#------------------------------PRE-PROCESS DATA---------------------------------------------------#
