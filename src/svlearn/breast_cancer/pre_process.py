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
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define preprocessing function
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the breast cancer data

    Args:
        data (pd.DataFrame): incoming data

    Returns:
        pd.DataFrame: processed data
    """

    # Convert 'diagnosis' column to numerical values
    data['target'] = data['diagnosis'].map({'B': 0, 'M': 1})
    data = data.drop(['diagnosis'], axis=1)

    # Scale numerical columns
    feature_columns = data.columns.difference(['target'])
    scaler = StandardScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns]) 
    
    return data

#------------------------------PRE-PROCESS DATA---------------------------------------------------# 