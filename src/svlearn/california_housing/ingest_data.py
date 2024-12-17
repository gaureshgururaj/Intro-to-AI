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


def ingest_cal_housing_data() -> pd.DataFrame:
    """Get the pandas dataframe containing the california housing data

    Returns:
        pd.DataFrame: The Cal housing prices dataframe
    """
    return pd.read_csv("https://raw.githubusercontent.com/supportvectors/ml-100/master/housing.csv")