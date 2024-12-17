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


def ingest_airbnb_data() -> pd.DataFrame:
    """Return the pandas dataframe for AirBnb data

    Returns:
        pd.DataFrame: Airbnb dataframe
    """
    return pd.read_csv("https://raw.githubusercontent.com/supportvectors/data-wrangling-datasets/main/Airbnb_Open_Data.csv", low_memory=False) 