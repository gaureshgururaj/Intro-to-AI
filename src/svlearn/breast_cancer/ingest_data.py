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

def ingest_breast_cancer_data() -> pd.DataFrame:
    """Returns the breast cancer data

    Returns:
        pd.DataFrame: Breast Cancer dataframe
    """
    return pd.read_csv("https://raw.githubusercontent.com/Akshaya1601/ml-100/patch-1/wisconsin-diagnostic-breast-cancer.csv", low_memory=False) 