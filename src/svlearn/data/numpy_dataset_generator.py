#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#  
#   Use is limited to the duration and purpose of the training at SupportVectors.
#  
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

import numpy as np
from typing import Tuple, Optional

#  -------------------------------------------------------------------------------------------------


class NumpyDatasetGenerator:
    """
    A class to generate various datasets using NumPy.

    Attributes:
        num_samples (int): Number of samples in the dataset.
        num_features (int): Number of features for each sample.
    """
    #  -------------------------------------------------------------------------------------------------

    def __init__(self, num_samples: int, num_features: int):
        """
        Initializes the dataset generator with the number of samples and features.

        Args:
            num_samples (int): The number of samples to generate.
            num_features (int): The number of features in each sample. 

        """
        self.num_samples = num_samples
        self.num_features = num_features
    
    #  -------------------------------------------------------------------------------------------------

    def generate_sum_features_dataset(self, mean: float = 0, std: float = 1):
        """
        Generates a dataset with features drawn from a normal distribution
        with the specified mean and standard deviation. The target array y
        is the sum of the features in each sample.

        Args:
            mean (float): The mean of the normal distribution. Defaults to 0.
            standard_deviation (float): The standard deviation of the normal distribution. Defaults to 1.

        Returns:
            tuple: A tuple containing:
                - X (np.ndarray): The generated feature matrix of shape (num_samples, num_features).
                - y (np.ndarray): The target vector of shape (num_samples, ) where each entry is the
                  sum of the features of the corresponding sample.
        """
        # Generate features from a normal distribution with the specified mean and standard deviation
        X = np.random.normal(loc=mean, scale=std, size=(self.num_samples, self.num_features))
        
        # Calculate the sum of features for each sample
        y = X.sum(axis=1)
        
        return X, y

    #  -------------------------------------------------------------------------------------------------

    def generate_negative_number_dataset(self, mean: float = 6, std: float = 3) -> Tuple[np.ndarray , np.ndarray]:
        """
        Generates a dataset with features drawn from a normal distribution
        with the provided mean and standard deviation. The target array y indicates
        if a sample contains a negative number (1 if it does, 0 if it doesn't).

        Returns:
            tuple: A tuple containing:
                - X (np.ndarray): The generated feature matrix of shape (num_samples, num_features).
                - y (np.ndarray): The target vector of shape (num_samples, ) where each entry is 1 if the
                  corresponding sample contains a negative number, otherwise 0.
        Args:
            mean (float, optional): The mean of the normal distribution. Defaults to 6.
            std (float, optional): The standard deviation of the normal distribution. Defaults to 3.

        Returns:
            Tuple[np.ndarray , np.ndarray]: _description_
        """
        # Generate features from a normal distribution with mean and standard deviation 
        X = np.random.normal(loc=mean, scale=std, size=(self.num_samples, self.num_features))
        
        # Check if each sample contains a negative number
        y = (X < 0).any(axis=1).astype(int)

        
        return X , y
    
    #  -------------------------------------------------------------------------------------------------

    def generate_sum_greater_than_100_dataset(self, mean: Optional[float] = None, std: Optional[float] = None) -> Tuple[np.ndarray , np.ndarray]:
        """
        Generates a dataset with features drawn from a normal distribution
        with the specified mean and standard deviation. The target array y
        indicates if the sum of the features in a sample is greater than 100
        (1 if it is, 0 if it isn't).

        Args:
            mean (float): The mean of the normal distribution. Defaults to 0.
            std (float): The standard deviation of the normal distribution. Defaults to 1.

        Returns:
            tuple: A tuple containing:
                - X (np.ndarray): The generated feature matrix of shape (num_samples, num_features).
                - y (np.ndarray): The target vector of shape (num_samples, ) where each entry is 1 if the
                sum of the corresponding sample's features is greater than 100, otherwise 0.
        """
        if not mean:
            mean = round(100 / self.num_features , 2)
        
        if not std:
            std = round(mean / 3 , 2)

        # Generate features from a normal distribution with the specified mean and standard deviation
        X = np.random.normal(loc=mean, scale=std, size=(self.num_samples, self.num_features))
        
        # Check if the sum of each sample's features is greater than 100
        y = (X.sum(axis=1) > 100)
    
        return X, y
    
    #  -------------------------------------------------------------------------------------------------

    def generate_prime_number_dataset(self, mean=0, standard_deviation=1):
        """
        Generates a dataset with integer features drawn from a normal distribution
        with the specified mean and standard deviation. The target array y
        indicates if any feature in a sample is a prime number (1 if it is, 0 if it isn't).

        Args:
            mean (float): The mean of the normal distribution. Defaults to 0.
            standard_deviation (float): The standard deviation of the normal distribution. Defaults to 1.

        Returns:
            tuple: A tuple containing:
                - X (np.ndarray): The generated feature matrix of shape (num_samples, num_features) with integer values.
                - y (np.ndarray): The target vector of shape (num_samples, ) where each entry is 1 if the
                corresponding sample contains at least one prime number, otherwise 0.
        """
        def is_prime(n):
            """
            Checks if a number is a prime number.

            Args:
                n (int): The number to check for primality.

            Returns:
                bool: True if the number is prime, False otherwise.
            """
            if n <= 1:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    return False
            return True

        # Generate features from a normal distribution with the specified mean and standard deviation
        X = np.random.normal(loc=mean, scale=standard_deviation, size=(self.num_samples, self.num_features)).astype(int)
        
        # Check if each sample contains at least one prime number
        y = np.array([1 if any(is_prime(num) for num in sample) else 0 for sample in X])
        
        return X, y

    #  -------------------------------------------------------------------------------------------------

    def generate_variable_std_dataset(self, mean=0 , std_min: int = 5 , std_max: int = 15):
        """
        Generates a dataset where each sample has features drawn from a normal distribution
        with the specified mean and a unique standard deviation. The standard deviation is
        randomly generated for each sample.

        Args:
            mean (float): The mean of the normal distribution. Defaults to 0.

        Returns:
            tuple: A tuple containing:
                - X (np.ndarray): The generated feature matrix of shape (num_samples, num_features).
                - y (np.ndarray): The target vector of shape (num_samples, ) where each entry is the
                  standard deviation used to generate the features of the corresponding sample.
        """
        # Generate unique standard deviation for each sample
        std_devs = np.random.uniform(5, 15, size=self.num_samples)  # Generate random std deviations between 0.1 and 5
        
        # Initialize the feature matrix
        X = np.zeros((self.num_samples, self.num_features))
        
        # Generate features for each sample using the unique standard deviation
        for i in range(self.num_samples):
            X[i] = np.random.normal(loc=mean, scale=std_devs[i], size=self.num_features) ** 2
        
        # The target vector y is the standard deviation used for each sample
        y = std_devs
        
        return X, y
    
    #  -------------------------------------------------------------------------------------------------

    def generate_max_index_labels_dataset(self):
        """
        Generates a dataset with features drawn uniformly between 0 and 100.
        The target array y is the index of the maximum value in each sample.

        Returns:
            tuple: A tuple containing:
                - X (np.ndarray): The generated feature matrix of shape (num_samples, num_features), 
                  where each feature value is drawn uniformly between 0 and 100.
                - y (np.ndarray): The target vector of shape (num_samples, ) where each entry is the 
                  index of the maximum value in the corresponding sample's features.
        """
        # Generate features uniformly between 0 and 100
        X = np.random.uniform(low=0, high=100, size=(self.num_samples, self.num_features))
        
        # Find the index of the maximum value for each sample
        y = X.argmax(axis=1)
        
        return X, y

    #  -------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    # test samples
    num_samples = 1000
    num_features = 10
    
    generator = NumpyDatasetGenerator(num_samples, num_features)
    X, y = generator.generate_sum_greater_than_100_dataset()
    
    print("Sample X data:\n", X[:5])  # Print first 5 samples of X
    print("Sample y data:\n", y[:5])  # Print first 5 values of y
