# ------------------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
# ------------------------------------------------------------------------------------------------------------

import numpy as np

# ReLU activation function
def relu(x: np.ndarray) -> np.ndarray:
    """Relu function

    Args:
        x (np.ndarray): input np array

    Returns:
        np.ndarray: output np array with only positive values unchanged, negative being 0
    """
    return np.maximum(0, x)

# Derivative of ReLU
def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Relu derivative function

    Args:
        x (np.ndarray): input np array

    Returns:
        np.ndarray: output np array having derivative of the Relu function at the various input values of array
    """
    return np.where(x > 0, 1, 0)

# Simple neural network class
class SimpleNeuralNetwork:
    """This class implements a SimpleNeuralNetwork from scratch.  
    It assumes a single input, a single output, and 3 neurons in the hidden layer, with Relu activations
    in the 3 neurons of the hidden layer.
    """
    def __init__(self):
        """The init method randomly initializes weights and bias terms (3 weights from input to hidden layer, 
        3 weights from hidden layer to output, 3 bias terms for each of the three neurons in the hidden layer, 
        and 1 bias term for the output neuron. )
        """
        # Initialize weights randomly
        generator = np.random.default_rng(42)
        self.input_to_hidden_weights = generator.standard_normal((1, 3))  # Weights from input to hidden layer
        self.hidden_to_output_weights = generator.standard_normal((3, 1))  # Weights from hidden layer to output
        self.hidden_bias = generator.standard_normal((1, 3))  # Bias for hidden layer
        self.output_bias = generator.standard_normal((1, 1))  # Bias for output layer
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """The forward pass of the simple neural network

        Args:
            x (np.ndarray): Takes in an np array - in this case of a single dimension because there is only one input (ie it is a scalar)

        Returns:
            np.ndarray: The output is also an np array with only one dimension (is also a scalar value)
        """
        # Forward pass
        self.hidden_layer_input = np.dot(x, self.input_to_hidden_weights) + self.hidden_bias
        self.hidden_layer_output = relu(self.hidden_layer_input)
        
        self.output_layer_input = np.dot(self.hidden_layer_output, self.hidden_to_output_weights) + self.output_bias
        output = self.output_layer_input  # Linear output for regression
        
        return output
    
    def backward(self, x: np.ndarray, y: np.ndarray, output: np.ndarray, learning_rate: float=0.001):
        """This does the backward pass using the MSE loss between the predicted output and the y values passed during training

        Args:
            x (np.ndarray): input np.array of shape (?, 1)
            y (np.ndarray): output y np.array of shape (?, 1)
            output (np.ndarray): predicted output np.array of shape (?, 1)
            learning_rate (float, optional): learning rate. Defaults to 0.01.
        """
        # Loss derivative (Mean Squared Error) (d Loss / d output)
        loss_derivative = 2 * (output - y)
        
        # Backpropagate the error (d Loss / d  output_layer_input)
        d_output_layer_input = loss_derivative
        
        # (d Loss / d hidden_to_output_weights)
        d_hidden_to_output_weights = np.dot(self.hidden_layer_output.T, d_output_layer_input)
        
        # (d Loss / d output_bias)
        d_output_bias = np.sum(d_output_layer_input, axis=0)
        
        # (d Loss / d hidden_layer_output)
        d_hidden_layer_output = np.dot(d_output_layer_input, self.hidden_to_output_weights.T)
        
        # (d Loss / d hidden_layer_input)
        d_hidden_layer_input = d_hidden_layer_output * relu_derivative(self.hidden_layer_input)
        
        # (d Loss / d input_to_hidden_weights)
        d_input_to_hidden_weights = np.dot(x.T, d_hidden_layer_input)
        
        # (d Loss / d hidden_bias)
        d_hidden_bias = np.sum(d_hidden_layer_input, axis=0)
        
        # Update weights and biases
        self.hidden_to_output_weights -= learning_rate * d_hidden_to_output_weights
        self.output_bias -= learning_rate * d_output_bias
        
        self.input_to_hidden_weights -= learning_rate * d_input_to_hidden_weights
        self.hidden_bias -= learning_rate * d_hidden_bias
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int =1000, learning_rate: float =0.01):
        """The training loop over 1000 epochs

        Args:
            x (np.ndarray): input array of shape (?, 1)
            y (np.ndarray): output array of shape (?, 1)
            epochs (int, optional): number of epochs. Defaults to 1000.
            learning_rate (float, optional): learning rate. Defaults to 0.01.
        """
        for epoch in range(epochs):
            first_step_of_epoch = True
            for x_in, y_in in zip(x,y):
                output = self.forward(x_in)
                loss = np.sum((output - y_in)**2, axis = 0)
                if epoch%20 == 0 and first_step_of_epoch:
                    print(f'epoch: {epoch}: loss: {loss}')
                    first_step_of_epoch = False
                self.backward(x_in, y_in, output, learning_rate)

# Example usage
if __name__ == "__main__":
    # Generate 1000 points uniformly spread out in the range [0, 1]
    x = np.linspace(0, 1, 1000).reshape(-1, 1)

    # Calculate y = 4x^2 + 11x + 3
    y = 4 * x**2 + 11 * x + 3

    # Initialize and train the neural network
    nn = SimpleNeuralNetwork()
    nn.train(x, y, epochs=100, learning_rate=0.0001)

    # Test the network
    
    x_sample = np.array([0.2, 0.8, 0.4, 0.9]).reshape(-1, 1)
    y_sample = 4 * x_sample**2 + 11*x_sample + 3
    output_sample = nn.forward(x_sample)
    print(f"Input being tested: {x_sample}")
    print(f"Expected output: {y_sample}")
    print(f"Predicted output: {output_sample}")
