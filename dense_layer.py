from layer import Layer
import numpy as np


class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def fwd_prop(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def bwd_prop(self, output_err, learning_rate):
        input_err = np.dot(output_err, self.weights.T)
        weights_err = np.dot(self.input.T, output_err)

        self.weights -= learning_rate * weights_err
        self.bias -= learning_rate * output_err
        return input_err
