import numpy as np
import random
from datetime import datetime

class DNN:
    def __init__(self, input_count, hidden_count, hidden_layers_vector, output_count):
        self.weights = []
        self.output = []
        self.hidden_count_vector = hidden_layers_vector

        last = input_count
        # create weights for hidden layers
        for i in range(hidden_count):
            self.weights.append(self.random_array(hidden_layers_vector[i], last))
            last = hidden_layers_vector[i]
        self.weights.append(self.random_array(output_count, last))
        self.hidden_count_vector.append(output_count)

    def predict(self, input_vector):
        input = input_vector
        for i in range(len(self.weights)):
            input = self.step(input, self.weights[i])
        return input

    def fit(self):
        return

    def dot_product(self, a, b):
        assert (len(a) == len(b))
        output = []
        for i in range(len(a)):
            output.append(a[i] * b[i])
        return output

    def vector_sum(self, a):
        output = 0
        for i in range(len(a)):
            output += a[i]
        return output

    # takes input and weigts and proceed with one step of a weighning
    # to produce the output_count of neurons
    def step(self, input, weights):
        output = []
        for i in range(len(weights)):
            a = self.dot_product(input, weights[i])
            output.append(self.vector_sum(a))
        return output

    # create 2D random array
    def random_array(self, x, y):
        output = []
        for i in range(x):
            row = []
            for j in range(y):
                row.append(round(random.random(), 3))
            output.append(row)
        return output

hidden_layers_count = []
dnn = DNN(4, 0, hidden_layers_count, 3)

inp = [1,2,3,4]

out = dnn.predict(inp)

print(out)