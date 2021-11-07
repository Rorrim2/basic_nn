import numpy as np
import random
from datetime import datetime
from Layer import Layer

class DNN:
    def __init__(self, input_count, hidden_count, output_count, alpha):
        first_layer = Layer([[0.1, 0.2, -0.1],
          [-0.1, 0.1, 0.9],
          [0.1, 0.4, 0.1]])
        second_layer = Layer([[0.3, 1.1, -0.3],
          [0.1, 0.2, 0.0],
          [0.0, 1.3, 0.1]])
        self.weights = [first_layer, second_layer]
        self.all_error = 0.0
        self.alpha = alpha


    def predict(self, input_vector):
        hidden = np.array(np.matmul(input_vector, np.transpose(self.weights[0].layer_weights)))
        hidden = self.relu(hidden)
        output = np.array(np.matmul(hidden, np.transpose(self.weights[1].layer_weights)))
        return output

    def fit(self, input_vector, expected_output):
        #1
        hidden = np.array(np.matmul(input_vector, np.transpose(self.weights[0].layer_weights)))
        hidden = self.relu(hidden)
        output = np.array(np.matmul(hidden, np.transpose(self.weights[1].layer_weights)))
        #2
        layer_2_delta = output - expected_output
        self.all_error = np.sum(layer_2_delta**2)
        #3
        layer_1_delta = np.matmul(layer_2_delta, self.weights[1].layer_weights)
        layer_1_delta = layer_1_delta * self.relu2deriv(hidden)
        #4

        layer_2_weight_delta = np.reshape(np.kron(layer_2_delta, hidden), (np.shape(layer_2_delta)[0], np.shape(hidden)[0]))
        layer_1_weight_delta = np.reshape(np.kron(layer_1_delta, input_vector), (np.shape(layer_1_delta)[0], np.shape(input_vector)[0]))
        self.weights[1].layer_weights = self.weights[1].layer_weights - self.alpha * layer_2_weight_delta
        self.weights[0].layer_weights = self.weights[0].layer_weights - self.alpha * layer_1_weight_delta
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

    # create 2D random array
    def random_array(self, x, y):
        output = []
        for i in range(x):
            row = []
            for j in range(y):
                row.append(round(random.random(), 3))
            output.append(row)
        return output

    def relu(self, x):
        return (x > 0) * x

    def relu2deriv(self, output):
        return (output > 0) * 1

dnn = DNN(3, 3, 3, 0.01)

input1 = [8.5, 0.65, 1.2]
input2 = [9.5, 0.8, 1.3]
input3 = [9.9, 0.8, 0.5]
input4 = [9.0, 0.9, 1.0]
expected_output1 = [0.1, 1, 0.1]
expected_output2 = [0.0, 1, 0.0]
expected_output3 = [0.0, 0, 0.1]
expected_output4 = [0.1, 1, 0.2]

er = 0.0
for i in range(50):
    er = 0.0
    dnn.fit(np.array(input1), np.array(expected_output1))
    er += dnn.all_error
    dnn.fit(np.array(input2), np.array(expected_output2))
    er += dnn.all_error
    dnn.fit(np.array(input3), np.array(expected_output3))
    er += dnn.all_error
    dnn.fit(np.array(input4), np.array(expected_output4))
    er += dnn.all_error


print(er)

