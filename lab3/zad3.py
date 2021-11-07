import numpy as np
import random
from datetime import datetime
from Layer import Layer

class DNN:
    def __init__(self, input_count, hidden_count, output_count, alpha):
        first_layer = Layer(self.random_array(hidden_count, input_count))
        second_layer = Layer(self.random_array(output_count, hidden_count))
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

dnn = DNN(3, 4, 4, 0.01)

number_of_iterations = 20

for i in range(number_of_iterations):
    f = open("training_colors.txt", "r")

    content_training = f.readlines()

    for line in content_training:
        splitted = line.split()

        input = [float(splitted[0]), float(splitted[1]), float(splitted[2])]

        expected_output = np.zeros(4)
        expected_output[int(splitted[3]) - 1] = 1
        dnn.fit(np.array(input), np.array(expected_output))
    f.close()

f_test = open("test_colors.txt", "r")

content_test = f_test.readlines()

all = 0
right = 0

for line in content_test:
    splitted = line.split()

    input = [float(splitted[0]), float(splitted[1]), float(splitted[2])]

    print(splitted[3])
    out = dnn.predict(np.array(input))
    print(out)
    all+=1
    max = 0.0
    j = 0
    for i in range(len(out)):
        if i == 0:
            max = out[i]
            j = i
        elif out[i] > max:
            max = out[i]
            j = i
    if int(j) == int(splitted[3]) - 1:
        right+=1

print(all)
print(right)
print(dnn.weights)
f_test.close()

#ta siec potrzebuje wiecej iteracji niz siec z zadania 4 lab 2
