import numpy as np
import random
from datetime import datetime

class DNN:
    def __init__(self, input_count, output_count, alpha):
        self.weights = []
        self.all_error = 0.0
        self.alpha = alpha

        self.weights = self.random_array(output_count, input_count)
        # self.weights = [[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 1.3, 0.1], [0.2, 0.5, 0.6]]

    def predict(self, input_vector):
        output = []
        for i in range(len(self.weights)):
            a = self.dot_product(input_vector, self.weights[i])
            output.append(self.vector_sum(a))
        return output

    def fit(self, input_vector, expected_output):
        output = []
        error_raw = []
        for w in range(len(self.weights)):
            a = self.dot_product(input_vector, self.weights[w])
            prediction = sum(a)
            output.append(prediction)

            error = (prediction - expected_output[w])**2
            delta = prediction - expected_output[w]
            for i in range(len(self.weights[w])):
                weight_delta = input_vector[i] * delta
                self.weights[w][i] = self.weights[w][i] - weight_delta * self.alpha
        self.all_error += self.vector_sum(error_raw)
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

dnn = DNN(3, 4, 0.01)

number_of_iterations = 10

for i in range(number_of_iterations):
    f = open("training_colors.txt", "r")

    content_training = f.readlines()

    for line in content_training:
        splitted = line.split()

        input = [float(splitted[0]), float(splitted[1]), float(splitted[2])]

        expected_output = np.zeros(4)
        expected_output[int(splitted[3]) - 1] = 1
        dnn.fit(input, expected_output)
    f.close()

f_test = open("test_colors.txt", "r")

content_test = f_test.readlines()

all = 0
right = 0

for line in content_test:
    splitted = line.split()

    input = [float(splitted[0]), float(splitted[1]), float(splitted[2])]


    print(splitted[3])
    out = dnn.predict(input)
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
