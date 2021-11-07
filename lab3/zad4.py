import numpy as np
import random
from datetime import datetime
import struct

class DNN:
    def __init__(self, input_count, hidden_count, output_count, alpha):
        self.weights = []
        self.all_error = 0.0
        self.alpha = alpha

        self.weights.append(self.random_array(hidden_count, input_count))
        self.weights.append(self.random_array(output_count, hidden_count))



    def predict(self, input_vector):
        output = []
        hidden = []
        for i in range(len(self.weights[0])):
            hidden.append(self.vector_sum(self.dot_product(input_vector, self.weights[0][i])))
        for i in range(len(self.weights[1])):
            output.append(self.vector_sum(self.dot_product(hidden, self.weights[1][i])))
        return output

    def fit(self, input_vector, expected_output):
        #1
        values = [[], []]
        for i in range(len(self.weights[0])):
            values[0].append(self.relu(self.vector_sum(self.dot_product(input_vector, self.weights[0][i]))))
        for i in range(len(self.weights[1])):
            values[1].append(self.vector_sum(self.dot_product(values[0], self.weights[1][i])))
        #2
        deltas_all = [[], []]
        for i in range(len(values[1])):
            deltas_all[0].append(values[1][i] - expected_output[i])
        #3
        for i in range(len(values[0])):
            delta = self.relu2deriv(values[0][i]) * sum(
                self.dot_product(deltas_all[0], [item[i] for item in self.weights[1]]))
            deltas_all[1].append(delta)
        #4
        for i in range(len(self.weights[1])):
            for j in range(len(self.weights[1][i])):
                self.weights[1][i][j] = self.weights[1][i][j] - ((values[1][i] - expected_output[i]) * values[0][j]) * self.alpha

        for i in range(len(self.weights[0])):
            for j in range(len(self.weights[0][i])):
                self.weights[0][i][j] = self.weights[0][i][j] - deltas_all[1][i] * input_vector[j] * self.alpha
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
                row.append(round(random.uniform(-0.1, 0.1), 3))
            output.append(row)
        return output

    def relu(self, x):
        return (x > 0) * x

    def relu2deriv(self, output):
        return output > 0

dnn = DNN(784, 40, 10, 0.01)

number_of_iterations = 1

f_train_labels =  open('train-labels.idx1-ubyte', 'rb')
f_train_images =  open('train-images.idx3-ubyte', 'rb')

labels_magic_number = struct.unpack('>i', f_train_labels.read(4))[0]
images_magic_number = struct.unpack('>i', f_train_images.read(4))[0]

labels_number_of_labels = struct.unpack('>i', f_train_labels.read(4))[0]
images_number_of_labels = struct.unpack('>i', f_train_images.read(4))[0]

row_num = struct.unpack('>i', f_train_images.read(4))[0]
col_num = struct.unpack('>i', f_train_images.read(4))[0]

for i in range(number_of_iterations):
    for image in range(10000):
        print(image)
        data = []
        for x in range(row_num * col_num):
            data.append(round(f_train_images.read(1)[0] / 255.0, 3))

        expected_output = np.zeros(10)

        expected_output[f_train_labels.read(1)[0] - 1] = 1
        dnn.fit(data, expected_output)

f_train_labels.close()
f_train_images.close()

f_test_labels =  open('t10k-labels.idx1-ubyte', 'rb')
f_test_images =  open('t10k-images.idx3-ubyte', 'rb')

labels_magic_number = struct.unpack('>i', f_test_labels.read(4))[0]
images_magic_number = struct.unpack('>i', f_test_images.read(4))[0]

labels_number_of_labels = struct.unpack('>i', f_test_labels.read(4))[0]
images_number_of_labels = struct.unpack('>i', f_test_images.read(4))[0]

row_num = struct.unpack('>i', f_test_images.read(4))[0]
col_num = struct.unpack('>i', f_test_images.read(4))[0]


all = 0
right = 0

for image in range(200):
    data = []
    for x in range(row_num * col_num):
        data.append(round(f_test_images.read(1)[0] / 255.0, 3))

    expected_output = np.zeros(10)

    index = f_test_labels.read(1)[0]
    expected_output[index] = 1
    out = dnn.predict(data)
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
    print(out)
    print("index")
    print(index)
    print("j")
    print(j)
    if int(j) == int(index):
        right+=1

print(all)
print(right)


f_test_labels.close()
f_test_images.close()