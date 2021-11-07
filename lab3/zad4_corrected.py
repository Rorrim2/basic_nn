import numpy as np
import random
from datetime import datetime
import struct
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
                row.append(round(random.uniform(-0.1, 0.1), 3))
            output.append(row)
        return output

    def relu(self, x):
        return (x > 0) * x

    def relu2deriv(self, output):
        return (output > 0) * 1

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
    for image in range(images_number_of_labels):
        print(image)
        data = []
        for x in range(row_num * col_num):
            data.append(round(f_train_images.read(1)[0] / 255.0, 3))

        expected_output = np.zeros(10)

        expected_output[f_train_labels.read(1)[0]] = 1
        dnn.fit(np.array(data), np.array(expected_output))

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

for image in range(images_number_of_labels):
    data = []
    for x in range(row_num * col_num):
        data.append(round(f_test_images.read(1)[0] / 255.0, 3))

    expected_output = np.zeros(10)

    index = f_test_labels.read(1)[0]
    expected_output[index] = 1
    out = dnn.predict(np.array(data))
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
    # print(out)
    # print("index")
    # print(index)
    # print("j")
    # print(j)
    if int(j) == int(index):
        right+=1

print(all)
print(right)


f_test_labels.close()
f_test_images.close()