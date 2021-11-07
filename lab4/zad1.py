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
        self.layer_1_values = np.array([])
        self.layer_2_values = np.array([])

        self.layer_2_delta = np.array([])
        self.layer_1_delta = np.array([])


    def predict(self, input_vector):
        self.layer_1_values = np.array(np.matmul(input_vector, np.transpose(self.weights[0].layer_weights)))
        self.layer_1_values = self.relu(self.layer_1_values)
        self.layer_2_values = np.array(np.matmul(self.layer_1_values, np.transpose(self.weights[1].layer_weights)))
        return self.layer_2_values

    def fit(self, input_vector, expected_output):
        #1
        self.layer_1_values = np.array(np.matmul(input_vector, np.transpose(self.weights[0].layer_weights)))
        self.layer_1_values = self.dropout(self.relu(self.layer_1_values), 0.5)
        self.layer_2_values = np.array(np.matmul(self.layer_1_values, np.transpose(self.weights[1].layer_weights)))
        #2
        self.layer_2_delta = self.layer_2_values - expected_output

        #3
        self.layer_1_delta = np.matmul(self.layer_2_delta, self.weights[1].layer_weights)
        self.layer_1_delta = self.layer_1_delta * self.relu2deriv(self.layer_1_values)
        #4
        layer_2_weight_delta = np.reshape(np.kron(self.layer_2_delta, self.layer_1_values), (np.shape(self.layer_2_delta)[0], np.shape(self.layer_1_values)[0]))
        layer_1_weight_delta = np.reshape(np.kron(self.layer_1_delta, input_vector), (np.shape(self.layer_1_delta)[0], np.shape(input_vector)[0]))

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

    def dropout(self, arr, zero_prob):
        nums = np.ones(len(arr))
        temp = 1/zero_prob
        nums[:int(len(arr)/temp)] = 0
        np.random.shuffle(nums)

        arr = arr * (temp) * nums
        return arr


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

dnn = DNN(784, 40, 10, 0.005)

number_of_iterations = 200
number_of_images = 1000

f_train_labels =  open('train-labels.idx1-ubyte', 'rb')
f_train_images =  open('train-images.idx3-ubyte', 'rb')

labels_magic_number = struct.unpack('>i', f_train_labels.read(4))[0]
images_magic_number = struct.unpack('>i', f_train_images.read(4))[0]

labels_number_of_labels = struct.unpack('>i', f_train_labels.read(4))[0]
images_number_of_labels = struct.unpack('>i', f_train_images.read(4))[0]

row_num = struct.unpack('>i', f_train_images.read(4))[0]
col_num = struct.unpack('>i', f_train_images.read(4))[0]

data = []
labels = []

for i in range(number_of_images):
    data_row = []
    for x in range(row_num * col_num):
        data_row.append(round(f_train_images.read(1)[0] / 255.0, 3))
    labels.append(f_train_labels.read(1)[0])
    data.append(data_row)

for i in range(number_of_iterations):
    print(i)
    for image in range(number_of_images):
        expected_output = np.zeros(10)

        expected_output[labels[image]] = 1
        dnn.fit(np.array(data[image]), np.array(expected_output))

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

for image in range(number_of_images):
    data2 = []
    for x in range(row_num * col_num):
        data2.append(round(f_test_images.read(1)[0] / 255.0, 3))

    expected_output = np.zeros(10)

    index = f_test_labels.read(1)[0]
    expected_output[index] = 1
    out = dnn.predict(np.array(data2))
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
    if int(j) == int(index):
        right+=1

print(all)
print(right)


f_test_labels.close()
f_test_images.close()

#40, 1000, 1000, 350 - 816/1000
#100, 10000, 10000, 350 - 9406/10000