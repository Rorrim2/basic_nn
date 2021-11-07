import numpy as np
import random
from datetime import datetime
import struct
from Layer import Layer

class DNN:
    def __init__(self, input_count, hidden_count, output_count, alpha):
        first_layer = Layer(self.random_array(hidden_count, input_count, -0.1, 0.1))
        second_layer = Layer(self.random_array(output_count, hidden_count, -0.01, 0.01))
        self.weights = [first_layer, second_layer]

        self.all_error = 0.0
        self.alpha = alpha
        self.layer_1_values = np.array([])
        self.layer_2_values = np.array([])

        self.layer_2_delta = np.array([])
        self.layer_1_delta = np.array([])


    def predict(self, input_vector):
        self.layer_1_values = np.array(np.matmul(input_vector, np.transpose(self.weights[0].layer_weights)))
        self.layer_1_values = self.tanh(self.layer_1_values)
        self.layer_2_values = np.array(np.matmul(self.layer_1_values, np.transpose(self.weights[1].layer_weights)))
        return self.layer_2_values

    def fit(self, input_vector, expected_output):
        #1
        self.layer_1_values = np.array(np.matmul(input_vector, np.transpose(self.weights[0].layer_weights)))
        self.layer_1_values = self.tanh(self.layer_1_values)
        for i in range(self.layer_1_values.shape[0]):
            self.layer_1_values[i] = self.dropout(self.layer_1_values[i], 0.5)
        self.layer_2_values = np.array(np.matmul(self.layer_1_values, np.transpose(self.weights[1].layer_weights)))
        self.layer_2_values = self.softmax(self.layer_2_values)
        #2
        self.layer_2_delta = (self.layer_2_values - expected_output)/100
        #3
        self.layer_1_delta = np.matmul(self.layer_2_delta, self.weights[1].layer_weights)
        self.layer_1_delta = self.layer_1_delta * self.tanh2deriv(self.layer_1_values)
        #4
        layer_2_weight_delta = np.matmul(np.transpose(self.layer_2_delta), self.layer_1_values)
        layer_1_weight_delta = np.matmul(np.transpose(self.layer_1_delta), input_vector)
        # layer_2_weight_delta = np.reshape(np.kron(self.layer_2_delta, self.layer_1_values), (np.shape(self.layer_2_delta)[0], np.shape(self.layer_1_values)[0]))
        # layer_1_weight_delta = np.reshape(np.kron(self.layer_1_delta, input_vector), (np.shape(self.layer_1_delta)[0], np.shape(input_vector)[0]))

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
    def random_array(self, x, y, range_left, range_right):
        output = []
        for i in range(x):
            row = []
            for j in range(y):
                row.append(round(random.uniform(range_left, range_right), 3))
            output.append(row)
        return output

    def relu(self, x):
        return (x > 0) * x

    def relu2deriv(self, output):
        return (output > 0) * 1

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid2deriv(self, output):
        return output * (1 - output)

    def tanh(self, x):
        return np.tanh(x)

    def tanh2deriv(self, output):
        return 1 - (output**2)

    def softmax(self, x):
        temp = np.exp(x)
        return temp / np.sum(temp, axis=1, keepdims=True)

dnn = DNN(784, 100, 10, 0.0001)

number_of_iterations = 350
number_of_images = 10000
mini_batch_stack = 100

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

for i in range(images_number_of_labels):
    data_row = []
    for x in range(row_num * col_num):
        data_row.append(round(f_train_images.read(1)[0] / 255.0, 3))
    labels.append(f_train_labels.read(1)[0])
    data.append(data_row)

for i in range(number_of_iterations):
    print(i)
    for image in range(int(images_number_of_labels/mini_batch_stack)):
        one_mini_batch_portion = []
        one_mini_batch_labels = []
        expected_output = []
        for mini in range(mini_batch_stack):
            one_mini_batch_portion.append(data[(image * mini_batch_stack) + mini])
            expected_output.append(np.zeros(10))
            expected_output[mini][labels[(image * mini_batch_stack) + mini]] = 1
        dnn.fit(np.array(one_mini_batch_portion), np.array(expected_output))

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
'''
100, 1000, 1000, 350:
0.02 - 864/1000
0.01 - 835/1000
0.001 - 639/1000
0.05 - 816/1000
0.03 - 854/1000
0.025 - 865/1000
0.015 - 850/1000
100, 10000, 10000, 350: 
0.015 - 980/10000
0.001 - 9002/10000
0.002 - 9116/10000
0.003 - 9182/10000
0.004 - 9088/10000
0.0025 - 9149/10000
100, 60000, 10000, 350:
0.0025 - 1078/10000
0.0001 - 8838/10000
'''
