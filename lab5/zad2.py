import numpy as np
import random
from datetime import datetime
import struct
from Layer import Layer

class CNN:
    def __init__(self, input_shape, filters_shape, filters_count, output_count, alpha):
        filters = []
        filters = self.random_array(filters_count, filters_shape[0] * filters_shape[1], -0.01, 0.01)
        self.kernels = np.array(filters)

        neurons_count_x = input_shape[0] - (filters_shape[0] - 1)
        neurons_count_y = input_shape[1] - (filters_shape[1] - 1)

        self.second_layer = Layer(self.random_array(output_count, neurons_count_x*neurons_count_y * filters_count,
                                                    -0.1, 0.1))

        self.alpha = alpha


    def predict(self, input_vector):
        image_sections = self.section_image(input_vector, (3, 3))
        kernel_layer = np.matmul(image_sections, np.transpose(self.kernels))
        kernel_layer_flatten = np.reshape(self.relu(kernel_layer), (1, kernel_layer.shape[0] * kernel_layer.shape[1]))
        kernel_layer_flatten = kernel_layer_flatten[0]
        layer_2_values = np.matmul(kernel_layer_flatten, np.transpose(self.second_layer.layer_weights))
        return layer_2_values

    def fit(self, input_vector, expected_output):
        image_sections = self.section_image(input_vector, (3,3))
        kernel_layer = np.matmul(image_sections, np.transpose(self.kernels))
        kernel_layer_flatten = np.reshape(self.relu(kernel_layer), (1, kernel_layer.shape[0] * kernel_layer.shape[1]))
        kernel_layer_flatten = kernel_layer_flatten[0]
        layer_2_values = np.matmul(kernel_layer_flatten, np.transpose(self.second_layer.layer_weights))

        layer_2_delta = layer_2_values - expected_output
        layer_1_delta = np.matmul(layer_2_delta, self.second_layer.layer_weights)
        layer_1_delta = layer_1_delta * self.relu2deriv(kernel_layer_flatten)
        layer_1_delta_reshaped = np.reshape(layer_1_delta, kernel_layer.shape)

        layer_2_weight_delta = np.reshape(layer_2_delta, (layer_2_delta.shape[0], 1)) * kernel_layer_flatten
        layer_1_weight_delta = np.matmul(np.transpose(layer_1_delta_reshaped), image_sections)

        self.second_layer.layer_weights = self.second_layer.layer_weights - self.alpha * layer_2_weight_delta
        self.kernels = self.kernels - self.alpha * layer_1_weight_delta
        return layer_2_values

    def dot_product(self, a, b):
        assert (len(a) == len(b))
        output = []
        for i in range(len(a)):
            output.append(a[i] * b[i])
        return output

    def section_image(self, image, shape):
        new_x = image.shape[0] - (shape[0] - 1)
        new_y = image.shape[1] - (shape[1] - 1)

        sections = []

        for i in range(new_x):
            for j in range(new_y):
                temp = np.reshape(np.transpose(image[i:i + shape[0], j:j + shape[1]]),
                                  (1, shape[0] * shape[1]))
                sections.append(temp[0])

        return np.array(sections)

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
                row.append(random.uniform(range_left, range_right))
            output.append(row)
        return output

    def relu(self, x):
        return (x > 0) * x

    def relu2deriv(self, output):
        return (output > 0) * 1

dnn = CNN((28,28), (3,3), 16, 10, 0.01)



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

number_of_images = 1000
number_of_iterations = 50


data = []

for i in range(number_of_images):
    data_one = np.zeros((row_num, col_num))
    for x in range(row_num):
        for y in range(col_num):
            data_one[x][y] = (f_train_images.read(1)[0] / 255.0)
    labels.append(f_train_labels.read(1)[0])
    data.append(data_one)

for i in range(number_of_iterations):
    all = 0
    right = 0
    print(i)
    for image in range(number_of_images):
        expected_output = np.zeros(10)
        index = labels[image]
        expected_output[index] = 1
        out = dnn.fit(np.array(data[image]), np.array(expected_output))

        all += 1
        max = 0.0
        j = 0
        for k in range(len(out)):
            if k == 0:
                max = out[k]
                j = k
            elif out[k] > max:
                max = out[k]
                j = k
        if int(j) == int(index):
            right += 1

    print(all)
    print(right)

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
    data_one = np.zeros((row_num, col_num))

    for x in range(row_num):
        for y in range(col_num):
            data_one[x][y] = (f_test_images.read(1)[0] / 255.0)

    expected_output = np.zeros(10)

    index = f_test_labels.read(1)[0]
    expected_output[index] = 1
    out = dnn.predict(np.array(data_one))
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