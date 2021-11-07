import numpy as np

def section_image(image, shape):
    new_x = image.shape[0] - (shape[0] - 1)
    new_y = image.shape[1] - (shape[1] - 1)

    sections = []

    for i in range(new_x):
        for j in range(new_y):
            temp = np.reshape(np.transpose(image[i:i+shape[0], j:j+shape[1]]),
                                       (1, shape[0] * shape[1]))
            sections.append(temp[0])

    return np.array(sections)

def relu(x):
    return (x > 0) * x

def max_pooling(image, mask_size, step):
    new_x = int((image.shape[0] - mask_size) / step + 1)
    new_y = int((image.shape[1] - mask_size) / step + 1)


    output = np.zeros((new_x, new_y))
    for x in range(output.shape[0]):
        for y in range((output.shape[1])):
            output[x, y] = np.max(image[x * step: x * step+mask_size,
                                 y * step: y * step+mask_size])
    return output


input = np.array([[3, 6, 7, 5, 3, 5],
                  [6, 2, 9, 1, 2, 7],
                  [0, 9, 3, 6, 0, 6],
                  [2, 6, 1, 8, 7, 9],
                  [2, 0, 2, 3, 7, 5],
                  [9, 2, 2, 8, 9, 7]])

out = max_pooling(input, 2, 2)
print(out)

# input = np.array([[8.5, 0.65, 1.2],
#                   [9.5, 0.8, 1.3],
#                   [9.9, 0.8, 0.5],
#                   [9.0, 0.9, 1.0]])
#
# expected_output = [0, 1]
#
# kernel_1_weights = [0.1, 0.2, -0.1, -0.1, 0.1, 0.9, 0.1, 0.4, 0.1]
# kernel_2_weights = [0.3, 1.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1]
#
# kernels = [kernel_1_weights, kernel_2_weights]
#
# layer_2_weights = [[0.1, -0.2, 0.1, 0.3],
#                    [0.2, 0.1, 0.5, -0.3]]
#
# alpha = 0.01
#
# #section image
# image_sections = section_image(input, (3,3))
# kernel_layer = np.matmul(image_sections, np.transpose(kernels))
# kernel_layer_flatten = np.reshape(relu(kernel_layer), (1, kernel_layer.shape[0] * kernel_layer.shape[1]))
# kernel_layer_flatten = kernel_layer_flatten[0]
#
# layer_2_values = np.matmul(kernel_layer_flatten, np.transpose(layer_2_weights))
# #since here starts train
#
# layer_2_delta = layer_2_values - expected_output
#
# layer_1_delta = np.matmul(layer_2_delta, layer_2_weights)
#
# layer_1_delta_reshaped = np.reshape(layer_1_delta, kernel_layer.shape)
#
# layer_2_weight_delta = np.reshape(layer_2_delta, (layer_2_delta.shape[0], 1)) * kernel_layer_flatten
#
# layer_1_weight_delta = np.matmul(np.transpose(layer_1_delta_reshaped), image_sections)
#
# layer_2_weights = layer_2_weights - alpha * layer_2_weight_delta
#
# kernels = kernels - alpha * layer_1_weight_delta
#
# print(kernels)

