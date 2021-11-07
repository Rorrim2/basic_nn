import numpy as np
import math

def convolution(image, filter, step, padding):
    size_x = image.shape[0] + padding*2
    size_y = image.shape[1] + padding*2
    new_x = math.ceil((image.shape[0] + padding*2 - filter.shape[0] + 1)/step)
    new_y = math.ceil((image.shape[1] + padding*2 - filter.shape[1] + 1)/step)
    convolve_image = np.zeros((size_x, size_y))
    if padding > 0:
        convolve_image[padding:-padding, padding:-padding] = image
    else:
        convolve_image = image

    output_image = np.zeros((new_x, new_y))
    for i in range(new_x):
        for j in range(new_y):
            output_image[i,j] = np.sum(convolve_image[(i*step):(i*step+3), (j*step):(j*step+3)] * filter)

    return output_image


input_image = np.array([[1,1,1,0,0],
                        [0,1,1,1,0],
                        [0,0,1,1,1],
                        [0,0,1,1,0],
                        [0,1,1,0,0]])

filter = np.array([[1,0,1],
                   [0,1,0],
                   [1,0,1]])

output_image = convolution(input_image, filter, 2, 1)
print(output_image)


# input_image = np.array([[1,2,3,4,5],
#                         [2,3,4,5,6],
#                         [3,4,5,6,7],
#                         [4,5,6,7,8],
#                         [5,6,7,8,9]])
#
# print(input_image[1:-1, 1:-1])
