import numpy as np
import random
from datetime import datetime

def dot_product(a, b): 
    assert(len(a) == len(b))
    output = []
    for i in range(len(a)):
        output.append(a[i] * b[i])
    return output

def vector_sum(a):
    output = 0
    for i in range(len(a)):
        output += a[i]
    return output

#takes input and weigts and proceed with one step of a weighning 
# to produce the output_count of neurons
def step(input, weights, output_count):
    output = []
    for i in range(output_count):
        a = dot_product(input, weights[i])
        output.append(vector_sum(a))
    return output

#create 2D random array
def random_array(x, y):
    output = []
    for i in range(x):
        row = []
        for j in range(y):
            row.append(round(random.random(), 3))
        output.append(row)
    return output

def init_weights(input_vector, hidden_count, hidden_layers_vector, output_count):
    hidden = []
    last = len(input_vector)
    # create weights for hidden layers
    for i in range(hidden_count):
        hidden.append(random_array(hidden_layers_vector[i], last))
        last = hidden_layers_vector[i]

    weights = random_array(hidden_layers_vector[-1], output_count)

def predict(input_vector, hidden_count, hidden_layers_vector, output_count):
    hidden = []
    last = len(input_vector)
    #create weights for hidden layers
    for i in range(hidden_count):
        hidden.append(random_array(hidden_layers_vector[i], last))
        last = hidden_layers_vector[i]
    #solve hidden layers
    input = input_vector
    for i in range(hidden_count):
        input = step(input, hidden[i], hidden_layers_vector[i])
    #create weights for output
    weights = random_array(hidden_layers_vector[-1], output_count)
    output = step(input, weights, output_count)
    return output


random.seed(datetime.now())
input = [1, 2, 3, 4, 5, 6, 7, 8]
hidden_layers_neuron_count = [4, 2]

pred = predict(input, 2, hidden_layers_neuron_count, 2)

print(pred)