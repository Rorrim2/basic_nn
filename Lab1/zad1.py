def neural_network(input, weight):
    return input * weight

inputs = [2.9, 1.5, 2.8]
weights = [0.9, 0.7, 0.3]

pred = neural_network(inputs[0], weights[0])

print(pred)