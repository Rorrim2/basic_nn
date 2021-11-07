def dot_product(a, b):
    assert(len(a) == len(b))
    output = []
    for i in range(len(a)):
        output.append(a[i] * b[i])
    return output

def sum(a):
    output = 0
    for i in range(len(a)):
        output += a[i]
    return output

def neural_network(input, weight):
    output = []
    for i in range(len(weight)):
        a = dot_product(input, weight[i])
        output.append(sum(a))
    return output

inputs = [8.5, 0.65, 1.2]
weights = [[0.1, 0.1, -0.3],
           [0.1, 0.2, 0.0],
           [0.0, 1.3, 0.1]]

pred = neural_network(inputs, weights)

print(pred)