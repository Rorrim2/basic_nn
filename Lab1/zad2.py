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
    a = dot_product(input, weight)
    return sum(a)

inputs = [2.9, 1.5, 2.8]
weights = [0.9, 0.7, 0.3]

pred = neural_network(inputs, weights)

print(pred)