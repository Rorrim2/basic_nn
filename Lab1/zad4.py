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

def deep_neural_network(input, weight):
    tranfer = []
    hidden = weight[0]
    for i in range(len(hidden)):
        a = dot_product(input, hidden[i])
        tranfer.append(sum(a))
    output = []
    out = weight[1]
    for i in range(len(out)):
        a = dot_product(tranfer, out[i])
        output.append(sum(a))
    return output

inputs = [8.5, 0.65, 1.2]

hidden = [[0.1, 0.2, -0.1],
          [-0.1, 0.1, 0.9],
          [0.1, 0.4, 0.1]]

output = [[0.3, 1.1, -0.3],
          [0.1, 0.2, 0.0],
          [0.0, 1.3, 0.1]]

weights = [hidden, output]

pred = deep_neural_network(inputs, weights)

print(pred)