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

def relu(x):
    return (x > 0) * x

def relu2deriv(output):
    return output>0

def deep_neural_network(input, weight, expected_output, alpha):
    tranfer = []
    #1
    for i in range(len(weight[0])):
        a = dot_product(input, weight[0][i])
        tranfer.append(relu(sum(a)))

    output = []
    for i in range(len(weight[1])):
        a = dot_product(tranfer, weight[1][i])
        output.append(sum(a))
    #2
    deltas_all = []
    deltas_layer = []
    errors = []
    for i in range(len(output)):
        prediction = output[i] - expected_output[i]
        errors.append(prediction**2)
        deltas_layer.append(prediction)
    deltas_all.append(deltas_layer)
    #3
    delta_layer = []
    for i in range(len(tranfer)):
        delta = relu2deriv(tranfer[i]) * sum(dot_product(deltas_all[0], weight[1][:][i]))
        delta_layer.append(delta)
    deltas_all.append(delta_layer)
    #4
    for i in range(len(weight[1])):
        for j in range(len(weight[1][i])):
            weight_layer_delta = ((output[i] - expected_output[i]) * tranfer[j])
            weight[1][i][j] = weight[1][i][j] - weight_layer_delta * alpha

    for i in range(len(weight[0])):
        for j in range(len(weight[0][i])):
            weight_layer_delta = deltas_all[1][i] * input[j]
            weight[0][i][j] = weight[0][i][j] - weight_layer_delta * alpha

    return errors

input1 = [8.5, 0.65, 1.2]
input2 = [9.5, 0.8, 1.3]
input3 = [9.9, 0.8, 0.5]
input4 = [9.0, 0.9, 1.0]

hidden = [[0.1, 0.2, -0.1],
          [-0.1, 0.1, 0.9],
          [0.1, 0.4, 0.1]]

output = [[0.3, 1.1, -0.3],
          [0.1, 0.2, 0.0],
          [0.0, 1.3, 0.1]]

weights = [hidden, output]

expected_output1 = [0.1, 1, 0.1]
expected_output2 = [0.0, 1, 0.0]
expected_output3 = [0.0, 0, 0.1]
expected_output4 = [0.1, 1, 0.2]

alpha = 0.01

num_of_iterations = 50

error_acum = 0.0

for i in range(num_of_iterations):
    # print(weights)
    err1 = deep_neural_network(input1, weights, expected_output1, alpha)
    # print(weights)
    # err2 = deep_neural_network(input2, weights, expected_output2, alpha)
    # err3 = deep_neural_network(input3, weights, expected_output3, alpha)
    # err4 = deep_neural_network(input4, weights, expected_output4, alpha)
    #
    # error_acum = sum(err1) + sum(err2) + sum(err3) + sum(err4)

    if i == 49:
        print(err1)