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

def neural_network(inputs, initial_weights, goal, alpha, num_of_iterations):
    prediction = 0.0
    error_it = 0.0

    #init output array
    output = []
    for i in range(len(inputs)):
        output_r = []
        for j in range(len(inputs[i])):
            output_r.append(0)
        output.append(output_r)

    weights = initial_weights
    for i in range(num_of_iterations):
        error_it = 0
        for l in range(len(inputs)):
            error_row = []
            for w in range(len(weights)):
                a = dot_product(inputs[l], weights[w])
                prediction = sum(a)
                output[l][w] = prediction

                error = (prediction - goal[l][w]) * (prediction - goal[l][w])
                error_row.append(error)
                delta = prediction - goal[l][w]
                for j in range(len(weights[w])):
                    weight_delta = inputs[l][j] * delta
                    weights[w][j] = weights[w][j] - weight_delta * alpha
            error_it += sum(error_row)
    return output, error_it

inputs = [[8.5, 0.65, 1.2],
          [9.5, 0.8, 1.3],
          [9.9, 0.8, 0.5],
          [9.0, 0.9, 1.0]]

weights = [[0.1, 0.1, -0.3],
           [0.1, 0.2, 0.0],
           [0.0, 1.3, 0.1]]
alpha = 0.01

expected_outputs = [[0.1, 1, 0.1],
                    [0.0, 1, 0.0],
                    [0.0, 0, 0.1],
                    [0.1, 1, 0.2]]

pred, err = neural_network(inputs, weights, expected_outputs, alpha, 10000)

print("Pred: ")
print(pred)
print("Err: ")
print(err)

#blad zatrzymuje sie na poziomie okolo 0.11