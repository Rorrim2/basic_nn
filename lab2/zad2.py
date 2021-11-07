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

    weights = initial_weights
    for i in range(num_of_iterations):
        error_it = 0.0
        for l in range(len(inputs)):
            a = dot_product(inputs[l], weights)
            prediction = sum(a)
            error = (prediction - goal[l]) * (prediction - goal[l])
            
            delta = prediction - goal[l]
            for j in range(len(weights)):
                weight_delta = inputs[l][j] * delta
                weights[j] = weights[j] - weight_delta * alpha
            error_it += error
        print("Ieration: " + str(i) + ' error: ')
        print(error_it)
    return error_it

weights = [0.1, 0.2, -0.1]
inputs = [[8.5, 0.65, 1.2],
          [9.5, 0.8, 1.3],
          [9.9, 0.8, 0.5],
          [9.0, 0.9, 1.0]]
expected_outputs = [1,1,0,1]
alpha = 0.01

err = neural_network(inputs, weights, expected_outputs, alpha, 2950)

#blad jest mniejszy niz 0.05 dla prawie 3000 iteracji

