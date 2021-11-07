def neural_network(input, initial_weight, goal, alpha, num_of_iterations):
    prediction = 0.0
    error = 0.0
    weight = initial_weight

    for i in range(num_of_iterations):
        prediction = input * weight
        error = (prediction - goal) * (prediction - goal)
        delta = prediction - goal
        weight_delta = input * delta
        weight = weight - weight_delta * alpha

    return prediction, error


pred, out = neural_network(2, 0.5, 0.8, 0.1, 16)

print(pred)
print(out)

#przy 7 iteracjach wynik zgadza się dla dwoch miejsc po przecinku,
#przy 12 iteracjach wynik zgadza się dla trzech miejsc po przecinku,
#a przy 16 dla czterech 
#predkosc uczenia zalezy od wspolczynnika alpha, ale im wieksza alpha tym bardzeij spada dokladnosc