import numpy as np
import random

class Genetic:
    def __init__(self, gene_count, population_size, weight, value):
        chromosomes = []
        for i in range(population_size):
            temp = np.random.randint(2, size=gene_count)
            chromosomes.append(temp)
        self.population = np.array(chromosomes)
        self.weights = weight
        self.values = value


    def evolve(self):
        fitness_dictionary = {}
        level_of_adjustment = []
        for i in range(self.population.shape[0]):
            level_of_adjustment.append(self.check_adjustment(self.population[i]))
            fitness_dictionary[self.population[i].tostring()] = \
                self.check_adjustment(self.population[i])

        elite = self.we_are_elite(round(self.population.shape[0] * 0.2), level_of_adjustment)

        self.population = self.roulette_wheel_pop(self.population,
                                                  self.get_probability_list(fitness_dictionary),
                                                  self.population.shape[0])

        self.replicate()

        for i in self.population:
            self.mutate_every(5, i)

        for n in range(round(self.population.shape[0] * 0.2)):
            self.population[n] = elite[n]
        return

    def we_are_elite(self, p, adjustment):
        elite = []
        for n in range(p):
            index = np.argmax(adjustment)
            adjustment[index] = 0
            elite.append(self.population[index])
        return elite

    def roulette_wheel_pop(self, population, probabilities, number):
        chosen = []
        for n in range(number):
            r = random.random()
            for (i, individual) in enumerate(population):
                if r <= probabilities[i]:
                    chosen.append(list(individual))
                    break
        return np.array(chosen)

    def get_probability_list(self, population_fitness_dictionary):
        fitness = population_fitness_dictionary.values()
        total_fit = float(sum(fitness))
        relative_fitness = [f / total_fit for f in fitness]
        probabilities = [sum(relative_fitness[:i + 1])
                         for i in range(len(relative_fitness))]
        return probabilities

    def check_adjustment(self, chromosome):
        len = chromosome.shape[0]
        value = 0
        weight = 0
        for n in range(len):
            value += chromosome[n] * self.values[n]
            weight += chromosome[n] * self.weights[n]
        if weight > 35:
            value = 0
        return value

    def check_state(self):
        adjust = np.zeros(self.population.shape[0])
        for i in range(self.population.shape[0]):
            adjust[i] = self.check_adjustment(self.population[i])

        return np.sum(adjust) / 8

    def mutate(self, probability, chromosome):
        if np.random.choice(10) <= (probability/10):
            chromosome[np.random.choice(chromosome.shape[0])] \
                = int(not chromosome[np.random.choice(chromosome.shape[0])])

    def mutate_every(self, probability, chromosome):
        for n in range(chromosome.shape[0]):
            if np.random.choice(100) <= probability:
                chromosome[n] = int(not chromosome[n])

    def replicate(self):
        np.random.shuffle(self.population)
        part = self.population.shape[0] - 1
        children = []
        for i in range(part):
            child, _ = self.replicate_in_half(self.population[i], self.population[i + 1])
            children.append(child)
        child, _ = self.replicate_in_half(self.population[0], self.population[part])
        children.append(child)
        for i in range(part + 1):
            self.population[i] = children[i]


    def replicate_in_half(self, parent_x, parent_y):
        part = int(parent_x.shape[0] / 2)

        child_x = np.concatenate((parent_x[0:part], parent_y[part:parent_x.shape[0]]), axis=0)
        child_y = np.concatenate((parent_y[0:part], parent_x[part:parent_x.shape[0]]), axis=0)
        return child_x, child_y

    def replicate_in_random_spot(self, parent_x, parent_y):
        part = np.random.choice(parent_x.shape[0])
        child_x = np.concatenate((parent_x[0:part], parent_y[part:10]), axis=0)
        child_y = np.concatenate((parent_y[0:part], parent_x[part:10]), axis=0)
        return child_x, child_y

    def get_population(self):
        return self.population

gen = Genetic(10, 8, np.array([3, 13, 10, 9, 7, 1, 8, 8, 2, 9]),
              np.array([266, 442, 671, 526, 388, 245, 210, 145, 126, 322]))

generations = 500

for i in range(generations):
    gen.evolve()

    print(gen.check_state())

print(gen.get_population())