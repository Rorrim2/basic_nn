import numpy as np
import random

class Genetic:
    def __init__(self, gene_count, population_size):
        chromosomes = []
        for i in range(population_size):
            temp = np.random.randint(2, size=gene_count)
            chromosomes.append(temp)
        self.population = np.array(chromosomes)


    def evolve(self):
        done = self.check_end_condition(3,0)
        if done:
            return done
        fitness_dictionary = {}
        for i in range(self.population.shape[0]):
            fitness_dictionary[self.population[i].tostring()] = \
                (-1) * self.check_adjustment(self.population[i])

        self.population = self.roulette_wheel_pop(self.population,
                                                  self.get_probability_list(fitness_dictionary),
                                                  self.population.shape[0])


        self.replicate()

        for i in self.population:
            self.mutate(10, i)

        return done

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
        a = self.get_a_as_int(chromosome)
        b = self.get_b_as_int(chromosome)
        res = 2 * a * a + b - 33
        if res < 0:
            res = res * 10
        return np.abs(res)

    def check_end_condition(self, end_condition_population, end_condition_chromosome):
        count = 0
        for i in self.population:
            temp = self.check_adjustment(i)
            if temp == end_condition_chromosome:
                count += 1
        done = False
        if count >= end_condition_population:
            done = True
        return done

    def check_state(self):
        adjust = np.zeros(self.population.shape[0])
        for i in range(self.population.shape[0]):
            adjust[i] = self.check_adjustment(self.population[i])
        return np.sum(adjust)

    def mutate(self, probability, chromosome):
        if np.random.choice(10) <= (probability/10):
            chromosome[np.random.choice(chromosome.shape[0])] \
                = int(not chromosome[np.random.choice(chromosome.shape[0])])

    def replicate(self):
        np.random.shuffle(self.population)
        part = int(self.population.shape[0] / 2) - 1
        children = []
        for i in range(part):
            child, _ = self.replicate_in_half(self.population[i], self.population[i + 1])
            children.append(child)
        child, _ = self.replicate_in_half(self.population[0], self.population[part + 1])
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

    def get_a_as_int(self, chromosome):
        ret = 0
        for i in range(4):
            ret = ret + (chromosome[i] * np.power(2, (3 - i)))
        ret = ret
        return ret

    def get_b_as_int(self, chromosome):
        ret = 0
        for i in range(4):
            ret = ret + (chromosome[i + 4] * np.power(2, (3 - i)))
        ret = ret
        return ret

    def get_population(self):
        return self.population

gen = Genetic(8, 10)

generations = 0

done = False

while not done:
    print(generations)
    done =  gen.evolve()
    generations += 1


for i in range(gen.get_population().shape[0]):
    print(gen.get_a_as_int(gen.get_population()[i]))
    print(gen.get_b_as_int(gen.get_population()[i]))
    print(" ")

