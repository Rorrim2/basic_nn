import numpy as np


class Genetic:
    def __init__(self, gene_count, population_size):
        chromosomes = []
        for i in range(population_size):
            temp = np.random.randint(2, size=10)
            chromosomes.append(temp)
        self.population = np.array(chromosomes)


    def evolve(self):
        done = self.check_end_condition(10,10)
        if done:
            return done
        adjust_max = np.zeros(10)
        for i in range(self.population.shape[0]):
            adjust_max[i] = self.check_adjustment(self.population[i])

        adjust_min = adjust_max.copy()

        index_x_max = np.argmax(adjust_max)
        adjust_max[index_x_max] = 0
        index_y_max = np.argmax(adjust_max)

        index_x_min = np.argmin(adjust_min)
        adjust_min[index_x_min] = 10
        index_y_min = np.argmin(adjust_min)

        child_x, child_y = self.replicate_in_random_spot(self.population[index_x_max],
                                                  self.population[index_y_max])
        self.population[index_x_min] = child_x
        self.population[index_y_min] = child_y

        self.mutate(60, self.population[index_x_max])
        self.mutate(60, self.population[index_y_max])

        return done


    def check_adjustment(self, chromosome):
        return np.sum(chromosome)

    def check_end_condition(self, end_condition_population, end_condition_chromosome):
        count = 0
        for i in self.population:
            temp = self.check_adjustment(i)
            if temp >= end_condition_chromosome:
                count += 1
        done = False
        if count >= end_condition_population:
            done = True
        return done

    def check_state(self):
        adjust_max = np.zeros(10)
        for i in range(self.population.shape[0]):
            adjust_max[i] = self.check_adjustment(self.population[i])
        return np.sum(adjust_max)

    def mutate(self, probability, chromosome):
        if np.random.choice(10) <= (probability/10):
            chromosome[np.random.choice(10)] = int(not chromosome[np.random.choice(10)])

    def replicate_in_half(self, parent_x, parent_y):
        child_x = np.concatenate((parent_x[0:5], parent_y[5:10]), axis=0)
        child_y = np.concatenate((parent_y[0:5], parent_x[5:10]), axis=0)
        return child_x, child_y

    def replicate_in_random_spot(self, parent_x, parent_y):
        part = np.random.choice(10)
        child_x = np.concatenate((parent_x[0:part], parent_y[part:10]), axis=0)
        child_y = np.concatenate((parent_y[0:part], parent_x[part:10]), axis=0)
        return child_x, child_y

    def get_population(self):
        return self.population

gen = Genetic(10, 10)

generations = 100

for i in range(generations):
    gen.evolve()

    print(gen.check_state())

print(gen.get_population())

