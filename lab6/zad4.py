import numpy as np
import random

class Genetic:
    def __init__(self, gene_count, population_size, x, y):
        chromosomes = []
        for i in range(population_size):
            temp = np.arange(25)
            np.random.shuffle(temp)
            chromosomes.append(temp)
        self.population = np.array(chromosomes)
        self.ix = x
        self.iy = y


    def evolve(self):
        fitness_dictionary = {}
        level_of_adjustment = []
        for i in range(self.population.shape[0]):
            level_of_adjustment.append(self.check_adjustment(self.population[i]))
            fitness_dictionary[self.population[i].tostring()] = \
                (-1) * self.check_adjustment(self.population[i])

        elite = self.we_are_elite(round(self.population.shape[0] * 0.2), level_of_adjustment)

        self.population = self.roulette_wheel_pop(self.population,
                                                  self.get_probability_list(fitness_dictionary),
                                                  self.population.shape[0])
        self.replicate()

        for i in self.population:
            self.mutate_every(2, i)

        for n in range(round(self.population.shape[0] * 0.2)):
            np.random.shuffle(self.population)
            self.population[n] = elite[n]

        return

    def we_are_elite(self, p, adjustment):
        elite = []
        for n in range(p):
            index = np.argmin(adjustment)
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
        dist = 0
        x1 = self.ix[chromosome[0]]
        y1 = self.iy[chromosome[0]]

        x2 = self.ix[chromosome[len - 1]]
        y2 = self.iy[chromosome[len - 1]]
        dist += self.get_dist(x1, x2, y1, y2)

        for n in range(len - 1):
            x1 = self.ix[chromosome[n]]
            y1 = self.iy[chromosome[n]]

            x2 = self.ix[chromosome[n+1]]
            y2 = self.iy[chromosome[n+1]]
            dist += self.get_dist(x1, x2, y1, y2)

        return dist

    def get_dist(self, x1, x2, y1, y2):
        return np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

    def check_state(self):
        last_max = 0

        for i in range(self.population.shape[0]):
            if i == 0:
                last_max = self.check_adjustment(self.population[i])
            else:
                if self.check_adjustment(self.population[i]) < last_max:
                    last_max = self.check_adjustment(self.population[i])
        return last_max

    def mutate(self, probability, chromosome):
        if np.random.choice(10) <= (probability/10):
            chromosome[np.random.choice(chromosome.shape[0])] \
                = int(not chromosome[np.random.choice(chromosome.shape[0])])

    def mutate_every(self, probability, chromosome):
        for n in range(chromosome.shape[0] - 1):
            if np.random.choice(100) <= probability:
                m = np.random.choice(round(chromosome.shape[0]/2))
                # m = chromosome.shape[0] - n - 1
                temp = chromosome[n]
                chromosome[n] = chromosome[m]
                chromosome[m] = temp

    def replicate(self):
        np.random.shuffle(self.population)
        part = self.population.shape[0] - 1
        children = []
        for i in range(part):
            child1, child2 = self.ordered_crossover(self.population[i], self.population[i + 1])
            children.append(child1)
            children.append(child2)
        child1, child2 = self.ordered_crossover(self.population[0], self.population[part])
        children.append(child1)
        children.append(child2)
        np.random.shuffle(children)
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

    def ordered_crossover(self, parentX, parentY):
        start, end = 0, 0
        childX = np.full(parentX.shape[0], -1)
        childY = np.full(parentX.shape[0], -1)
        while (np.abs(start - end) < 5 or np.abs(start - end) > 15):
            start, end = sorted([random.randrange(parentX.shape[0]) for _ in range(2)])

        childX[start:end] = parentX[start:end]
        childY[start:end] = parentY[start:end]
        i, j = 0, 0
        cpX, cpY = 0, 0
        tempX, tempY = 0, 0
        if start != 0:
            while (cpX != start or cpY != start):
                tempX = parentX[i]
                tempY = parentY[j]
                if cpY != start and not np.any(childX[:] == tempY):
                    childX[cpY] = tempY
                    cpY += 1
                if cpX != start and not np.any(childY[:] == tempX):
                    childY[cpX] = tempX
                    cpX += 1
                if cpY != start:
                    j += 1
                if cpX != start:
                    i += 1
        cpY, cpX = end, end
        while (cpX != parentY.shape[0] or cpY != parentY.shape[0]):
            tempX = parentX[i]
            tempY = parentY[j]
            if cpY != parentY.shape[0] and not np.any(childX[:] == tempY):
                childX[cpY] = tempY
                cpY += 1
            if cpX != parentY.shape[0] and not np.any(childY[:] == tempX):
                childY[cpX] = tempX
                cpX += 1
            if cpY != parentY.shape[0]:
                j += 1
            if cpX != parentY.shape[0]:
                i += 1
        return childX, childY

    def get_population(self):
        return self.population

gen = Genetic(10, 100, np.array([119, 37, 197, 85, 12, 100, 81, 121, 85, 80,
                                 91, 106, 123, 40, 78, 190, 187, 37, 17, 67,
                                 78, 87, 184, 111, 66]),
              np.array([38, 38, 55, 165, 50, 53, 142, 137, 145, 197,
                        176, 55, 57, 81, 125, 46, 40, 107, 11, 56,
                        133, 23, 197, 12, 178]))

generations = 900

for i in range(generations):
    gen.evolve()

    print(gen.check_state())

f = open("pop.txt", "w")

for chromosome in gen.get_population():
    for gene in chromosome:
        f.write(str(gene))
        f.write(" ")
    f.write('\n')


# print()