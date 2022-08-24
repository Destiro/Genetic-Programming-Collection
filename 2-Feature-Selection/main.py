## CREDIT: DEAP library template code, assisted in the start of the development in this part of the assignment
## CREDIT: DEAP Documentation

import random
import numpy
import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from sklearn.neighbors import KNeighborsClassifier

# GA Variables
NGEN = 150
ELITISM_POP = 30
CHILDREN = 100
CXPB = 0.65
MUTPB = 0.35


""" Populate GA with Data """
def read_data(file_name):
    data = []
    classes = []
    with open(file_name) as wt:
        for line in wt.readlines():
            classes.append(int(line.rstrip().split(',')[-1]))
            data.append([(float(i)) for i in line.rstrip().split(',')[:-1]])

    return normalise_data(data), classes


def normalise_data(data):
    # find ranges
    ranges = []
    mins = []
    for i in range(len(data[0])):
        max = data[0][i]
        min = data[0][i]
        for j in range(len(data)):
            if data[j][i] > max:
                max = data[j][i]
            if data[j][i] < min:
                min = data[j][i]
        mins.append(min)
        ranges.append(max-min)

    # Apply normalisation
    normalised_data = [[((row[i]-mins[i]) / ranges[i]) for i in range(len(ranges))] for row in data]

    return normalised_data


def column(data, i):
    return [row[i] for row in data]


items, classifiers = read_data("../data/wbcd/wbcd.data")


# TODO FIX
""" Fitness / Evaluation """
def fitnessFunction(individual):
    if len(individual) == 0:
        return 0,

    wrapperFilterFunction(individual)
    return 0,


def wrapperFilterFunction(individual):
    # Creating KNN Model
    neigh = KNeighborsClassifier(n_neighbors=2)
    X, y = []
    for item in items:
        for feature in individual:
            X.append(items[feature]) ## make new item with only features then add it
    y = [i.getClassif() for i in items]
    neigh.fit(X, y)

    # Testing against the Model


    return 0,


""" Crossover """
def setCrossover(ind1, ind2):
    temp = set(ind1)                # Used in order to keep type
    ind1 &= ind2                    # Intersection (inplace)
    ind2 ^= temp                    # Symmetric Difference (inplace)
    return ind1, ind2


""" Mutation """
def setMutation(individual):
    if random.random() < 0.5:
        if len(individual) > 0:     # We cannot pop from an empty set
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(len(items)))
    return individual,


def setupToolbox():
    creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", set, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_item", random.randrange, len(items))

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, len(items[0].getData()))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitnessFunction)
    toolbox.register("mate", setCrossover)
    toolbox.register("mutate", setMutation)
    toolbox.register("select", tools.selNSGA2)
    return toolbox


def plotConvergence(data):
    averages = []

    # Average the data
    for j in range(len(data[0])):
        point = 0
        for i in range(len(data)):
            point += data[i][j]
        averages.append(1514-(point/5))

    plt.plot(averages)
    plt.xlabel('Generation')
    plt.ylabel('Average Value to Optimal (Optimal-Value)')
    plt.title('Convergence for 100_995 (5 Runs)')
    plt.show()


def main():
    toolbox = setupToolbox()
    pop = toolbox.population(n=ELITISM_POP)
    hof = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, ELITISM_POP, CHILDREN, CXPB, MUTPB, NGEN, stats, halloffame=hof)

    best = []
    for record in log:
        best.append(record["max"][0])

    return pop, stats, hof, best

if __name__ == "__main__":
    runs = []

    for i in range(5):
        random.seed(i)
        pop, stats, hof, best = main()
        runs.append(best)
    plotConvergence(runs)