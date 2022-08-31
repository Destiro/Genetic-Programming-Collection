## CREDIT: DEAP library template code, assisted in the start of the development in this part of the assignment
## CREDIT: DEAP Documentation

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from deap.benchmarks.tools import hypervolume

# GA Variables
NGEN = 10
ELITISM_POP = 30
CHILDREN = 100
CXPB = 0.7
MUTPB = 0.25

type_of_filter = 1


""" Populate GA with Data """
def read_data(file_name):
    data = []
    classes = []
    with open(file_name) as wt:
        for line in wt.readlines():
            if ',' in line:
                classes.append(str(line.rstrip().split(',')[-1]))
                data.append([(int(i)) for i in line.rstrip().split(',')[2:-1]])
            else:
                classes.append(str(line.rstrip().split(' ')[-1]))
                data.append([(int(i)) for i in line.rstrip().split(' ')[:-1]])

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

items, classifiers = read_data("../data/vehicle/vehicle.dat")
dataF_items = pd.DataFrame(items)


def getIndivByFeatures(individual):
    list_indiv = [item for item in individual]
    return [[row[i] for i in range(len(items[0])) if i in list_indiv] for row in items]


""" Fitness / Evaluation """
def fitnessFunction(individual):
    if len(individual) == 0:
        return 0,

    # Creating KNN Model
    X = getIndivByFeatures(individual)
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X, classifiers)

    # Testing against the Model
    return (classifiers == neigh.predict(X)).sum() / len(items), len(individual)


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


def runTests(individual):
    X = getIndivByFeatures(individual)
    nb = GaussianNB().fit(X, classifiers)
    predictedModel = nb.predict(X)
    return (classifiers == predictedModel).sum() / len(X)


def setupToolbox():
    creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", set, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_item", random.randrange, len(items[0]))

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, len(items[0]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitnessFunction)
    toolbox.register("mate", setCrossover)
    toolbox.register("mutate", setMutation)
    toolbox.register("select", tools.selNSGA2)
    return toolbox


def plotConvergence(data, dataset, run):
    # Format data for plotting
    x = []
    y = []
    xy = []
    for indiv in data:
        y.append(len(indiv)/len(items[0]))
        x.append(1-runTests(indiv))
        xy.append()

    hv = hypervolume(np.column_stack((x,y)))
    # Plotting data
    plt.scatter(x, y)
    plt.xlabel('Classification Error')
    plt.ylabel('Ratio of Features Selected')
    plt.title(dataset + ' - Run '+str(run)+" - HV = "+str(hv))
    plt.show()


def main():
    toolbox = setupToolbox()
    pop = toolbox.population(n=ELITISM_POP)
    hof = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, ELITISM_POP, CHILDREN, CXPB, MUTPB, NGEN, stats, halloffame=hof)

    return pop, stats, hof

if __name__ == "__main__":
    # Vehicle
    for i in range(3):
        random.seed(i)
        pop, stats, hof = main()
        plotConvergence(hof.items, "Vehicle", i)

    # Clean
    items, classifiers = read_data("../data/musk/clean1.data")
    dataF_items = pd.DataFrame(items)
    for i in range(3):
        random.seed(i)
        pop, stats, hof = main()
        plotConvergence(hof.items, "Clean1", i)



