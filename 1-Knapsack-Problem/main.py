## CREDIT: DEAP library template code, assisted in the start of the development in this part of the assignment
## CREDIT: DEAP Documentation

import random
import numpy

from KnapItem import KnapItem
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

IND_INIT_SIZE = 5
MAX_ITEM = 50


def read_data(file_name):
    data = []
    index = 0
    with open(file_name) as wt:
        cap = int(wt.readlines()[0].rstrip().split()[1])

    with open(file_name) as wt:
        for line in wt.readlines()[1:]:
            knap = KnapItem(index, (int(line.rstrip().split()[0])), (int(line.rstrip().split()[1])))
            data.append(knap)
            index += 1

    return data, cap


items, capacity = read_data("../data/knapsack-data/10_269")


def evalKnapsack(individual):
    weight = 0.0
    value = 0.0
    for item in individual:
        weight += items[item].getCapacity()
        value += items[item].getValue()
    if len(individual) > MAX_ITEM or weight > capacity:
        return 10000, 0             # Ensure overweighted bags are dominated
    return weight, value


def cxSet(ind1, ind2):
    """Apply a crossover operation on input sets. The first child is the
    intersection of the two sets, the second child is the difference of the
    two sets.
    """
    temp = set(ind1)                # Used in order to keep type
    ind1 &= ind2                    # Intersection (inplace)
    ind2 ^= temp                    # Symmetric Difference (inplace)
    return ind1, ind2


def mutSet(individual):
    """Mutation that pops or add an element."""
    if random.random() < 0.5:
        if len(individual) > 0:     # We cannot pop from an empty set
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(len(items)))
    return individual,


def setupToolbox():
    creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
    creator.create("Individual", set, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_item", random.randrange, len(items))

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_item, IND_INIT_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalKnapsack)
    toolbox.register("mate", cxSet)
    toolbox.register("mutate", mutSet)
    toolbox.register("select", tools.selNSGA2)
    return toolbox

def main():
    random.seed(64)
    NGEN = 50
    MU = 50
    LAMBDA = 100
    CXPB = 0.7
    MUTPB = 0.2

    toolbox = setupToolbox()

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof)

    return pop, stats, hof

if __name__ == "__main__":
    main()