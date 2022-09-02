## CREDIT: DEAP library template code, assisted in the start of the development in this part of the assignment
## CREDIT: DEAP Documentation

import numpy
import functions
import math
import operator
import random

from deap import algorithms, base, creator, tools, gp


def read_data(file_name):
    data = []

    with open(file_name) as wt:
        for line in wt.readlines()[2:]:
            data.append((float(line.rstrip().split()[0]), float(line.rstrip().split()[-1])))

    return data


def addPrimitives():
    primitives = gp.PrimitiveSet("MAIN", 1)
    primitives.addPrimitive(functions.add, 2)
    primitives.addPrimitive(functions.sub, 2)
    primitives.addPrimitive(functions.mul, 2)
    primitives.addPrimitive(functions.div, 2)
    primitives.addPrimitive(functions.abs, 1)
    primitives.addPrimitive(functions.neg, 1)
    primitives.addPrimitive(math.cos, 1)
    primitives.addPrimitive(math.sin, 1)
    primitives.addPrimitive(math.tan, 1)
    primitives.addEphemeralConstant("terminals", lambda: random.randint(-20, 20))
    primitives.renameArguments(ARG0='x')
    return primitives


def fitness_function(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    # Fitness function = the absolute value of each y-x
    sqerrors = (operator.abs(x[1]-func(x[0])) for x in points)
    return math.fsum(sqerrors) / len(points),


def generate_toolbox():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    tb = base.Toolbox()
    tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    tb.register("individual", tools.initIterate, creator.Individual, tb.expr)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("compile", gp.compile, pset=pset)

    tb.register("evaluate", fitness_function, points=read_data("regression.txt"))
    tb.register("select", tools.selTournament, tournsize=3)
    tb.register("mate", gp.cxOnePoint)
    tb.register("expr_mut", gp.genFull, min_=0, max_=2)
    tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=pset)

    tb.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    tb.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    return tb


pset = addPrimitives()
toolbox = generate_toolbox()


def main():
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    #Pop, Toolbox, Cross_prob, mutation_prob, num_generations
    pop, log = algorithms.eaSimple(pop, toolbox, 0.7, 0.1, 100, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof


popx, logx, hofx = main()
print("Best Tree: ", hofx[0])
print("Best Fitness: ", hofx[0].fitness.values[0])
