## CREDIT: DEAP library template code, assisted in the start of the development in this part of the assignment
## CREDIT: DEAP Documentation

import numpy
import functions
import math
import operator
import random

from deap import algorithms, base, creator, tools, gp



def addPrimitives():
    primitives = gp.PrimitiveSet("MAIN", 1)
    primitives.addPrimitive(functions.add, 2)
    primitives.addPrimitive(functions.sub, 2)
    primitives.addPrimitive(functions.mul, 2)
    primitives.addPrimitive(functions.div, 2)
    primitives.addPrimitive(functions.abs, 1)
    primitives.addPrimitive(functions.neg, 1)
    primitives.addPrimitive(math.sin, 1)
    primitives.addPrimitive(math.cos, 1)
    primitives.addEphemeralConstant("terminals", lambda: random.randint(-10, 10))
    primitives.renameArguments(ARG0='x')
    return primitives


def test_point(point):
    if point <= 0:
        return (2*point) + (point*point) + 3
    else:
        return 1/point + math.sin(point)


def fitness_function(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    # Fitness function = the absolute value of each y-x
    sqerrors = (operator.abs(func(x) - test_point(x)) for x in points)
    return math.fsum(sqerrors) / len(points),


def generate_toolbox():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    tb = base.Toolbox()
    tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    tb.register("individual", tools.initIterate, creator.Individual, tb.expr)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("compile", gp.compile, pset=pset)

    tb.register("evaluate", fitness_function, points=[x/5. for x in range(-200, 200)])  # every 0.25 from -20 to 20
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
    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    #Pop, Toolbox, Cross_prob, mutation_prob, num_generations
    pop, log = algorithms.eaSimple(pop, toolbox, 0.65, 0.15, 150, stats=mstats,
                                   halloffame=hof, verbose=False)
    # print log
    return pop, log, hof


if __name__ == "__main__":
    for i in range(3):
        random.seed(i)
        popx, logx, hofx = main()
        print(str(i)+": Best Tree: ", hofx[0])
        print(str(i)+": Best Fitness: ", hofx[0].fitness.values[0])
