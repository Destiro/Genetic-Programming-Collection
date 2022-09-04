## CREDIT: DEAP library template code, assisted in the start of the development in this part of the assignment
## CREDIT: DEAP Documentation

import operator
import random

import numpy
import math

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


POP_SIZE = 100
W = 0.7298  # Intertia - Favours Exploitation
D = 20  # Dimensions
C1 = 1.49618  # Phi1
C2 = 1.49618  # Phi2



creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
    smin=None, smax=None, best=None)


def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, [s * W for s in part.speed], map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif speed > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))


def generate_toolbox(evaluate):
    toolbox = base.Toolbox()
    toolbox.register("particle", generate, size=D, pmin=-30, pmax=30, smin=-10, smax=10)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=C1, phi2=C2)
    toolbox.register("evaluate", evaluate)
    return toolbox


toolbox = generate_toolbox(benchmarks.rosenbrock)


def main():
    pop = toolbox.population(n=POP_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    GEN = 1000
    best = None

    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        # print(logbook.stream)

    return pop, logbook, best

if __name__ == "__main__":
    # First run Rosenbrocks function
    result = []
    for i in range(30):
        pop, log, best = main()
        result.append(benchmarks.rosenbrock(best)[0])
    print("Rosenbrock d=20, Mean results = "+str(numpy.mean(result))+", STD = "+str(numpy.std(result)))

    # Next run Griewank, D=20
    result = []
    for i in range(30):
        toolbox = generate_toolbox(benchmarks.griewank)
        pop, log, best = main()
        result.append(benchmarks.griewank(best)[0])
    print("Griewank d=20, Mean results = " + str(numpy.mean(result)) + ", STD = " + str(numpy.std(result)))

    # Last run Griewank D=50
    result = []
    for i in range(30):
        D = 50
        pop, log, best = main()
        result.append(benchmarks.griewank(best)[0])
    print("Griewank d=50, Mean results = " + str(numpy.mean(result)) + ", STD = " + str(numpy.std(result)))