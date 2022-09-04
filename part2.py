import random
import time

import numpy
from deap import creator, base, tools, algorithms
from sklearn import neighbors
from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import GaussianNB

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)
instances = []
instance_classes = []
num_features = 0


def getData(fileName):
    global instances, instance_classes, num_features
    file_data = []
    instance_classes = []
    for line in open(fileName, "r"):
        line_data = list(map(float, line.strip().split(",")))
        file_data.append(line_data[:-1])
        instance_classes.append(line_data[-1])

    num_features = len(file_data[0])

    # Normalise data
    features = [[instance[a] for instance in file_data] for a in range(num_features)]
    feature_mins = [min(features[a]) for a in range(num_features)]
    feature_ranges = [max(features[a]) - feature_mins[a] for a in range(num_features)]

    instances = [[((instance[a] - feature_mins[a]) / feature_ranges[a]) for a in range(num_features)]
                 for instance in file_data]


def transform_instances(individual):
    return [[instance[a] for a in range(num_features) if individual[a] == 1] for instance in instances]


def filter_evaluate(individual):
    transformed_instances = transform_instances(individual)
    return sum(mutual_info_classif(transformed_instances, instance_classes, discrete_features=False)),


def wrapper_evaluate(individual):
    if sum(individual) == 0:
        return 0

    transformed_instances = transform_instances(individual)
    classifier = neighbors.KNeighborsClassifier().fit(transformed_instances, instance_classes)
    return classifier.score(transformed_instances, instance_classes),


def runGA(filename, evaluate):
    getData(filename)

    toolbox = base.Toolbox()
    toolbox.register("attr_binary", random.choice, [0, 1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_binary, num_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selRoulette)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)

    # GA Parameters
    num_gens = 50
    pop_size = 100
    elite_num = round(0.05 * pop_size)
    crossover_prob = 0.85
    mutation_prob = 0.15

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    algorithms.eaMuPlusLambda(pop, toolbox, elite_num, pop_size, crossover_prob, mutation_prob, num_gens,
                              halloffame=hof, verbose=False)

    return hof[0]


def naive_bayes(individual):
    transformed_instances = transform_instances(individual)
    classifier = GaussianNB()
    classifier.fit(transformed_instances, instance_classes)
    return classifier.score(transformed_instances, instance_classes)


def test_fitness_function(func, data):
    times = []
    results = []
    for i in range(5):
        start_time = time.perf_counter()
        results.append(runGA(data, func))
        times.append(time.perf_counter() - start_time)
        print("Run {} complete, time taken: {}s".format(i + 1, times[i]))

    accuracies = [naive_bayes(individual) for individual in results]
    return numpy.mean(accuracies), numpy.std(accuracies), numpy.mean(times), numpy.std(times)


for file in ["wbcd/wbcd.data", "sonar/sonar.data"]:
    print("\n"+file+":")
    print("\nRunning GA with filter-based fitness function 5 times...")
    filter_mean_acc, filter_std_acc, filter_mean_time, filter_std_time = \
        test_fitness_function(filter_evaluate, file)

    print("\nRunning GA with wrapper-based fitness function 5 times...")
    wrapper_mean_acc, wrapper_std_acc, wrapper_mean_time, wrapper_std_time = \
        test_fitness_function(wrapper_evaluate, file)

    print("\nFilter-based fitness function:")
    print("Mean Computation Time = {}\nComputation Time STD = {}".format(filter_mean_time, filter_std_time))
    print("Mean Classification Accuracy = {}\nClassification Accuracy STD = {}".format(filter_mean_acc, filter_std_acc))

    print("\nWrapper-based fitness function:")
    print("Mean Computation Time = {}\nComputation Time STD = {}".format(wrapper_mean_time, wrapper_std_time))
    print("Mean Classification Accuracy = {}\nClassification Accuracy STD = {}".format(wrapper_mean_acc, wrapper_std_acc))
