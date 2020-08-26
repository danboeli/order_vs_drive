import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


class Locations:
    def __init__(self, number_of_locations):
        self.number_of_locations = number_of_locations
        self.locations = np.random.rand(number_of_locations, 2)
        self.logistics_center = np.random.rand(1, 2)
        self.locations = self.locations * 20 - 10
        self.logistics_center = self.logistics_center * 20 - 10


def distance_all_to_mall(locations):
    return 2 * sum(
        np.sqrt(
            locations.locations[:, 0] * locations.locations[:, 0] +
            locations.locations[:, 1] * locations.locations[:, 1]
        )
    )  # euclidean distance of all locations to mall at 0,0 times 2 as each car has to travel to mall and back


def distance_delivery(locations):
    coords_list = [locations.logistics_center, locations.locations]  # todo reformat to make it work
    problem_no_fit = mlrose.TSPOpt(length=locations.number_of_locations + 1, coords=coords_list,
                                   maximize=False)
    best_state, best_fitness = mlrose.genetic_alg(problem_no_fit, mutation_prob=0.2,
                                                  max_attempts=100, random_state=2)
    print(best_state)
    return best_fitness


def run_simulation(number_of_locations, configs):
    simulated_distances = np.zeros((configs, 2))  # distance to mall AND distance delivery
    for i in range(configs):
        my_city = Locations(number_of_locations)
        plt.scatter(my_city.locations[:, 0], my_city.locations[:, 1], marker='o', c='red')
        plt.scatter(my_city.logistics_center[0, 0], my_city.logistics_center[0, 1], marker='D', c='yellow')
        plt.scatter(0, 0, marker='s', c='blue')
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.show()
        simulated_distances[i, 0] = distance_all_to_mall(my_city)
        simulated_distances[i, 1] = distance_delivery(my_city)
        print(simulated_distances[i, 0])
        print(simulated_distances[i, 1])


run_simulation(number_of_locations=3, configs=1)
