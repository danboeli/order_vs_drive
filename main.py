#  import mlrose
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


class Locations:
    def __init__(self, number_of_locations):
        self.locations = np.random.rand(number_of_locations, 2)
        self.logistics_center = np.random.rand(1, 2)
        self.locations = self.locations * 20 - 10
        self.logistics_center = self.logistics_center * 20 - 10


def distance_all_to_mall(locations):
    return 0


def distance_delivery(locations):
    return 0


def run_simulation(number_of_locations, configs):
    simulated_distances = np.zeros((configs, 2))  # distance to mall AND distance delivery
    for i in range(configs):
        my_city = Locations(number_of_locations)
        plt.scatter(my_city.locations[:, 0], my_city.locations[:, 1], marker='o', c='red')
        plt.scatter(my_city.logistics_center[0, 0], my_city.logistics_center[0, 1], marker='D', c='yellow')
        plt.scatter(0, 0, marker='s', c='blue')
        plt.show()
        simulated_distances[i, 0] = distance_all_to_mall(my_city)
        simulated_distances[i, 1] = distance_delivery(my_city)


run_simulation(number_of_locations=10, configs=2)
