import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import matplotlib.pyplot as plt
import time

plt.style.use('seaborn-whitegrid')


class Locations:
    def __init__(self, number_of_locations, city_type='square'):
        self.number_of_locations = number_of_locations
        self.city_type = city_type
        self.locations = np.zeros((number_of_locations, 2))
        self.logistics_center = np.zeros((1, 2))
        self.coords_list = np.zeros((number_of_locations + 1, 2))

    def __call__(self, solution):
        self.coords_list = self.coords_list[solution, :]

    def update(self):
        # Options are square or circle. Note that for circle, the locations are evenly distributed along
        # the radius. This means higher density for small radii, which might be a good assumption
        # in a city
        rand_locations = np.random.rand(self.number_of_locations, 2)
        rand_logistics_center = np.random.rand(1, 2)
        if self.city_type == 'square':
            self.locations = rand_locations * 20 - 10
        else:
            if self.city_type == 'circular':
                rand_locations[:, 0] = rand_locations[:, 0] * 10 * 4 / np.pi  # Equalize the area of the
                # circle to the square
            elif self.city_type == 'equalized_circular':
                rand_locations[:, 0] = np.sqrt(rand_locations[:, 0]) * 10 * 4 / np.pi  # Equalize the area of the
                # circle to the square
            else:
                print('Not an supported city type.')
                assert False
            rand_locations[:, 1] = rand_locations[:, 1] * 2 * np.pi
            self.locations[:, 0] = rand_locations[:, 0] * np.cos(rand_locations[:, 1])
            self.locations[:, 1] = rand_locations[:, 0] * np.sin(rand_locations[:, 1])

        self.logistics_center = rand_logistics_center * 20 - 10
        self.coords_list = np.concatenate((self.logistics_center, self.locations))


def distance_all_to_mall(locations):
    return 2 * sum(
        np.sqrt(
            locations.locations[:, 0] * locations.locations[:, 0] +
            locations.locations[:, 1] * locations.locations[:, 1]
        )
    )  # euclidean distance of all locations to mall at 0,0 times 2 as each car has to travel to mall and back


def distance_delivery(locations):
    problem_no_fit = mlrose.TSPOpt(length=locations.number_of_locations + 1, coords=locations.coords_list,
                                   maximize=False)
    best_state, best_fitness = mlrose.genetic_alg(problem_no_fit, mutation_prob=0.2,
                                                  max_attempts=100, random_state=2)
    locations(best_state)
    return best_fitness


def create_plot(locations, i, time, distance):
    plt.scatter(locations.locations[:, 0], locations.locations[:, 1], marker='o', c='red')
    plt.scatter(locations.logistics_center[0, 0], locations.logistics_center[0, 1], marker='D', c='yellow')
    plt.scatter(0, 0, marker='s', c='blue')
    plt.xlim(-13, 13)
    plt.ylim(-13, 13)
    [plt.annotate(x[0], (x[1], x[2])) for x in np.array([range(1, locations.number_of_locations + 1),
                                                         locations.locations[:, 0], locations.locations[:, 1]]).T]
    plt.annotate('Central Mall', (0, 0))
    plt.annotate('Logistics center 0', (locations.logistics_center[0, 0], locations.logistics_center[0, 1]))
    plt.plot((locations.coords_list[-1, 0], locations.coords_list[0, 0]),
             (locations.coords_list[-1, 1], locations.coords_list[0, 1]), 'b--')
    plt.plot(locations.coords_list[:, 0], locations.coords_list[:, 1], 'b--')
    plt.text(-12, 12, 'TSP Distance: {}, Pick-up Distance: {}'.format(distance[1], distance[0]))
    filename = 'time_{}_config_{}.png'.format(time, i)
    plt.savefig(filename)
    plt.close()


def run_simulation(number_of_locations, configs, city_type, now_time):
    simulated_distances = np.zeros((configs, 2))  # distance to mall AND distance delivery
    my_city = Locations(number_of_locations, city_type)

    for i in range(configs):
        my_city.update()

        simulated_distances[i, 0] = distance_all_to_mall(my_city)
        simulated_distances[i, 1] = distance_delivery(my_city)

        create_plot(my_city, i, now_time, simulated_distances[i, :])

    return simulated_distances


def create_stats(number_of_locations, configs, city_type, simulated_distances, time):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.violinplot([simulated_distances[:, 0], simulated_distances[:, 1]], [1, 2],
                  showmeans=True, showextrema=True, showmedians=True, bw_method=0.5)
    ax.set_title('Simulation Result')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Pick-up stuff', 'Order stuff'])
    ax.set_xlim(0, 3)
    ax.set_xlabel('')
    ax.set_ylabel('Distance traveled')
    ax.text(0.1, max(simulated_distances[:, 0] + 0.5),
            'Locations: {}, Arrangement: {}, Samples: {}'.format(number_of_locations, city_type, configs))
    filename = 'stats_time_{}.png'.format(time)
    plt.savefig(filename)
    plt.close()


now_time = time.time()
number_of_locations = 10
configs = 50
city_type = 'equalized_circular'
simulated_distances = run_simulation(number_of_locations, configs, city_type, now_time)
create_stats(number_of_locations, configs, city_type, simulated_distances, now_time)
