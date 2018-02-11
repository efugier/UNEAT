"""
This module contains:
    - 4 classes: Connexion, Neuron, NeuralNetwork, SpawingPool
    - Shaping function that modify a neural network:
        - newConnexion
        - newRecursiveConnexion
    - save and load functions

TODO:
    - optimise the connexions for the activation_function
    - explicit fitness sharing
        -https://pdfs.semanticscholar.org/b32f/71b58da453f60e63f97a1bc7ae2e717b4ea3.pdf
"""

from copy import deepcopy
from pickle import Pickler, Unpickler

from random import random, choice
from numpy import tanh, exp
from numpy.random import normal


def sigmoid(x):
    return 1 / (1 + exp(-4.9 * x))

# def sigmoid(x):
#     return tanh(3 * x)

# CLASSES


class Connexion:
    """The connexion are the genes from the genetic algorithm point of view"""

    # pylint: disable=too-few-public-methods
    # Using dictionnaries here would a tiny bit more efficient
    # but classes keeps the code clean and homegeneous

    def __init__(self, id_, i, o, w=0):
        """i is the ID of the input neuron
           o is the ID of the output neuron
           w is the weight of the connexion"""
        self.id_ = id_
        self.i = i
        self.o = o
        self.weight = w

        # Allows to desactivate a connexion
        # it is better to desactivate a connexion than removing it
        # in order to keep track of the history of the evolution
        self.is_active = True

    def display(self):
        print("i:", self.i, " weight:", self.weight,
              " o:", self.o, " is active:", self.is_active)


class Neuron:
    """A very basic neuron which has a value, an id, and a list of input"""

    # pylint: disable=too-few-public-methods
    # Using dictionnaries here would a tiny bit more efficient
    # but classes keeps the code clean and homegeneous

    def __init__(self, id_, input_list=None):
        # List of the [ID, weight] of the neurones that are connected to this one
        if not input_list:
            self.input_list = []
        else:
            self.input_list = input_list

        self.value = 0
        self.id_ = id_

        # Important for the evaluation of the network
        self.already_evaluated = False


class NeuralNetwork:
    """A neural network which is a genome from the genetic algorithm point of view"""

    def __init__(self, id_, nb_input, nb_output, activation_funtion=sigmoid):
        self.nb_input = 1 + nb_input  # Adding a bias node
        self.nb_output = nb_output
        self.activation_funtion = activation_funtion

        self.neurons = {}
        self.connexions = {}

        self.id_ = id_

        self.fitness = 0

        # recursive connexion must be treated seprately
        self.recursive_connexions = {}

        self.neurons[0] = Neuron(0)
        self.neurons[0].value = 1  # bias node always set to 1
        self.neurons[0].already_evaluated = True

        # Connecting all the input node to the each output node
        # (Minimal strructure of a NN)
        connexion_id = 0
        for i in range(self.nb_input):
            for o in range(self.nb_input, self.nb_input + self.nb_output):
                self.connexions[connexion_id] = Connexion(
                    connexion_id, i, o, 2 * random() - 1)
                connexion_id += 1

        self.generateNetwork()

    def display(self):
        for c in self.connexions.values():
            c.display()

    # def isForwardOld(self, neuron_id1, neuron_id2):
    #     """Makes sure a feed forward connexion should go from id1 to id2
    #     return True if yes, False if no
    #     O(|connexions|)"""
    #     print("-------")
    #     for c in self.connexions.values():
    #         c.display()
    #     print(neuron_id1, "->", neuron_id2)
    #     for i, _ in self.neurons[neuron_id2].input_list:
    #         if neuron_id2 == i or neuron_id2 in range(self.nb_input) or not self.isForward(neuron_id1, i):
    #             return False

    #     return True

    def generateNetwork(self):
        """generates the neural network using its connexion list
            O(|self.connexions|)"""
        self.neurons = {}

        self.neurons[0] = Neuron(0)
        self.neurons[0].value = 1  # bias node always set to 1
        self.neurons[0].already_evaluated = True

        for e in range(1, self.nb_input):
            self.neurons[e] = Neuron(e)
        for i in range(self.nb_input, self.nb_output):
            self.neurons[i] = Neuron(i)

        for c in self.connexions.values():
            if c.is_active:
                # Checking the existence of the neurons
                if not c.i in self.neurons:  # O(1)
                    self.neurons[c.i] = Neuron(c.i)
                if not c.o in self.neurons:
                    self.neurons[c.o] = Neuron(c.o)

                # Connecting the neurons
                if not c.i in [i for i, _ in self.neurons[c.o].input_list]:
                    self.neurons[c.o].input_list.append([c.i, c.weight])

    def evaluateNeuron(self, neuron_id):
        """Recursive function that evaluates a neuron
           by evluating its conexions"""

        # Pointer for faster access and cleaner code
        neuron = self.neurons[neuron_id]
        if neuron.already_evaluated:
            return neuron.value

        s = neuron.value
        for id_, weight in neuron.input_list:
            s += weight * self.evaluateNeuron(id_)

        neuron.value = self.activation_funtion(s)
        neuron.already_evaluated = True

        return neuron.value

    def evaluateNetwork(self, input_vector):
        """evaluates the neural network
           basically a depth first search
           O(|self.connexions|)"""
        # Setting the input values
        self.neurons[0].value = 1  # Bias node
        self.neurons[0].already_evaluated = True
        for i in range(1, self.nb_input):
            self.neurons[i].value = input_vector[i - 1]
            self.neurons[i].already_evaluated = True

        # Evaluating the NN
        res = [self.evaluateNeuron(o)
               for o in range(self.nb_input, self.nb_input + self.nb_output)]

        # Storing the values that are useful for the recursive connexions
        temp = {}
        for c in self.recursive_connexions.values():
            if not c.o in temp:
                temp[c.o] = 0
            temp[c.o] += c.weight * self.neurons[c.i].value

        # Applying activation function
        for id_ in temp:
            temp[id_] = self.activation_funtion(temp[id_])

        # Reseting the network
        for neuron_id in self.neurons:  # Resets the network
            self.neurons[neuron_id].value = 0
            self.neurons[neuron_id].already_evaluated = False

         # Applying the recursive connexions
        for neuron_id in temp:
            self.neurons[neuron_id].value = temp[neuron_id]

        return res

    def markEvaluationTree(self, neuron_id, necessary_ids: dict):
        """Mark all the neurons that are necessary to the evaluation of neuron_id
           by setting their values in necessary_ids to 1
           O(|connexions|)"""
        necessary_ids[neuron_id] = 1
        for id_, _ in self.neurons[neuron_id].input_list:
            self.markEvaluationTree(id_, necessary_ids)

    def isForward(self, neuron_id1, neuron_id2):
        """Tells wether a connexion from neuron_id1 to neuron_id2 would be
           feed forward, i.e. if id2 is not necessary for the the evaluation of id1
           O(|connexions|)"""
        necessary_ids = {}  # ids of the neurons that are necessary for the evaluation of neuron_id1
        self.markEvaluationTree(neuron_id1, necessary_ids)

        if neuron_id2 in necessary_ids:
            return False
        return True


class SpawingPool:
    """The class which supervises the evolution"""

    # pylint: disable=too-many-instance-attributes
    # This class' very purpose is to encapsulate the evolution's variables

    def __init__(self, nb_input=2, nb_output=1, population_size=150, max_nb_neurons=30,
                 evaluation_function=None):
        self.nb_input = nb_input
        self.nb_output = nb_output
        self.max_nb_neurons = max_nb_neurons

        # A function that can evalutate a LIST of neural network
        # and return a LIST of fitness in the SAME ORDER
        self.evaluation_function = evaluation_function

        self.generation_nb = 0

        self.best_individual = None
        self.max_fitness = 0

        # The goal here is that 2 instances of the same neuron in 2 different NN
        # should have the same id, same for the connexions

        # neuron_catalog[i] contains the id of the neuron
        # that the connexion of id i gave birth to
        self.neuron_catalog = {}
        self.latest_neuron_id = 1 + nb_input + nb_output

        # contains all the connexions that exist in every NN
        self.connexion_catalog = {}
        connexion_id = 0
        for i in range(self.nb_input + 1):  # + 1 for bias node
            for o in range(self.nb_input + 1, self.nb_input + 1 + self.nb_output):
                self.connexion_catalog[connexion_id] = Connexion(
                    connexion_id, i, o, 2 * random() - 1)
                connexion_id += 1
        self.latest_connexion_id = connexion_id

        self.recursive_connexion_catalog = {}
        self.latest_recursive_connexion_id = -1

        # coefficients for the distance calculation
        self.disjoint_coeff = 1
        self.recursive_disjoint_coeff = 1
        self.weight_average_coeff = 1

        self.same_species_threshold = 3

        self.squaring_factor = 1  # high value => all or nothing
        self.scaling_factor = 1

        self.population_size = population_size
        self.population = []

        # Species are lists of individuals
        self.species = []

        # Genetic Algorithm parameters
        self.crossover_rate = 0.75

        self.elimination_rate = 0.4

        self.weight_mutation_proba = 0.8
        self.uniform_perturbation_proba = 0.9

        self.new_connexion_proba = 0.05
        self.new_recursive_connexion_proba = 0
        self.force_input_proba = 0.1

        self.new_neuron_proba = 0.03  # 0.3 if larger population

    def setConnexionId(self, connexion: Connexion):
        """checks if the connexion already exists on in another
           induvidual and updates the ids accrodingly
           O(|connexion_catalog|)"""
        for c in self.connexion_catalog.values():
            if isSameConnexion(connexion, c):
                connexion.id_ = c.id_
                break
        else:
            self.latest_connexion_id += 1
            connexion.id_ = self.latest_connexion_id
            self.connexion_catalog[connexion.id_] = connexion

    def newConnexion(self, nn: NeuralNetwork, force_input=False):
        """Creates a connexion between two unconnected neurons the feed forward way
        force_input forces a connexion from one of the input nodes
        O(|neurons|^2 + |connexions|)"""

        if force_input:
            neuron_id1 = choice(range(nn.nb_input))
            candidates = [id2 for id2 in nn.neurons
                          if id2 != neuron_id1 and not neuron_id1 in nn.neurons[id2].input_list
                          and not id2 in range(nn.nb_input)]
            if candidates:
                neuron_id2 = choice(candidates)
                # No need to check if the connexion is well-oriented

        else:
            neuron_id1 = choice([id1 for id1 in nn.neurons.keys()
                                 if not id1 in range(nn.nb_input, nn.nb_output)])
            candidates = [id2 for id2 in nn.neurons
                          if id2 != neuron_id1 and not neuron_id1 in nn.neurons[id2].input_list
                          and not id2 in range(nn.nb_input)]

            if candidates:
                neuron_id2 = choice(candidates)

                # Making sure the connexion is well oriented
                if not nn.isForward(neuron_id1, neuron_id2):
                    neuron_id1, neuron_id2 = neuron_id2, neuron_id1

        if candidates:
            # Checking if this connexion already exists
            connexion = Connexion(-1, neuron_id1,
                                  neuron_id2, 2 * random() - 1)

            self.setConnexionId(connexion)
            nn.connexions[connexion.id_] = connexion

    def newRecursiveConnexion(self, nn: NeuralNetwork, force_input=False):
        """Creates a connexion between two unconnected neurons the recursive way
        force_input forces a connexion to one of the input nodes
        O(|neurons| + |recursive_connexions| + |recursive_connexion_catalog|)"""

        if force_input:
            neuron_id2 = choice(range(nn.nb_input))
        else:
            neuron_id2 = choice(list(nn.neurons.keys()))

        candidates = []
        for id1 in nn.neurons:
            for c in nn.recursive_connexions.values():
                if not (c.i == id1 and c.o == neuron_id2):
                    candidates.append(id1)

        if candidates:
            neuron_id1 = choice(candidates)

            # Checking if this connexion already exists
            connexion = Connexion(-1, neuron_id1,
                                  neuron_id2, 2 * random() - 1)
            for c in self.recursive_connexion_catalog:
                if isSameConnexion(connexion, c):
                    connexion.id_ = c.id_
                    break
            else:
                self.latest_recursive_connexion_id += 1
                connexion.id_ = self.latest_recursive_connexion_id
                self.recursive_connexion_catalog[connexion.id_] = connexion

            nn.recursive_connexions[connexion.id_] = connexion

    def addNeuron(self, nn: NeuralNetwork):
        """Adds a neuron on a pre-existing connexion:
        o-o => o-o-o
        Disables the old connexion
        O(|connexions|)"""

        candidates = [c for c in nn.connexions.values() if c.is_active]
        connexion = choice(candidates)
        connexion.is_active = False

        new_neuron_id = -1

        # If that connexion already gave birth to a neuron
        if connexion.id_ in self.neuron_catalog:
            new_neuron_id = self.neuron_catalog[connexion.id_]
        else:
            self.latest_neuron_id += 1
            new_neuron_id = self.latest_neuron_id

            # Updating the catalog
            self.neuron_catalog[connexion.id_] = new_neuron_id

        new_connexion1 = Connexion(-1, connexion.i,
                                   new_neuron_id, 2 * random() - 1)
        new_connexion2 = Connexion(-1, new_neuron_id,
                                   connexion.o, 2 * random() - 1)

        self.setConnexionId(new_connexion1)
        self.setConnexionId(new_connexion2)

        nn.connexions[new_connexion1.id_] = new_connexion1
        nn.connexions[new_connexion2.id_] = new_connexion2

    def mutate(self, nn: NeuralNetwork):
        # mutate weights
        if random() < self.weight_mutation_proba:
            c = choice(list(nn.connexions.values()))
            # for c in nn.connexions.values():
            if random() < self.uniform_perturbation_proba:
                r = normal(c.weight)
                c.weight = normal(c.weight)
            else:
                c.weight = 2 * random() - 1

        # mutate add connexion
        if random() < self.new_connexion_proba:
            if random() < self.force_input_proba:
                self.newConnexion(nn, True)
            else:
                self.newConnexion(nn, False)

        if random() < self.new_recursive_connexion_proba:
            if random() < self.force_input_proba:
                self.newRecursiveConnexion(nn, True)
            else:
                self.newRecursiveConnexion(nn, False)

        # mutate add neuron
        if random() < self.new_neuron_proba:
            self.addNeuron(nn)

    def mate(self, new_id_, sp, weak_ids=None):
        """produces a child of 2 neurons from the specy in parameter"""
        if not weak_ids:
            weak_ids = []

        id1 = choice(sp)
        candidates = [id2 for id2 in sp if id2 != id1]
        if not candidates:
            new_nn = deepcopy(self.population[id1])
            new_nn.id_ = new_id_
            self.mutate(new_nn)
            return new_nn

        id2 = choice(candidates)

        # id1 is always the most fit nn
        if self.population[id1].fitness < self.population[id2].fitness:
            id1, id2 = id2, id1

        nn1, nn2 = self.population[id1], self.population[id2]
        new_nn = NeuralNetwork(new_id_, self.nb_input, self.nb_output)

        # Normal connexions
        for id_ in self.connexion_catalog:
            if id_ in nn1.connexions and id_ in nn2.connexions:
                if random() < 0.5:
                    new_nn.connexions[id_] = deepcopy(nn1.connexions[id_])
                else:
                    new_nn.connexions[id_] = deepcopy(nn2.connexions[id_])

                # Activation of the fitest parent to prevent neuron disconnexion (very important)
                if nn1.connexions[id_].is_active:
                    new_nn.connexions[id_].is_active = True

            # Inheriting dijoints genes from the most fit parent
            elif id_ in nn1.connexions:
                new_nn.connexions[id_] = deepcopy(nn1.connexions[id_])

        # Recursive Connexions
        for id_ in self.recursive_connexion_catalog:
            if id_ in nn1.recursive_connexions and id_ in nn2.recursive_connexions:
                if random() < 0.5:
                    new_nn.recursive_connexions[id_] = deepcopy(
                        nn1.recursive_connexions[id_])
                else:
                    new_nn.recursive_connexions[id_] = deepcopy(
                        nn2.recursive_connexions[id_])

            # Inheriting dijoints genes from the most fit parent
            elif id_ in nn1.recursive_connexions:
                new_nn.recursive_connexions[id_] = deepcopy(
                    nn1.recursive_connexions[id_])

        return new_nn

    def buildDistanceMatrix(self):
        """Calculates the distance matrix for the current population
           O(|population|^2 * (|connexion_catalog| + |recursive_connexion_catalog|)"""

        distance_matrix = [[0] * len(self.population)
                           for _ in range(len(self.population))]

        i = 0
        while i < self.population_size:
            j = 0
            while j < i:  # no need to do i=j (=> d = 0)
                d = self.distance(self.population[i], self.population[j])
                distance_matrix[i][j] = d
                distance_matrix[j][i] = d
                j += 1
            i += 1

        return distance_matrix

    def distance(self, nn1, nn2):
        """calculates the distance between two neural networks
           (can be optimize to O(|nn.connexions| + |recursive_connexion_catalog|))
           O(|connexion_catalog| + |recursive_connexion_catalog|)"""
        disjoint_genes_count = 0
        recursive_disjoint_genes_count = 0
        common_gene_count = 0
        average_weight_difference = 0

        # Normal connexions
        for id_ in self.connexion_catalog:
            if id_ in nn1.connexions:
                if id_ in nn2.connexions:
                    average_weight_difference += abs(nn1.connexions[id_].weight -
                                                     nn2.connexions[id_].weight)
                    common_gene_count += 1
                else:
                    disjoint_genes_count += 1

            elif id_ in nn2.connexions:
                disjoint_genes_count += 1

        # Recursive connexions
        for id_ in self.recursive_connexion_catalog:
            if id_ in nn1.recursive_connexions:
                if id_ in nn2.recursive_connexions:
                    average_weight_difference += abs(nn1.connexions[id_] -
                                                     nn2.connexions[id_])
                    common_gene_count += 1
                else:
                    recursive_disjoint_genes_count += 1

            elif id_ in nn2.connexions:
                recursive_disjoint_genes_count += 1

        # Distance calculation
        n = max(1, len(nn1.connexions), len(nn2.connexions))
        n_rec = max(1, len(nn1.recursive_connexions),
                    len(nn2.recursive_connexions))
        disjoint = disjoint_genes_count / n
        recursive_disjoint = recursive_disjoint_genes_count / n_rec

        average_weight_difference /= common_gene_count

        distance = self.disjoint_coeff * disjoint + \
            self.recursive_disjoint_coeff * recursive_disjoint + \
            self.weight_average_coeff * average_weight_difference

        return distance

    def initPopulation(self):
        self.population = [None] * self.population_size
        for id_ in range(self.population_size):
            self.population[id_] = NeuralNetwork(
                id_, self.nb_input, self.nb_output)

    def sharingFunction(self, id1, id2, distance_matrix):
        d = distance_matrix[id1][id2]
        if d < self.same_species_threshold:
            return 1 - (d / self.same_species_threshold) ** self.squaring_factor
        return 0

    def newGeneration(self):
        # calculation of the distance matrix
        distance_matrix = self.buildDistanceMatrix()

        # creation of the species
        self.species = []
        for id_ in range(self.population_size):
            for sp in self.species:
                representative_id = sp[0]
                if distance_matrix[id_][representative_id] < self.same_species_threshold:
                    sp.append(id_)
                    break
            else:
                self.species.append([id_])

        # evaluation of the individuals using fitness sharing
        for sp in self.species:
            raw_fitness_list = self.evaluation_function(
                [self.population[id_] for id_ in sp])
            for i, id_ in enumerate(sp):  # fitness of the nn
                # raw fitness
                f = raw_fitness_list[i]

                if f > self.max_fitness:
                    self.best_individual = deepcopy(self.population[id_])
                    self.max_fitness = f

                # niche count
                m = sum([self.sharingFunction(id_, j, distance_matrix)
                         for j in range(self.population_size)])

                # shared fitness
                shared_fitness = (f ** self.scaling_factor) / m

                self.population[id_].fitness = shared_fitness

        id_list = sorted([id_ for id_ in range(self.population_size)],
                         key=lambda id_: self.population[id_].fitness)

        # individuals that are to be replaced by new ones
        weak_ids = id_list[:int(self.elimination_rate * self.population_size)]

        # elitism
        immune_ids = []
        for sp in self.species:
            if len(sp) >= 5:
                immune_ids.append(
                    max(sp, key=lambda id_: self.population[id_].fitness))

        ids_for_elites = weak_ids[:len(immune_ids)]
        ids_for_new_individuals = weak_ids[len(immune_ids):]

        # Total fitness of each species
        sp_fitness_list = [
            sum([self.population[id_].fitness for id_ in sp]) for sp in self.species]
        total_fitness = sum(sp_fitness_list)

        # number of offsprings per species
        number_offspring_list = [
            int((f / total_fitness) * len(ids_for_new_individuals)) for f in sp_fitness_list]

        # Correction
        corr_number = len(ids_for_new_individuals) - sum(number_offspring_list)
        while corr_number > 0:
            number_offspring_list[corr_number %
                                  len(ids_for_new_individuals)] += 1
            corr_number -= 1

        new_population = [None] * self.population_size

        # elitism
        for i, elite_id in enumerate(immune_ids):
            new_population[ids_for_elites[i]] = self.population[elite_id]

        # new individuals
        j = 0
        for i, sp in enumerate(self.species):
            for _ in range(number_offspring_list[i]):
                new_population[ids_for_new_individuals[j]] = self.mate(
                    ids_for_new_individuals[j], sp)
                j += 1

        # others
        for id_ in range(self.population_size):
            if not new_population[id_]:
                self.mutate(self.population[id_])
                new_population[id_] = self.population[id_]

        self.population = new_population

        for nn in self.population:
            nn.generateNetwork()


# FUNCTIONS

# Shaping functions


def isSameConnexion(c1: Connexion, c2: Connexion):
    if c1.i == c2.i and c1.o == c2.o:
        return True
    return False


# Genetic function


# Service functions


def save(obj, file_name='latest_NN'):
    """Saves an object"""
    with open(file_name, 'wb') as file_:
        pickler = Pickler(file_)
        pickler.dump(obj)

    print("Saved", obj, " as ", file_name)


def load(file_name='latest_NN'):
    """Loads a saved object"""
    with open(file_name, 'rb') as file_:
        unpickler = Unpickler(file_)
        obj = unpickler.load()
    return obj


# Test functions

def evalOR(nn_list):
    pat = [[[0, 0], [0]],
           [[0, 1], [1]],
           [[1, 0], [1]],
           [[1, 1], [1]]]

    res = []
    for nn in nn_list:
        fit = 4 - sum([(nn.evaluateNetwork(i)[0] - o[0])**2 for i, o in pat])
        res.append(fit)

    return res


def printOR(nn: NeuralNetwork):
    pat = [[[0, 0], [0]],
           [[0, 1], [1]],
           [[1, 0], [1]],
           [[1, 1], [1]]]
    for p in pat:
        print(p[0], '->', (nn.evaluateNetwork(p[0])))


def solveOR():
    spawing_pool = SpawingPool(evaluation_function=evalOR)

    spawing_pool.initPopulation()

    for _ in range(10):
        spawing_pool.newGeneration()
        print(spawing_pool.best_individual.fitness)
        # spawing_pool.best_individual.display()
        printOR(spawing_pool.best_individual)


def test():
    nn = NeuralNetwork(0, 2, 1)
    nn.generateNetwork()
    print(len(nn.connexions))
    for c in nn.connexions.values():
        c.display()
    nn.connexions = {0: Connexion(0, 0, 3, -1),
                     1: Connexion(1, 1, 3, 1),
                     2: Connexion(2, 2, 3, 1)}
    nn.generateNetwork()
    printOR(nn)


solveOR()
# test()
