from copy import deepcopy
from random import random, choice
from numpy.random import normal
from neuralnetwork import NeuralNetwork
from connexion import Connexion
from species import Species

from parameters import *


class SpawingPool:
    """The class which supervises the evolution"""

    def __init__(self, population_size=150, max_nb_neurons=30, evaluation_function=None):
        self.population_size = population_size
        self.max_nb_neurons = max_nb_neurons

        # A function that can evalutate a LIST of neural network
        # and return a LIST of fitness in the SAME ORDER
        self.evaluation_function = evaluation_function

        # The goal here is that 2 instances of the same neuron in 2 different NN
        # should have the same id, same for the connexions

        # neuron_catalog[i] contains the id of the neuron
        # that the connexion of id i gave birth to
        self.neuron_catalog = {}
        self.latest_neuron_id = 1 + nb_input + nb_output

        # contains all the connexions that exist in every NN
        # referenced by their UNIQUE id
        self.connexion_catalog = {}
        connexion_id = 0
        for i in range(nb_input + 1):  # + 1 for bias node
            for o in range(nb_input + 1, nb_input + 1 + nb_output):
                self.connexion_catalog[connexion_id] = Connexion(
                    connexion_id, i, o, 2 * random() - 1)
                connexion_id += 1
        self.latest_connexion_id = connexion_id

        self.recursive_connexion_catalog = {}
        self.latest_recursive_connexion_id = -1

        # Genectic attributes
        self.population_size = population_size
        self.population = []
        self.generation_nb = 0
        self.stagnation = 0

        self.best_individual = None
        self.max_fitness = 0

        # Species are lists of individuals
        self.species = []

    def getIndividual(self, id_):
        return self.population[id_]

    def initPopulation(self):
        self.population = [None] * self.population_size
        for id_ in range(self.population_size):
            self.population[id_] = NeuralNetwork(
                id_, nb_input, nb_output)

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
        if random() < weight_mutation_proba:
            c = choice(list(nn.connexions.values()))
            # for c in nn.connexions.values():
            if random() < uniform_perturbation_proba:
                r = normal(c.weight)
                c.weight = r
            else:
                c.weight = 2 * random() - 1

        # mutate add connexion
        if random() < new_connexion_proba:
            if random() < force_input_proba:
                self.newConnexion(nn, True)
            else:
                self.newConnexion(nn, False)

        if random() < new_recursive_connexion_proba:
            if random() < force_input_proba:
                self.newRecursiveConnexion(nn, True)
            else:
                self.newRecursiveConnexion(nn, False)

        # mutate add neuron
        if random() < new_neuron_proba:
            self.addNeuron(nn)

    def newGeneration(self):
        # calculation of the distance matrix
        distance_matrix = buildDistanceMatrix(self.population)

        # creation of the species
        for id_ in range(self.population_size):
            for sp in self.species:
                if distance_matrix[id_][sp.representative_id] < same_species_threshold:
                    sp.members_ids.append(id_)
                    break
            else:
                sp = Species([id_])
                sp.representative_id = id_
                self.species.append(sp)

        # evaluation of the individuals using fitness sharing
        for sp in self.species:
            raw_fitness_list = self.evaluation_function(
                [self.population[id_] for id_ in sp.members_ids])
            for i, id_ in enumerate(sp.members_ids):  # fitness of the nn
                # raw fitness
                f = raw_fitness_list[i]

                if f > self.max_fitness:
                    self.best_individual = deepcopy(self.population[id_])
                    self.max_fitness = f

                # niche count
                m = sum([sharingFunction(id_, j, distance_matrix)
                         for j in range(self.population_size)])

                # shared fitness
                shared_fitness = (f ** scaling_factor) / m

                self.population[id_].fitness = shared_fitness

        # Stagnation and potential
        for sp in self.species:
            best = self.population[max(
                sp.members_ids, key=lambda id_: self.population[id_].fitness)]
            if (self.population[sp.representative_id].fitness - best.fitness) / best.fitness > 0.1:
                sp.stagnation = 0
            else:
                sp.stagnation += 1
            if sp.stagnation > max_stagnation:
                sp.is_stale = True
                print("********removing species")

        self.species = [sp for sp in self.species if not sp.is_stale]

        id_list = sorted([id_ for id_ in range(self.population_size)],
                         key=lambda id_: self.population[id_].fitness)

        # individuals that are to be replaced by new ones
        weak_ids = id_list[:int(elimination_rate * self.population_size)]

        # elitism
        immune_ids = []
        for sp in self.species:
            if len(sp.members_ids) >= 5:
                immune_ids.append(
                    max(sp.members_ids, key=lambda id_: self.population[id_].fitness))

        ids_for_elites = weak_ids[:len(immune_ids)]
        ids_for_new_individuals = weak_ids[len(immune_ids):]

        # Total fitness of each species

        sp_potential_list = []
        for sp in self.species:
            fitness = sum(
                [self.population[id_].fitness for id_ in sp.members_ids])
            potential = fitness * (1 - sp.stagnation / max_stagnation)
            sp_potential_list.append(potential)
        total_potential = max(1, sum(sp_potential_list))

        # number of offsprings per species
        for p in sp_potential_list:
            print((p / total_potential) * len(ids_for_new_individuals))
        number_offspring_list = [
            int((p / total_potential) * len(ids_for_new_individuals)) for p in sp_potential_list]

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
                    ids_for_new_individuals[j], sp, weak_ids)
                j += 1

        # others
        for id_ in range(self.population_size):
            if not new_population[id_]:
                self.mutate(self.population[id_])
                new_population[id_] = self.population[id_]

        self.population = new_population

        for nn in self.population:
            nn.generateNetwork()

        self.generation_nb += 1


# FUNCTIONS


def distance(nn1: NeuralNetwork, nn2: NeuralNetwork):
    """calculates the distance between two neural networks
        O (|connexion|) """
    disjoint_genes_count = 0
    recursive_disjoint_genes_count = 0
    common_gene_count = 0
    common_recursive_gene_count = 0
    average_weight_difference = 0
    average_recursive_weight_difference = 0

    # Normal connexions
    for c1 in nn1.connexions:
        if c1 in nn2.connexions:
            common_gene_count += 1
            average_weight_difference += abs(
                nn1.connexions[c1].weight - nn2.connexions[c1].weight)

    # Disjoint genes count calculation
    disjoint_genes_count += (len(nn1.connexions) +
                             len(nn2.connexions) - 2 * common_gene_count)

    # Recursive connexions
    for c1 in nn1.recursive_connexions:
        if c1 in nn2.recursive_connexions:
            common_recursive_gene_count += 1
            average_recursive_weight_difference += abs(
                nn1.recursive_connexions[c1].weight - nn2.recursive_connexions[c1].weight)

    # Disjoint genes count calculation
    recursive_disjoint_genes_count += (len(nn1.recursive_connexions) + len(
        nn2.recursive_connexions) - 2 * common_recursive_gene_count)

    # Distance calculation
    n = max(1, disjoint_genes_count + common_gene_count)
    n_rec = max(1, recursive_disjoint_genes_count +
                common_recursive_gene_count)

    # Calculation of the proportions
    disjoint = disjoint_genes_count / n
    recursive_disjoint = recursive_disjoint_genes_count / n_rec

    average_weight_difference /= max(1, common_gene_count)
    average_recursive_weight_difference /= max(1, common_recursive_gene_count)

    d = (disjoint_coeff * disjoint
         + average_weight_coeff * average_weight_difference
         + recursive_disjoint_coeff * recursive_disjoint
         + average_recursive_weight_coeff * average_recursive_weight_difference)

    return d


def buildDistanceMatrix(population):
    """Calculates the distance matrix for the current population
       O(|population|^2 * (|connexion_catalog| + |recursive_connexion_catalog|)"""

    distance_matrix = [[0] * len(population)
                       for _ in range(len(population))]

    i = 0
    while i < len(population):
        j = 0
        while j < i:  # no need to do i=j (=> d = 0)
            d = distance(population[i], population[j])
            distance_matrix[i][j] = d
            distance_matrix[j][i] = d
            j += 1
        i += 1

    return distance_matrix


def isSameConnexion(c1: Connexion, c2: Connexion):
    if c1.i == c2.i and c1.o == c2.o:
        return True
    return False


def sharingFunction(id1, id2, distance_matrix):
    d = distance_matrix[id1][id2]
    if d < same_species_threshold:
        return 1 - (d / same_species_threshold) ** squaring_factor
    return 0
