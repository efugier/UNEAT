from copy import deepcopy
from random import random, choice
from ga import SpawingPool
from neuralnetwork import NeuralNetwork

from parameters import nb_input, nb_output, crossover_rate


class Species:
    def __init__(self, sp: SpawingPool, members_ids=None):
        self.potential = 0
        self.stagnation = 0
        self.is_stale = False
        self.representative_id = None

        self.spawing_pool = sp  # the Spawing Pool it belongs to

        if not members_ids:
            self.members_ids = []
        else:
            self.members_ids = members_ids

    def offspring(self, nb_offspring):
        r""" returns a list of nb_offspring offsprings
             /!\ they are id-less """
        res = [NeuralNetwork(-1, 2, 3) for _ in range(nb_offspring)]
        for _ in range(int(nb_offspring * crossover_rate)):
            pass
        for _ in range(int(nb_offspring * crossover_rate), nb_offspring):
            pass
        return res


def mate(nn1: NeuralNetwork, nn2: NeuralNetwork):
    """ creates the child of nn1 and nn2 """

    # Make sure nn1 is the fitest
    if nn1.fitness < nn2.fitness:
        nn1, nn2 = nn2, nn1

    new_nn = NeuralNetwork(-1, nb_input, nb_output)

    # Normal connexions
    for c_id in nn1.connexions:
        if c_id in nn2.connexions:
            if random() < 0.5:
                new_nn.connexions[c_id] = deepcopy(
                    nn1.connexions[c_id])
            else:
                new_nn.connexions[c_id] = deepcopy(
                    nn2.connexions[c_id])

            # Activation of the fitest parent to prevent neuron disconnexion (very important)
            if nn1.connexions[c_id].is_active:
                new_nn.connexions[c_id].is_active = True

        # Inheriting dijoints genes from the fitest parent
        else:
            new_nn.connexions[c_id] = deepcopy(nn1.connexions[c_id])

    # Recursive Connexions
    for c_id in nn1.connexions:
        if c_id in nn2.recursive_connexions:
            if random() < 0.5:
                new_nn.recursive_connexions[c_id] = deepcopy(
                    nn1.recursive_connexions[c_id])
            else:
                new_nn.recursive_connexions[c_id] = deepcopy(
                    nn2.recursive_connexions[c_id])

        # Inheriting dijoints genes from the most fit parent
        elif c_id in nn1.recursive_connexions:
            new_nn.recursive_connexions[c_id] = deepcopy(
                nn1.recursive_connexions[c_id])

    return new_nn
