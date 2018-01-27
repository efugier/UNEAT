""" TODO:
2 step evaluation process:
    -1 normal connexions
    -2 recursives connexions

Neurone € dictionnary
Connections
"""

from copy import deepcopy
from pickle import Pickler, Unpickler

from random import random, choice
from numpy import tanh


# CLASSES

class Connexion:
    """The connexion are the genes from the genetic algorithm point of view"""

    # pylint: disable=too-few-public-methods
    # Using dictionnaries here would a tiny bit more efficient
    # but classes keeps the code homegeneous

    def __init__(self, i, o, w=0):
        """i is the ID of the input neuron
           o is the ID of the output neuron
           w is the weight of the connexion"""
        self.i = i
        self.o = o
        self.w = w


class Neuron:
    """A very basic neuron which has a value, an id, and a list of input"""

    # pylint: disable=too-few-public-methods
    # Using dictionnaries here would a tiny bit more efficient
    # but classes keeps the code homegeneous

    def __init__(self, id_, input_list=None):
        # List of the ID of the neurones that are connected to this one
        if not input_list:
            self.input_list = []
        else:
            self.input_list = input_list

        self.value = 0
        self.id_ = id_

        # Important for the evaluation of the network
        self.already_evaluated = False

        # Allows to desactivate a connexion
        # it is better to desactivate a connexion than removing it
        # in order to keep track of the history of the evolution
        self.is_active = True


class NeuralNetwork:
    """A neural network which is a genome from the genetic algorithm point of view"""

    def __init__(self, nb_input, nb_output, activation_funtion=tanh):
        self.nb_input = 1 + nb_input  # Adding a bias node
        self.nb_output = nb_output
        self.activation_funtion = activation_funtion

        self.neurons = {}
        self.connexions = []

        # recursive connexion must be treated seprately
        self.recursive_connexions = []

        self.neurons[0] = Neuron(0)
        self.neurons[0].value = 1  # bias node always set to 1

        # Connection all the input node to the output node
        # Minimal strructure of the NN
        for e in range(1, self.nb_input):
            for i in range(self.nb_output):
                self.connexions.append(Connexion(e, i, 1))

    def generate_netork(self):
        """generates the neural network using its connexion list
            O(|self.connexions|)"""
        self.neurons = {}

        self.neurons[0] = Neuron(0)
        self.neurons[0].value = 1  # bias node always set to 1

        for e in range(1, self.nb_input):
            self.neurons[e] = Neuron(e)
        for i in range(self.nb_input, self.nb_output):
            self.neurons[i] = Neuron(i)

        for c in self.connexions:
            if c.is_active:
                # Checking the existence of the neurons
                if not c.i in self.neurons:  # O(1)
                    self.neurons[c.i] = Neuron(c.i)
                if not c.o in self.neurons:
                    self.neurons[c.o] = Neuron(c.o)

                # Connexion
                if not c.i in self.neurons[c.o].input_list:
                    self.neurons[c.o].input_list.append(c.i)

    def evaluate_network(self, input_vector):
        """evaluates the neural network
           basically a depth first search
           O(|self.connexions|)"""

        # Setting the input values
        self.neurons[0].value = 1  # Bias node

        for i in range(1, self.nb_input):
            self.neurons[i].value = input_vector[i]
            self.neurons[i].already_evaluated = True

        # Evaluating the NN
        res = [self.evaluate_neuron(o)
               for o in range(self.nb_input, self.nb_output)]

        # Storing the useful values for the recursive connexions
        temp = {}
        for c in self.recursive_connexions:
            temp[c.o] = self.neurons[c.i].value

        # Reseting the network
        for neuron_id in self.neurons:  # Resets the network
            self.neurons[neuron_id].value = 0
            self.neurons[neuron_id].already_evaluated = False

         # Applying the recursive connexions
        for neuron_id in temp:
            self.neurons[neuron_id].value = temp[neuron_id]

        return res

    def evaluate_neuron(self, neuron_id):
        """Recursive function that evaluates a neuron
           By evluating its conexions"""
        # Should work because it would be a pointer the neuron
        # Needs to be tested though
        neuron = self.neurons[neuron_id]
        if neuron.already_evaluated:
            return neuron.value

        s = neuron.value
        for input_neuron_id in neuron.input_list:
            s += self.evaluate_neuron(input_neuron_id)

        neuron.value = self.activation_funtion(s)
        neuron.already_evaluated = True

        return neuron.value


class SpawingPool:
    """The class which supervises the evolution"""

    # pylint: disable=too-many-instance-attributes
    # This class' very purpose is to encapsulate the evolution's variables

    def __init__(self, nb_input=2, nb_output=1, poulation_size=150, max_nb_neurons=30):
        self.nb_input = nb_input
        self.nb_output = nb_output
        self.max_nb_neurons = max_nb_neurons

        self.generation_nb = 0

        self.best_individual = None
        self.max_fitness = 0

        self.latest_neuron_id = nb_input * nb_output - 1

        self.poulation_size = poulation_size
        self.population = []
        self.species = []

        self.nouveaux_genes = []


# FUNCTIONS

# Shaping functions

def isForward(neuron_id1, neuron_id2, neurons):
    """Makes sure a feed forward connexion should go from id1 to id2
       return True if yes, False if no
       O(|connexions|)"""

    for i in neurons[neuron_id2].input_list:
        if neuron_id2 == i or not isForward(neuron_id1, i, neurons):
            return False

    return True


def newConnexion(nn: NeuralNetwork, force_input=False):
    """Creates a connexion between two unconnected neurons the feed forward way
       force_input forces a connexion from one of the input nodes
       O(|neurons|^2)"""

    if force_input:
        neuron_id1 = choice(range(nn.nb_input))
        candidates = [id1 for id1 in nn.neurons
                      if id1 != neuron_id1 and not neuron_id1 in nn.neurons[id1].input_list]
        if candidates:
            neuron_id2 = choice(candidates)
            nn.connexions.append(
                Connexion(neuron_id1, neuron_id2, 2 * random() - 1))

    else:
        neuron_id2 = choice(nn.neurons.keys())
        candidates = [id1 for id1 in nn.neurons
                      if id1 != neuron_id2 and not id1 in nn.neurons[neuron_id2].input_list]

        if candidates:
            neuron_id1 = choice(candidates)

            # Making sure the connexion is well oriented
            if not isForward(neuron_id1, neuron_id2, nn.neurons):
                neuron_id1, neuron_id2 = neuron_id2, neuron_id1

            nn.connexions.append(
                Connexion(neuron_id1, neuron_id2, 2 * random() - 1))


def newRecusiveConnexion(nn: NeuralNetwork, force_input=False):
    """Creates a connexion between two unconnected neurons the recursive way
       force_input forces a connexion to one of the input nodes
       O(|neurons|*|recursive_connexions|)"""

    if force_input:
        neuron_id2 = choice(range(nn.nb_input))
    else:
        neuron_id2 = choice(nn.neurons.keys())

    candidates = []
    for id1 in nn.neurons:
        for c in nn.recursive_connexions:
            if not (c.i == id1 and c.o == neuron_id2):
                candidates.append(id1)

    if candidates:
        neuron_id1 = choice(candidates)

        nn.connexions.append(
            Connexion(neuron_id1, neuron_id2, 2 * random() - 1))


# Service functions

def save(obj, file_name='latest_NN'):
    with open(file_name, 'wb') as file_:
        pickler = Pickler(file_)
        pickler.dump(obj)

    print("Saved as ", file_name)


def load(file_name='latest_NN'):
    with open(file_name, 'rb') as file_:
        unpickler = Unpickler(file_)
        obj = unpickler.load()
    return obj
