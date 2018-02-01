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
from numpy import tanh


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
        self.w = w

        # Allows to desactivate a connexion
        # it is better to desactivate a connexion than removing it
        # in order to keep track of the history of the evolution
        self.is_active = True


class Neuron:
    """A very basic neuron which has a value, an id, and a list of input"""

    # pylint: disable=too-few-public-methods
    # Using dictionnaries here would a tiny bit more efficient
    # but classes keeps the code clean and homegeneous

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


class NeuralNetwork:
    """A neural network which is a genome from the genetic algorithm point of view"""

    def __init__(self, nb_input, nb_output, activation_funtion=tanh):
        self.nb_input = 1 + nb_input  # Adding a bias node
        self.nb_output = nb_output
        self.activation_funtion = activation_funtion

        self.neurons = {}
        self.connexions = {}

        # recursive connexion must be treated seprately
        self.recursive_connexions = {}

        self.neurons[0] = Neuron(0)
        self.neurons[0].value = 1  # bias node always set to 1

        # Connection all the input node to the output node
        # Minimal strructure of the NN
        connexion_id = 0
        for e in range(self.nb_input):
            for i in range(self.nb_output):
                self.connexions[connexion_id] = Connexion(
                    connexion_id, e, i, 1)
                connexion_id += 1

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

        for c in self.connexions.values():
            if c.is_active:
                # Checking the existence of the neurons
                if not c.i in self.neurons:  # O(1)
                    self.neurons[c.i] = Neuron(c.i)
                if not c.o in self.neurons:
                    self.neurons[c.o] = Neuron(c.o)

                # Connecting the neurons
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

        # Storing the values that are useful for the recursive connexions
        temp = {}
        for c in self.recursive_connexions.values():
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
           by evluating its conexions"""
        neuron = self.neurons[neuron_id]  # Pointer for faster access and cleaner code
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

    def __init__(self, nb_input=2, nb_output=1, poulation_size=150, max_nb_neurons=30,
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
        self.latest_connexion_id = (1 + nb_input) * nb_output

        self.recursive_connexion_catalog = {}
        self.latest_recursive_connexion_id = -1

        self.poulation_size = poulation_size
        self.population = []
        self.species = []

        self.nouveaux_genes = []

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
                          if id2 != neuron_id1 and not neuron_id1 in nn.neurons[id2].input_list]
            if candidates:
                neuron_id2 = choice(candidates)
                # No need to check if the connexion is well-oriented

        else:
            neuron_id2 = choice(nn.neurons.keys())
            candidates = [id1 for id1 in nn.neurons
                          if id1 != neuron_id2 and not id1 in nn.neurons[neuron_id2].input_list]

            if candidates:
                neuron_id1 = choice(candidates)

                # Making sure the connexion is well oriented
                if not isForward(neuron_id1, neuron_id2, nn.neurons):
                    neuron_id1, neuron_id2 = neuron_id2, neuron_id1

        if candidates:
            # Checking if this connexion already exists
            connexion = Connexion(-1, neuron_id1,
                                  neuron_id2, 2 * random() - 1)

            self.setConnexionId(connexion)
            nn.connexions[connexion.id_] = connexion

    def newRecusiveConnexion(self, nn: NeuralNetwork, force_input=False):
        """Creates a connexion between two unconnected neurons the recursive way
        force_input forces a connexion to one of the input nodes
        O(|neurons| + |recursive_connexions| + |recursive_connexion_catalog|)"""

        if force_input:
            neuron_id2 = choice(range(nn.nb_input))
        else:
            neuron_id2 = choice(nn.neurons.keys())

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

        candidates = [c for c in nn.connexions.values if c.is_active]
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


# FUNCTIONS

# Shaping functions


def isSameConnexion(c1, c2):
    if c1.i == c2.i and c1.o == c2.o:
        return True
    return False


def isForward(neuron_id1, neuron_id2, neurons: dict):
    """Makes sure a feed forward connexion should go from id1 to id2
       return True if yes, False if no
       O(|connexions|)"""

    for i in neurons[neuron_id2].input_list:
        if neuron_id2 == i or not isForward(neuron_id1, i, neurons):
            return False

    return True


# Genetic function

def distance(nn1: NeuralNetwork, nn2: NeuralNetwork):
    connexion_catalog = {}
    pass


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
