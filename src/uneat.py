""" TODO:
2 step evaluation process:
    -1 normal connexions
    -2 recursives connexions

Neurone € dictionnary
Connections
"""

MAX_NB_NEURON = 1000


class Connexion:
    """The connexion are the genes from the genetic algorithm point of view"""

    def __init__(self, i, o, w=0):
        """i is the ID of the input neuron
           o is the ID of the output neuron
           w is the weight of the connexion"""
        self.i = i
        self.o = o
        self.w = w


class Neuron:
    """A very basic neuron which has a value, an id, and a list of input"""

    def __init__(self, id_, input_list=None):
        # List of the ID of the neurones that are connected to this one
        if not input_list:
            self.input_list = []
        else:
            self.input_list = input_list

        self.value = 0
        self.id_ = id_

        # Allows to desactivate a connexion
        # it is better to desactivate a connexion than removing it
        # in order to keep track of the history of the evolution
        self.is_active = True


class NeuralNetwork:
    """A neural network which is a genome from the genetic algorithm point of view"""

    def __init__(self, nb_input, nb_output):
        self.nb_input = nb_input
        self.nb_output = nb_output

        self.neurons = {}
        self.connexions = []

        # recursive connexion must be treated seprately
        self.recursive_connexions = []

    def generate_netork(self):  # O(|self.connexions|)
        """generates the neural network using its connexion list"""
        self.neurons = {}

        for e in range(self.nb_input):
            self.neurons[e] = Neuron(e)
        for i in range(self.nb_input, self.nb_output):
            self.neurons[i] = Neuron(i)

        for c in self.connexions:
            if c.is_active:
                # Checkingxistence the existence of the neurons
                if not c.i in self.neurons:  # O(1)
                    self.neurons[c.i] = Neuron(c.i)
                if not c.o in self.neurons:
                    self.neurons[c.o] = Neuron(c.o)

                # Connexion
                if not c.i in self.neurons[c.o].input_list:
                    self.neurons[c.o].input_list.append(c.i)


class SpawingPool:
    """The class which supervise the evolution"""

    def __init__(self, nb_input=2, nb_output=1, poulation_size=150, max_nb_neurons=30):
        self.nb_input = nb_input
        self.nb_output = nb_output
        self.max_nb_neurons = max_nb_neurons

        self.generation_nb = 0

        self.best_individual = None
        self.max_fitness = -1

        self.latest_neuron_id = nb_input * nb_output - 1

        self.poulation_size = poulation_size
        self.population = []
        self.species = []

        self.nouveaux_genes = []
