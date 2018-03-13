from random import random
from numpy import exp

from connexion import Connexion
from neuron import Neuron


def sigmoid(x):
    return 1 / (1 + exp(-4.9 * x))

# def sigmoid(x):
#     return tanh(3 * x)


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
