"""
TODO:
- update the new generation function
"""

from pickle import Pickler, Unpickler
from connexion import Connexion
from neuralnetwork import NeuralNetwork
from ga import SpawingPool

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

def evalXOR(nn_list):
    pat = [[[0, 0], [0]],
           [[0, 1], [1]],
           [[1, 0], [1]],
           [[1, 1], [0]]]

    res = []
    for nn in nn_list:
        fit = 4 - sum([(nn.evaluateNetwork(i)[0] - o[0])**2 for i, o in pat])
        res.append(fit)

    return res


def printXOR(nn: NeuralNetwork):
    pat = [[[0, 0], [0]],
           [[0, 1], [1]],
           [[1, 0], [1]],
           [[1, 1], [0]]]
    for p in pat:
        print(p[0], '->', (nn.evaluateNetwork(p[0])))


def solveXOR():
    spawing_pool = SpawingPool(evaluation_function=evalXOR)

    spawing_pool.initPopulation()


    # nn = NeuralNetwork(42, 2, 1)
    # nn.connexions = {0: Connexion(0, 0, 3, -0.5),
    #                  1: Connexion(1, 1, 3, 1),
    #                  2: Connexion(2, 2, 3, 1),
    #                  3: Connexion(3, 1, 4, 1),
    #                  4: Connexion(4, 2, 4, 1),
    #                  5: Connexion(5, 0, 4, -1.7),
    #                  6: Connexion(6, 4, 3, -2)}
    # nn.generateNetwork()
    # nn2 = NeuralNetwork(31, 2, 1)

    # print("distance :    ", spawing_pool.distance(nn, nn2))

    # spawing_pool.population[42] = nn

    for _ in range(50):
        spawing_pool.newGeneration()
        print(len(spawing_pool.species))
        print(spawing_pool.best_individual.fitness)
        # spawing_pool.best_individual.display()
        printXOR(spawing_pool.best_individual)



def test():
    nn = NeuralNetwork(0, 2, 1)
    nn.connexions = {0: Connexion(0, 0, 3, -0.5),
                     1: Connexion(1, 1, 3, 1),
                     2: Connexion(2, 2, 3, 1),
                     3: Connexion(3, 1, 4, 1),
                     4: Connexion(4, 2, 4, 1),
                     5: Connexion(5, 0, 4, -1.7),
                     6: Connexion(6, 4, 3, -2)}
    nn.generateNetwork()
    print(len(nn.connexions))
    nn.display()
    printXOR(nn)


solveXOR()
# test()
