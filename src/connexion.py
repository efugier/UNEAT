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
