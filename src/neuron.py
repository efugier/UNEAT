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
