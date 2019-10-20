from training import Training
from testing import Testing

class Model(Training, Testing):
    """
    Employ a model
    """
    def __init__(self, arg):
        super(Model, self).__init__()
        self.arg = arg
