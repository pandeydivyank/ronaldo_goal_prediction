from training import Training
from testing import Testing
from model_architectures import ModelArchitectures

class Model(ModelArchitectures, Training, Testing):
    """
    Employ a model
    """
    def __init__(self, arg):
        super(Model, self).__init__()
        self.arg = arg
