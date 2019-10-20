from feature_selection import FeatureSelection
from dimensionality_reduction import DimensionalityReduction

class FeatureEngineering(FeatureSelection, DimensionalityReduction):
    """
    This will help us apply feature engineering on the data once it has been passed through
    DataEngineering:
        1. Feature Engineering using Domain Knowledge:
            a. Trying different permutations and combinations of features
            b.
        2. Important Feature Sekection
        3. Dimensionality Reduction

    """
    def __init__(self, arg):
        super(FeatureEngineering, self).__init__()
        self.arg = arg
