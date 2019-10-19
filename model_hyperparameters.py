from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class ModelHyperparameters:

    def __init__(self, train_data, validation_data, nfolds = 10, n_jobs = -1, verbose = False):
        self.train_data = train_data
        self.validation_data = validation_data
        self.n_jobs = n_jobs
        self.nfolds = nfolds
        self.verbose = verbose

    def svm_hyperparameter_dictionary(self):

        Cs = [10, 20, 30]
        gammas = [10, 20, 30]
        probabilities = [True, False]
        degrees = [3]
        shrinkings = [True, False]

        param_grid = {
        # 'probability' : probabilities,
        # 'shrinking' : shrinkings,
        'degree' : degrees,
        'C': Cs,
        'gamma' : gammas
        }

        return param_grid

    def random_forest_hyperparameter_dictionary(self):

        bootstraps = [True]
        max_depths = [10, 20, 30]
        max_featuress = [3, 4, 5]
        min_samples_leafs = [1, 2, 3]
        min_samples_splits = [2, 3, 4]
        n_estimatorss = [300]

        param_grid = {
            'bootstrap': bootstraps,
            'max_depth': max_depths,
            'max_features': max_featuress,
            'min_samples_leaf': min_samples_leafs,
            'min_samples_split': min_samples_splits,
            'n_estimators': n_estimatorss
        }

        return param_grid

    def svm_hyperparameter_tuning(self, kernel = 'rbf', param_grid = None):
        if param_grid is None:
            param_grid = self.svm_hyperparameter_dictionary()

        grid_search = GridSearchCV(SVC(kernel = kernel), param_grid, cv = self.nfolds, n_jobs = self.n_jobs, verbose = self.verbose)
        grid_search.fit(self.train_data['X'], self.train_data['y'])

        print('SVM BEST PARAMETERS: \n\n{}\n'.format(grid_search.best_params_))

        return grid_search.best_estimator_

    def random_forest_hyperparameter_tuning(self, param_grid = None):
        if param_grid is None:
            param_grid = self.random_forest_hyperparameter_dictionary()

        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv = self.nfolds, n_jobs = self.n_jobs, verbose = self.verbose)
        grid_search.fit(self.train_data['X'], self.train_data['y'])

        print('RANDOM FOREST BEST PARAMETERS: \n\n{}\n'.format(grid_search.best_params_))

        return grid_search.best_estimator_
