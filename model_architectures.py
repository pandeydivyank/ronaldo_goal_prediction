from model_hyperparameters import ModelHyperparameters
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

class ModelArchitectures(ModelHyperparameters):


    def svm_classifier(self, kernel = 'rbf', degree = 3, param_grid = None):
        '''
        kernel_types:  ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
        '''

        # print(self.svm_hyperparameter_tuning())

        # model = SVC(kernel = kernel, degree = degree, gamma = gamma)

        model = self.svm_hyperparameter_tuning(kernel = kernel, param_grid = param_grid)
        model.fit(self.train_data['X'], self.train_data['y'])

        y_train_pred = model.predict(self.train_data['X'])
        train_matrix = confusion_matrix(self.train_data['y'], y_train_pred)

        print("\n\n\nSVM TRAIN: \n\n {} \n\n\n".format(train_matrix))

        y_pred = model.predict(self.validation_data['X'])
        matrix = confusion_matrix(self.validation_data['y'], y_pred)

        print("\n\n\nSVM TEST: \n\n {} \n\n\n".format(matrix))

        return matrix

    def random_forest_classifier(self, param_grid = None):

        # model = RandomForestClassifier()

        model = self.random_forest_hyperparameter_tuning(param_grid = param_grid)
        model.fit(self.train_data['X'], self.train_data['y'])

        y_train_pred = model.predict(self.train_data['X'])
        train_matrix = confusion_matrix(self.train_data['y'], y_train_pred)

        print("\n\n\nRandom Forest TRAIN: \n\n {} \n\n\n".format(train_matrix))

        y_pred = model.predict(self.validation_data['X'])
        matrix = confusion_matrix(self.validation_data['y'], y_pred)

        print("\n\n\nRandom Forest TEST: \n\n {} \n\n\n".format(matrix))

        return matrix, model
